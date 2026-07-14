"""Train the frozen contextual_physics_v1 candidate on natural fit flights.

This runner performs no anomaly threshold selection and never reads truth-v2,
development, rehearsal, raw tar, archive, Downloads, or the blind holdout pool.
The Step-5 manifest is used only as an immutable inventory/split source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.context import CausalContextConfig  # noqa: E402
from adsb.contextual_scaling import (  # noqa: E402
    NATURAL_FIT_ROLE,
    StrictNaturalRobustScaler,
    StrictScalingConfig,
)
from adsb.contextual_windowing import (  # noqa: E402
    ContextualForecastBatch,
    build_contextual_forecast_windows,
)
from adsb.features import (  # noqa: E402
    VECTOR_RESIDUAL_FEATURES,
    build_feature_table,
)
from adsb.models.contextual_residual_forecaster import (  # noqa: E402
    ContextualForecasterConfig,
    ContextualResidualForecaster,
    channelwise_gaussian_nll,
    contextual_channel_scores,
    weighted_masked_channel_loss,
)
from adsb.rules import RULE_CHANNELS  # noqa: E402
from adsb.segmentation import segment_flights  # noqa: E402


FIT_DAY = "2026-02-28"
SEGMENT_GAP_S = 1800.0
SILVER_COLUMNS = (
    "_source_file",
    "source_id",
    "timestamp_utc",
    "lat",
    "lon",
    "alt",
    "alt_geom_m",
    "on_ground",
    "ground_speed_ms",
    "track_deg",
    "vertical_rate_ms",
)
ALL_RESIDUAL_CHANNELS = tuple(RULE_CHANNELS) + tuple(VECTOR_RESIDUAL_FEATURES)
CODE_PATHS = (
    "adsb/context.py",
    "adsb/contextual_scaling.py",
    "adsb/contextual_windowing.py",
    "adsb/models/contextual_residual_forecaster.py",
    "adsb/features.py",
    "adsb/segmentation.py",
    "scripts/adsb_train_contextual_physics_v1.py",
)


class ContextualTrainingContractError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_state(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty_lines = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    if dirty_lines:
        raise ContextualTrainingContractError(
            f"Training requires a clean tracked worktree; dirty={dirty_lines[:10]}"
        )
    return {"commit": commit, "tracked_worktree_clean": True}


def _code_hashes(repo_root: Path) -> dict[str, str]:
    return {relative: _sha256_file(repo_root / relative) for relative in CODE_PATHS}


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("candidate_namespace") != "contextual_physics_v1":
        raise ContextualTrainingContractError("Unexpected candidate namespace")
    if config["data"].get("synthetic_training_rows_required") != 0:
        raise ContextualTrainingContractError("Synthetic training must be exactly zero")
    if config["window"].get("target_horizon_rows") != 1:
        raise ContextualTrainingContractError("Only one-row-ahead training is implemented")
    if not all(config["selection_prohibitions"].values()):
        raise ContextualTrainingContractError("Every selection prohibition must be true")
    return config


def _verify_source_manifest(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = (repo_root / config["source_step5_manifest"]).resolve(strict=True)
    expected = config["source_step5_manifest_sha256"]
    observed = _sha256_file(path)
    if observed != expected:
        raise ContextualTrainingContractError(
            f"Step-5 source manifest hash changed: expected={expected} observed={observed}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits = manifest["split_contract"]["splits"]
    required_roles = {"fit", "calibration", "validation", "development", "rehearsal"}
    if set(splits) != required_roles:
        raise ContextualTrainingContractError("Source split roles are incomplete")
    fit_inputs = [record for record in manifest["inputs"] if record["role"] == "fit"]
    if len(fit_inputs) != 237:
        raise ContextualTrainingContractError("Expected exactly 237 fit-day Silver inputs")
    return {"path": path, "manifest": manifest, "fit_inputs": fit_inputs}


def _stable_uniform(seed: int, purpose: str, value: str) -> float:
    payload = f"{seed}\0{purpose}\0{value}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


def _sample_flights(
    flight_ids: Iterable[str], *, probability: float, seed: int, purpose: str
) -> tuple[str, ...]:
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ContextualTrainingContractError("Flight sample probability must be in (0, 1]")
    selected = tuple(
        flight_id
        for flight_id in sorted(map(str, flight_ids))
        if _stable_uniform(seed, purpose, flight_id) < probability
    )
    if not selected:
        raise ContextualTrainingContractError(f"No flights selected for {purpose}")
    return selected


def _source_id(flight_id: str) -> str:
    if not flight_id.startswith(f"{FIT_DAY}:") or "_" not in flight_id:
        raise ContextualTrainingContractError(f"Unexpected fit flight ID: {flight_id!r}")
    return flight_id.split(":", 1)[1].rsplit("_", 1)[0]


def _verify_fit_inputs(repo_root: Path, fit_inputs: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for record in fit_inputs:
        path = (repo_root / record["path"]).resolve(strict=True)
        forbidden = {part.lower() for part in path.parts} & {"archive", "downloads", "raw"}
        if forbidden:
            raise ContextualTrainingContractError(f"Forbidden input path: {path}")
        stat = path.stat()
        if stat.st_size != int(record["bytes"]):
            raise ContextualTrainingContractError(f"Input byte size changed: {path}")
        if _sha256_file(path) != record["sha256"]:
            raise ContextualTrainingContractError(f"Input SHA-256 changed: {path}")
        paths.append(path)
    return paths


def _selected_features(path: Path, selected_flights: set[str], selected_sources: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(SILVER_COLUMNS))
    frame = frame.loc[frame["source_id"].astype(str).isin(selected_sources)].copy()
    if frame.empty:
        return pd.DataFrame()
    segmented = segment_flights(frame, gap_s=SEGMENT_GAP_S)
    segmented["flight_id"] = segmented["flight_id"].map(lambda value: f"{FIT_DAY}:{value}")
    features = build_feature_table(segmented)
    time_values = pd.to_numeric(features["timestamp_utc"], errors="coerce")
    dt_s = time_values - time_values.shift(1)
    transition_valid = (
        features["flight_id"].eq(features["flight_id"].shift(1))
        & dt_s.gt(0)
        & dt_s.le(60.0)
    ).fillna(False)
    features.loc[~transition_valid, list(ALL_RESIDUAL_CHANNELS)] = np.nan
    return features.loc[features["flight_id"].isin(selected_flights)].reset_index(drop=True)


def _iter_selected_features(
    paths: Iterable[Path], selected_flights: tuple[str, ...]
) -> Iterable[tuple[Path, pd.DataFrame]]:
    flight_set = set(selected_flights)
    source_set = {_source_id(flight_id) for flight_id in selected_flights}
    for path in paths:
        features = _selected_features(path, flight_set, source_set)
        if not features.empty:
            yield path, features


def _fit_scaler(
    paths: list[Path], selected_flights: tuple[str, ...], channels: tuple[str, ...], clip: float
) -> tuple[StrictNaturalRobustScaler, dict[str, Any]]:
    channel_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    row_count = 0
    part_count = 0
    for _, features in _iter_selected_features(paths, selected_flights):
        row_count += len(features)
        part_count += 1
        for channel in channels:
            values = pd.to_numeric(features[channel], errors="coerce").to_numpy(float)
            channel_parts[channel].append(values[np.isfinite(values)])
    fit_frame = pd.DataFrame(
        {
            channel: pd.Series(np.concatenate(parts) if parts else np.array([], dtype=float))
            for channel, parts in channel_parts.items()
        }
    )
    scaler = StrictNaturalRobustScaler(StrictScalingConfig(clip=clip)).fit(
        fit_frame,
        channels,
        data_role=NATURAL_FIT_ROLE,
        contains_synthetic=False,
    )
    return scaler, {
        "selected_fit_rows_seen": row_count,
        "selected_fit_parts_with_rows": part_count,
        "finite_values_by_channel": {
            channel: int(sum(len(values) for values in parts))
            for channel, parts in channel_parts.items()
        },
    }


def _context_config(config: dict[str, Any]) -> CausalContextConfig:
    values = config["context"]
    return CausalContextConfig(
        phase_history_rows=int(values["phase_history_rows"]),
        level_rate_threshold_mps=float(values["level_rate_threshold_mps"]),
        cadence_edges_s=tuple(map(float, values["cadence_edges_s"])),
        max_gap_s=float(values["max_gap_s"]),
    )


def _make_batch(
    features: pd.DataFrame,
    *,
    scaler: StrictNaturalRobustScaler,
    config: dict[str, Any],
) -> ContextualForecastBatch:
    scaled = features.copy()
    transformed = scaler.transform(features)
    for channel in scaler.active_channels:
        scaled[channel] = transformed[channel]
    return build_contextual_forecast_windows(
        scaled,
        signal_columns=scaler.active_channels,
        target_channels=scaler.active_channels,
        history_rows=int(config["window"]["history_rows"]),
        context_config=_context_config(config),
    )


def _model_config(config: dict[str, Any], batch: ContextualForecastBatch) -> ContextualForecasterConfig:
    values = config["model"]
    return ContextualForecasterConfig(
        input_features=len(batch.input_features),
        target_channels=len(batch.target_channels),
        hidden_size=int(values["hidden_size"]),
        num_layers=int(values["num_layers"]),
        min_scale=float(values["min_scale"]),
        max_scale=float(values["max_scale"]),
    )


def _train(
    paths: list[Path],
    selected_flights: tuple[str, ...],
    *,
    scaler: StrictNaturalRobustScaler,
    config: dict[str, Any],
    run_dir: Path,
) -> tuple[ContextualResidualForecaster, dict[str, Any]]:
    training = config["training"]
    torch.manual_seed(int(training["seed"]))
    model: ContextualResidualForecaster | None = None
    optimizer: torch.optim.Optimizer | None = None
    weights: torch.Tensor | None = None
    input_features: tuple[str, ...] | None = None
    history: list[dict[str, Any]] = []
    derived_config_path = run_dir / "derived_training_config.json"

    for epoch in range(int(training["epochs"])):
        epoch_loss_sum = 0.0
        epoch_windows = 0
        epoch_batches = 0
        for _, features in _iter_selected_features(paths, selected_flights):
            batch = _make_batch(features, scaler=scaler, config=config)
            if len(batch.X) == 0:
                continue
            if model is None:
                model_config = _model_config(config, batch)
                model = ContextualResidualForecaster(model_config)
                optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
                weights = torch.ones(model_config.target_channels, dtype=torch.float32)
                input_features = batch.input_features
                derived = {
                    "model_config": model_config.__dict__,
                    "input_features": list(batch.input_features),
                    "target_channels": list(batch.target_channels),
                    "channel_weights": {channel: 1.0 for channel in batch.target_channels},
                    "scaler": scaler.to_dict(),
                    "frozen_before_first_optimizer_step": True,
                }
                _write_json_exclusive(derived_config_path, derived)
            elif batch.input_features != input_features:
                raise ContextualTrainingContractError("Input feature order changed between parts")

            assert model is not None and optimizer is not None and weights is not None
            permutation = torch.randperm(len(batch.X))
            for start in range(0, len(batch.X), int(training["batch_size"])):
                index = permutation[start : start + int(training["batch_size"])]
                xb = torch.from_numpy(batch.X[index])
                mb = torch.from_numpy(batch.X_mask[index])
                yb = torch.from_numpy(batch.y[index])
                ymb = torch.from_numpy(batch.y_mask[index])
                optimizer.zero_grad()
                location, scale = model(xb, mb)
                nll, _ = channelwise_gaussian_nll(yb, location, scale, ymb)
                loss = weighted_masked_channel_loss(nll, ymb, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(training["gradient_clip_norm"])
                )
                optimizer.step()
                epoch_loss_sum += float(loss.detach()) * len(index)
                epoch_windows += len(index)
                epoch_batches += 1
        if model is None or epoch_windows == 0:
            raise ContextualTrainingContractError("No scoreable training windows were produced")
        record = {
            "epoch": epoch + 1,
            "mean_weighted_gaussian_nll": epoch_loss_sum / epoch_windows,
            "windows": epoch_windows,
            "batches": epoch_batches,
        }
        history.append(record)
        print(
            f"epoch={record['epoch']} windows={epoch_windows} "
            f"mean_nll={record['mean_weighted_gaussian_nll']:.8f}",
            flush=True,
        )
    assert model is not None
    return model, {
        "epochs": history,
        "derived_training_config": derived_config_path.name,
        "derived_training_config_sha256": _sha256_file(derived_config_path),
    }


def _natural_diagnostics(
    model: ContextualResidualForecaster,
    paths: list[Path],
    selected_flights: tuple[str, ...],
    *,
    scaler: StrictNaturalRobustScaler,
    config: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(int(config["training"]["seed"]) + 1000)
    untrained = ContextualResidualForecaster(model.config)
    trained_scalar: list[np.ndarray] = []
    untrained_scalar: list[np.ndarray] = []
    target_magnitude: list[np.ndarray] = []
    channel_scores: dict[str, list[np.ndarray]] = defaultdict(list)
    windows = 0
    for _, features in _iter_selected_features(paths, selected_flights):
        batch = _make_batch(features, scaler=scaler, config=config)
        if len(batch.X) == 0:
            continue
        trained, _, _ = contextual_channel_scores(
            model, batch.X, batch.X_mask, batch.y, batch.y_mask
        )
        random_scores, _, _ = contextual_channel_scores(
            untrained, batch.X, batch.X_mask, batch.y, batch.y_mask
        )
        denominator = batch.y_mask.sum(axis=1).clip(min=1.0)
        trained_scalar.append(trained.sum(axis=1) / denominator)
        untrained_scalar.append(random_scores.sum(axis=1) / denominator)
        target_magnitude.append((np.abs(batch.y) * batch.y_mask).sum(axis=1) / denominator)
        for number, channel in enumerate(batch.target_channels):
            valid = batch.y_mask[:, number] > 0
            channel_scores[channel].append(trained[valid, number])
        windows += len(batch.X)
    if windows == 0:
        raise ContextualTrainingContractError("No calibration diagnostic windows were produced")
    trained_all = np.concatenate(trained_scalar)
    untrained_all = np.concatenate(untrained_scalar)
    magnitude_all = np.concatenate(target_magnitude)
    rho_untrained = float(spearmanr(trained_all, untrained_all).statistic)
    rho_magnitude = float(spearmanr(trained_all, magnitude_all).statistic)
    return {
        "role": "natural_calibration_diagnostic_only",
        "never_used_for_optimizer_or_threshold": True,
        "windows": windows,
        "rho_trained_vs_untrained": rho_untrained,
        "rho_trained_vs_target_magnitude": rho_magnitude,
        "magnitude_domination_flagged_at_0_8": bool(
            rho_untrained >= 0.8 or rho_magnitude >= 0.8
        ),
        "per_channel_standardized_surprise": {
            channel: {
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for channel, parts in channel_scores.items()
            for values in [np.concatenate(parts)]
        },
    }


def _write_checksums(run_dir: Path) -> None:
    files = []
    for path in sorted(run_dir.iterdir()):
        if path.name == "artifact_checksums.json" or not path.is_file():
            continue
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    _write_json_exclusive(
        run_dir / "artifact_checksums.json",
        {"schema_version": 1, "self_excluded": True, "files": files},
    )


def run(*, repo_root: Path, config_path: Path, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = repo_root.resolve(strict=True)
    destination = run_dir.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Run directory already exists: {destination}")
    if {part.lower() for part in destination.parts} & {"archive", "downloads", "raw"}:
        raise ContextualTrainingContractError("Run directory uses a forbidden path component")

    config_file = config_path.resolve(strict=True)
    config = _load_config(config_file)
    git_start = _git_state(root)
    code_start = _code_hashes(root)
    source = _verify_source_manifest(root, config)
    fit_paths = _verify_fit_inputs(root, source["fit_inputs"])
    split_records = source["manifest"]["split_contract"]["splits"]
    fit_ids = tuple(split_records["fit"]["flight_ids"])
    calibration_ids = tuple(split_records["calibration"]["flight_ids"])
    fit_selected = _sample_flights(
        fit_ids,
        probability=float(config["data"]["fit_flight_sample_probability"]),
        seed=int(config["data"]["fit_flight_sample_seed"]),
        purpose="contextual_physics_v1_fit",
    )
    calibration_selected = _sample_flights(
        calibration_ids,
        probability=float(config["data"]["calibration_diagnostic_sample_probability"]),
        seed=int(config["data"]["calibration_diagnostic_sample_seed"]),
        purpose="contextual_physics_v1_calibration_diagnostic",
    )
    if set(fit_selected) & set(calibration_selected):
        raise ContextualTrainingContractError("Fit and calibration diagnostic flights overlap")

    destination.mkdir(parents=True, exist_ok=False)
    run_manifest = {
        "schema_version": 1,
        "run_id": destination.name,
        "candidate_namespace": config["candidate_namespace"],
        "config_path": config_file.relative_to(root).as_posix(),
        "config_sha256": _sha256_file(config_file),
        "config_payload_sha256": _canonical_json_sha256(config),
        "source_step5_manifest": config["source_step5_manifest"],
        "source_step5_manifest_sha256": config["source_step5_manifest_sha256"],
        "git": git_start,
        "code_sha256": code_start,
        "fit_input_count": len(fit_paths),
        "fit_input_records_sha256": _canonical_json_sha256(source["fit_inputs"]),
        "split_contract_sha256": source["manifest"]["split_contract"]["contract_sha256"],
        "fit_flights_total": len(fit_ids),
        "fit_flights_selected": len(fit_selected),
        "fit_flight_ids_sha256": _canonical_json_sha256(list(fit_selected)),
        "calibration_flights_total": len(calibration_ids),
        "calibration_diagnostic_flights_selected": len(calibration_selected),
        "calibration_diagnostic_flight_ids_sha256": _canonical_json_sha256(
            list(calibration_selected)
        ),
        "synthetic_training_rows": 0,
        "synthetic_calibration_rows": 0,
        "truth_v2_accessed": False,
        "development_accessed": False,
        "rehearsal_accessed": False,
        "holdout_accessed": False,
        "threshold_selection_performed": False,
        "hyperparameter_sweep_performed": False,
    }
    _write_json_exclusive(destination / "run_manifest.json", run_manifest)

    scaler, scaler_evidence = _fit_scaler(
        fit_paths,
        fit_selected,
        tuple(config["channels"]),
        float(config["scaling"]["clip"]),
    )
    _write_json_exclusive(
        destination / "fit_scaler.json",
        {"scaler": scaler.to_dict(), "evidence": scaler_evidence},
    )
    model, training_report = _train(
        fit_paths,
        fit_selected,
        scaler=scaler,
        config=config,
        run_dir=destination,
    )
    checkpoint_path = destination / "model_state.pt"
    if checkpoint_path.exists():
        raise FileExistsError(checkpoint_path)
    torch.save(model.state_dict(), checkpoint_path)
    diagnostics = _natural_diagnostics(
        model,
        fit_paths,
        calibration_selected,
        scaler=scaler,
        config=config,
    )

    if _code_hashes(root) != code_start or _git_state(root) != git_start:
        raise ContextualTrainingContractError("Code or tracked Git state changed during training")
    report = {
        "run_id": destination.name,
        "status": "trained_not_thresholded",
        "elapsed_seconds": time.perf_counter() - started,
        "synthetic_training_rows": 0,
        "threshold_selection_performed": False,
        "fit_scaler": scaler.to_dict(),
        "training": training_report,
        "natural_calibration_diagnostic": diagnostics,
        "model_checkpoint": {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256_file(checkpoint_path),
        },
        "code_and_git_unchanged": True,
        "next_gate": "user-approved operational alert budget before conformal calibration",
    }
    _write_json_exclusive(destination / "training_report.json", report)
    _write_checksums(destination)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/adsb_contextual_physics_v1_train.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(repo_root=args.repo_root, config_path=args.config, run_dir=args.run_dir)
    except Exception as exc:
        destination = args.run_dir.resolve(strict=False)
        if destination.exists():
            marker = destination / "INCOMPLETE_DO_NOT_USE.md"
            if not marker.exists():
                marker.write_text(
                    "# INCOMPLETE — DO NOT USE\n\n"
                    f"Training stopped before a complete checksum chain: {type(exc).__name__}: {exc}\n",
                    encoding="utf-8",
                )
        raise
    print(json.dumps({
        "run_id": report["run_id"],
        "status": report["status"],
        "elapsed_seconds": report["elapsed_seconds"],
        "magnitude_flagged": report["natural_calibration_diagnostic"][
            "magnitude_domination_flagged_at_0_8"
        ],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
