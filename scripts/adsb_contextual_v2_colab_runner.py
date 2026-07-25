"""Observable, resumable CUDA runner for the frozen ADS-B contextual v2 fit.

This is an execution-engine change, not a model-selection change.  It preserves
the frozen sources, scaler, model, epoch count, batch size, learning rate, loss,
and per-part shuffle semantics.  It adds CUDA placement, progress telemetry and
mid-epoch recovery checkpoints suitable for interruptible Colab runtimes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.contextual_scaling import (
    NATURAL_FIT_ROLE,
    StrictNaturalRobustScaler,
    StrictScalingConfig,
)
from adsb.context import build_causal_context
from adsb.contextual_windowing import ContextualForecastBatch, _context_numeric_matrix
from adsb.models.contextual_residual_forecaster import (
    ContextualForecasterConfig,
    ContextualResidualForecaster,
    channelwise_gaussian_nll,
    weighted_masked_channel_loss,
)


class ColabRunnerError(RuntimeError):
    pass


INPUT_FEATURES = (
    "vertical_rate_residual",
    "speed_residual",
    "heading_residual",
    "altitude_source_residual",
    "east_velocity_residual",
    "north_velocity_residual",
    "context_log1p_dt_s",
    "track_sin",
    "track_cos",
    "phase=ground",
    "phase=climb",
    "phase=level",
    "phase=descent",
    "phase=unknown",
    "cadence=cadence_0",
    "cadence=cadence_1",
    "cadence=cadence_2",
    "cadence=cadence_3",
    "cadence=gap",
    "cadence=initial_or_invalid",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ColabRunnerError(f"Expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _torch_save_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_training_module(repo_root: Path):
    path = repo_root / "scripts/adsb_train_contextual_physics_v2.py"
    spec = importlib.util.spec_from_file_location("adsb_frozen_train_v2", path)
    if spec is None or spec.loader is None:
        raise ColabRunnerError(f"Cannot import frozen training script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_bundle(repo_root: Path, manifest_path: Path, mode: str) -> dict[str, Any]:
    manifest = _json(manifest_path)
    if manifest.get("candidate_namespace") != "contextual_physics_v2":
        raise ColabRunnerError("Unexpected bundle namespace")
    if manifest.get("frozen_training_parameters_changed") is not False:
        raise ColabRunnerError("Bundle does not attest frozen parameters")
    if manifest.get("fit_expansion_days") != ["2024-09-01", "2025-02-15", "2025-06-15"]:
        raise ColabRunnerError("Unexpected fit-expansion days")
    records = manifest.get("data", [])
    if len(records) != 784:
        raise ColabRunnerError(f"Expected 784 fit parts, observed {len(records)}")
    started = time.perf_counter()
    for number, record in enumerate(records, start=1):
        path = (repo_root / record["path"]).resolve(strict=True)
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise ColabRunnerError(f"Bundle path escapes repo root: {record['path']}") from exc
        if path.stat().st_size != int(record["bytes"]):
            raise ColabRunnerError(f"Byte-size mismatch: {record['path']}")
        if mode == "full" and _sha256_file(path) != record["sha256"]:
            raise ColabRunnerError(f"SHA-256 mismatch: {record['path']}")
        if number % 50 == 0 or number == len(records):
            print(
                f"verify_progress={number}/{len(records)} mode={mode} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
    for record in manifest.get("support_files", []):
        path = (repo_root / record["path"]).resolve(strict=True)
        if path.stat().st_size != int(record["bytes"]):
            raise ColabRunnerError(f"Support byte-size mismatch: {record['path']}")
        if mode == "full" and _sha256_file(path) != record["sha256"]:
            raise ColabRunnerError(f"Support SHA-256 mismatch: {record['path']}")
    print(
        json.dumps(
            {
                "status": "bundle_verified",
                "mode": mode,
                "parts": len(records),
                "rows": manifest["totals"]["parquet_rows"],
                "gib": round(manifest["totals"]["bytes"] / 2**30, 3),
            }
        ),
        flush=True,
    )
    return manifest


def _load_scaler(repo_root: Path, config: dict[str, Any]) -> tuple[StrictNaturalRobustScaler, dict[str, Any]]:
    path = repo_root / "artifacts/adsb/runs/20260724_contextual_physics_v2_train_v4/fit_scaler.json"
    payload = _json(path)
    values = payload["scaler"]
    if values.get("fit_role") != NATURAL_FIT_ROLE:
        raise ColabRunnerError("Scaler fit role is not natural_clean_fit")
    scaler = StrictNaturalRobustScaler(StrictScalingConfig(clip=float(values["clip"])))
    scaler.fit_from_statistics(
        {str(k): {"median": float(v["median"]), "mad": float(v["mad"])} for k, v in values["calibration"].items()},
        tuple(map(str, values["excluded_channels"])),
        tuple(map(str, config["channels"])),
        data_role=NATURAL_FIT_ROLE,
        contains_synthetic=False,
    )
    return scaler, payload


def _sources(repo_root: Path, bundle: dict[str, Any], train_module, config: dict[str, Any]):
    step = _json(repo_root / config["source_step5_manifest"])
    split = step["split_contract"]["splits"]
    fit_ids = train_module._sample_flights(
        tuple(split["fit"]["flight_ids"]),
        probability=float(config["data"]["fit_flight_sample_probability"]),
        seed=int(config["data"]["fit_flight_sample_seed"]),
        purpose="contextual_physics_v2_fit_step5",
    )
    fit_set = set(fit_ids)
    fit_source_set = {train_module._source_id(fid, train_module.STEP5_FIT_DAY) for fid in fit_ids}
    result = []
    for record in bundle["data"]:
        path = (repo_root / record["path"]).resolve(strict=True)
        if record["role"] == "fit_step5":
            result.append(
                train_module.FitSource(
                    path,
                    train_module.STEP5_FIT_DAY,
                    selected_flights=fit_set,
                    selected_sources=fit_source_set,
                )
            )
        else:
            result.append(
                train_module.FitSource(
                    path,
                    str(record["source_day"]),
                    selected_flights=None,
                    selected_sources=None,
                )
            )
    calibration_ids = train_module._sample_flights(
        tuple(split["calibration"]["flight_ids"]),
        probability=float(config["data"]["calibration_diagnostic_sample_probability"]),
        seed=int(config["data"]["calibration_diagnostic_sample_seed"]),
        purpose="contextual_physics_v2_calibration_diagnostic",
    )
    calibration_set = set(calibration_ids)
    step_paths = [
        (repo_root / record["path"]).resolve(strict=True)
        for record in bundle["data"]
        if record["role"] == "fit_step5"
    ]
    calibration_sources = [
        train_module.FitSource(
            path,
            train_module.STEP5_FIT_DAY,
            selected_flights=calibration_set,
            selected_sources=None,
        )
        for path in step_paths
    ]
    return result, calibration_sources, fit_ids, calibration_ids


def _device(value: str) -> torch.device:
    selected = "cuda" if value == "auto" and torch.cuda.is_available() else value
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ColabRunnerError("CUDA requested but torch.cuda.is_available() is false")
    if device.type != "cuda":
        raise ColabRunnerError("This Colab runner intentionally requires CUDA")
    return device


def _model_config(config: dict[str, Any], target_channels: int) -> ContextualForecasterConfig:
    values = config["model"]
    return ContextualForecasterConfig(
        input_features=len(INPUT_FEATURES),
        target_channels=target_channels,
        hidden_size=int(values["hidden_size"]),
        num_layers=int(values["num_layers"]),
        min_scale=float(values["min_scale"]),
        max_scale=float(values["max_scale"]),
    )


def _fast_make_batch(
    features: pd.DataFrame,
    *,
    scaler: StrictNaturalRobustScaler,
    config: dict[str, Any],
    train_module,
) -> ContextualForecastBatch:
    """Vectorized equivalent of the frozen per-window Python loop.

    The frozen implementation appends every 12-row window in Python.  This
    version preserves group order, validity rules, float32 conversion and
    feature order while vectorizing target positions within each flight.
    Metadata is intentionally omitted because optimizer training never reads
    it; the final calibration diagnostic still uses the frozen implementation.
    """

    scaled = features.copy()
    transformed = scaler.transform(features)
    for channel in scaler.active_channels:
        scaled[channel] = transformed[channel]
    context_config = train_module._context_config(config)
    context = build_causal_context(scaled, context_config)
    matrix, matrix_mask, input_names = _context_numeric_matrix(
        scaled,
        context,
        signal_columns=scaler.active_channels,
        config=context_config,
    )
    target = (
        scaled.loc[:, scaler.active_channels]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    target_mask = np.isfinite(target)
    history_rows = int(config["window"]["history_rows"])
    positions = pd.Series(np.arange(len(scaled)), index=scaled.index)
    windows: list[np.ndarray] = []
    window_masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    history_offsets = np.arange(history_rows, 0, -1, dtype=int)
    for _, group in scaled.groupby(context_config.flight_id_col, sort=False):
        pos = positions.loc[group.index].to_numpy(dtype=int)
        if len(pos) <= history_rows:
            continue
        times = pd.to_numeric(group[context_config.time_col], errors="coerce").to_numpy(float)
        intervals = np.lib.stride_tricks.sliding_window_view(times, history_rows + 1)
        differences = np.diff(intervals, axis=1)
        valid = np.all((differences > 0) & (differences <= context_config.max_gap_s), axis=1)
        local_targets = np.arange(history_rows, len(pos), dtype=int)[valid]
        if len(local_targets) == 0:
            continue
        history_positions = pos[local_targets[:, None] - history_offsets[None, :]]
        target_positions = pos[local_targets]
        windows.append(matrix[history_positions])
        window_masks.append(matrix_mask[history_positions])
        targets.append(np.where(target_mask[target_positions], target[target_positions], 0.0))
        target_masks.append(target_mask[target_positions].astype(float))

    feature_count = len(input_names)
    target_count = len(scaler.active_channels)
    if not windows:
        return ContextualForecastBatch(
            X=np.zeros((0, history_rows, feature_count), dtype=np.float32),
            X_mask=np.zeros((0, history_rows, feature_count), dtype=np.float32),
            y=np.zeros((0, target_count), dtype=np.float32),
            y_mask=np.zeros((0, target_count), dtype=np.float32),
            meta=pd.DataFrame(
                columns=["flight_id", "target_timestamp_utc", "context_phase", "context_cadence"]
            ),
            input_features=input_names,
            target_channels=scaler.active_channels,
        )
    return ContextualForecastBatch(
        X=np.concatenate(windows).astype(np.float32, copy=False),
        X_mask=np.concatenate(window_masks).astype(np.float32, copy=False),
        y=np.concatenate(targets).astype(np.float32, copy=False),
        y_mask=np.concatenate(target_masks).astype(np.float32, copy=False),
        meta=pd.DataFrame(
            columns=["flight_id", "target_timestamp_utc", "context_phase", "context_cadence"]
        ),
        input_features=input_names,
        target_channels=scaler.active_channels,
    )


def benchmark(
    *, repo_root: Path, bundle_path: Path, parts: int, verification: str, device_name: str
) -> dict[str, Any]:
    bundle = _verify_bundle(repo_root, bundle_path, verification)
    train_module = _load_training_module(repo_root)
    config = train_module._load_config(repo_root / "configs/adsb_contextual_physics_v2_train.json")
    scaler, _ = _load_scaler(repo_root, config)
    sources, _, _, _ = _sources(repo_root, bundle, train_module, config)
    device = _device(device_name)
    torch.manual_seed(int(config["training"]["seed"]))
    rows = windows = batches = 0
    feature_seconds = train_seconds = 0.0
    model = None
    optimizer = None
    weights = None
    for source_index, source in enumerate(sources[:parts]):
        started = time.perf_counter()
        features = source.load()
        batch = _fast_make_batch(
            features, scaler=scaler, config=config, train_module=train_module
        )
        feature_elapsed = time.perf_counter() - started
        feature_seconds += feature_elapsed
        rows += len(features)
        windows += len(batch.X)
        if len(batch.X) == 0:
            print(f"benchmark_part={source_index + 1}/{parts} windows=0", flush=True)
            continue
        if model is None:
            if tuple(batch.input_features) != INPUT_FEATURES:
                raise ColabRunnerError("Input feature order differs from frozen contract")
            model = ContextualResidualForecaster(_model_config(config, len(batch.target_channels))).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
            weights = torch.ones(len(batch.target_channels), dtype=torch.float32, device=device)
        permutation = torch.randperm(len(batch.X))
        started_train = time.perf_counter()
        for start in range(0, len(batch.X), int(config["training"]["batch_size"])):
            index = train_module._numpy_batch_indices(permutation, start, int(config["training"]["batch_size"]))
            tensors = [
                torch.from_numpy(array[index]).to(device, non_blocking=True)
                for array in (batch.X, batch.X_mask, batch.y, batch.y_mask)
            ]
            xb, mb, yb, ymb = tensors
            optimizer.zero_grad()
            location, scale = model(xb, mb)
            nll, _ = channelwise_gaussian_nll(yb, location, scale, ymb)
            loss = weighted_masked_channel_loss(nll, ymb, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            batches += 1
        torch.cuda.synchronize(device)
        train_elapsed = time.perf_counter() - started_train
        train_seconds += train_elapsed
        print(
            f"benchmark_part={source_index + 1}/{parts} rows={len(features)} windows={len(batch.X)} "
            f"feature_s={feature_elapsed:.2f} gpu_train_s={train_elapsed:.2f}",
            flush=True,
        )
    result = {
        "parts": parts,
        "rows": rows,
        "windows": windows,
        "batches": batches,
        "feature_seconds": feature_seconds,
        "gpu_train_seconds": train_seconds,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def _resume_payload(
    *,
    bundle_sha256: str,
    config_sha256: str,
    model: ContextualResidualForecaster,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, Any]],
    epoch_index: int,
    source_index: int,
    next_batch_start: int,
    permutation: torch.Tensor | None,
    epoch_loss_sum: float,
    epoch_windows: int,
    epoch_batches: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_role": "colab_interruption_recovery_only",
        "bundle_manifest_sha256": bundle_sha256,
        "config_sha256": config_sha256,
        "history": history,
        "epoch_index": epoch_index,
        "source_index": source_index,
        "next_batch_start": next_batch_start,
        "permutation": permutation,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_windows": epoch_windows,
        "epoch_batches": epoch_batches,
        "model_config": model.config.__dict__,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _restore_optimizer_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train(
    *,
    repo_root: Path,
    bundle_path: Path,
    run_dir: Path,
    verification: str,
    device_name: str,
    progress_every_batches: int,
    checkpoint_every_batches: int,
) -> dict[str, Any]:
    bundle = _verify_bundle(repo_root, bundle_path, verification)
    bundle_hash = _sha256_file(bundle_path)
    train_module = _load_training_module(repo_root)
    config_path = repo_root / "configs/adsb_contextual_physics_v2_train.json"
    config = train_module._load_config(config_path)
    config_hash = _sha256_file(config_path)
    scaler, scaler_payload = _load_scaler(repo_root, config)
    sources, calibration_sources, fit_ids, calibration_ids = _sources(
        repo_root, bundle, train_module, config
    )
    device = _device(device_name)
    destination = run_dir.resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    resume_path = destination / "colab_resume_checkpoint.pt"
    epoch_checkpoint_path = destination / "training_epoch_checkpoint.pt"
    report_path = destination / "training_report.json"
    if report_path.exists():
        report = _json(report_path)
        print(json.dumps({"status": "already_complete", "report": str(report_path)}), flush=True)
        return report

    if not (destination / "fit_scaler.json").exists():
        _write_json_atomic(destination / "fit_scaler.json", scaler_payload)
    manifest_path = destination / "run_manifest.json"
    if not manifest_path.exists():
        _write_json_atomic(
            manifest_path,
            {
                "schema_version": 1,
                "run_id": destination.name,
                "candidate_namespace": "contextual_physics_v2",
                "execution_engine": "colab_cuda_resumable_v1",
                "frozen_training_parameters_changed": False,
                "bundle_manifest_sha256": bundle_hash,
                "config_sha256": config_hash,
                "source_step5_manifest_sha256": bundle["source_step5_manifest_sha256"],
                "fit_expansion_manifest_sha256": bundle["fit_expansion_manifest_sha256"],
                "fit_expansion_days": bundle["fit_expansion_days"],
                "fit_parts": len(sources),
                "physical_rows": bundle["totals"]["parquet_rows"],
                "step5_fit_flights_selected": len(fit_ids),
                "calibration_diagnostic_flights_selected": len(calibration_ids),
                "synthetic_training_rows": 0,
                "synthetic_calibration_rows": 0,
                "truth_v2_accessed": False,
                "development_accessed": False,
                "rehearsal_accessed": False,
                "holdout_accessed": False,
                "threshold_selection_performed": False,
                "hyperparameter_sweep_performed": False,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
            },
        )

    training = config["training"]
    history: list[dict[str, Any]] = []
    epoch_index = source_index = next_batch_start = 0
    epoch_loss_sum = 0.0
    epoch_windows = epoch_batches = 0
    saved_permutation = None
    torch.manual_seed(int(training["seed"]))
    model_config = _model_config(config, len(scaler.active_channels))
    model = ContextualResidualForecaster(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint["bundle_manifest_sha256"] != bundle_hash or checkpoint["config_sha256"] != config_hash:
            raise ColabRunnerError("Resume checkpoint contract hash mismatch")
        if checkpoint["model_config"] != model_config.__dict__:
            raise ColabRunnerError("Resume checkpoint model config mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _restore_optimizer_device(optimizer, device)
        history = checkpoint["history"]
        epoch_index = int(checkpoint["epoch_index"])
        source_index = int(checkpoint["source_index"])
        next_batch_start = int(checkpoint["next_batch_start"])
        saved_permutation = checkpoint["permutation"]
        epoch_loss_sum = float(checkpoint["epoch_loss_sum"])
        epoch_windows = int(checkpoint["epoch_windows"])
        epoch_batches = int(checkpoint["epoch_batches"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        print(
            f"resume epoch={epoch_index + 1} source={source_index + 1}/{len(sources)} "
            f"batch_start={next_batch_start} completed_epochs={len(history)}",
            flush=True,
        )

    weights = torch.ones(model_config.target_channels, dtype=torch.float32, device=device)
    derived = {
        "model_config": model_config.__dict__,
        "input_features": list(INPUT_FEATURES),
        "target_channels": list(scaler.active_channels),
        "channel_weights": {channel: 1.0 for channel in scaler.active_channels},
        "scaler": scaler.to_dict(),
        "frozen_before_first_optimizer_step": True,
        "execution_device": str(device),
    }
    derived_path = destination / "derived_training_config.json"
    if not derived_path.exists():
        _write_json_atomic(derived_path, derived)
    elif _json(derived_path) != derived:
        raise ColabRunnerError("Derived training config changed across resume")

    overall_started = time.perf_counter()
    total_epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    while epoch_index < total_epochs:
        while source_index < len(sources):
            source = sources[source_index]
            feature_started = time.perf_counter()
            features = source.load()
            batch = _fast_make_batch(
                features, scaler=scaler, config=config, train_module=train_module
            )
            feature_seconds = time.perf_counter() - feature_started
            if tuple(batch.input_features) != INPUT_FEATURES:
                raise ColabRunnerError("Input feature order changed")
            if tuple(batch.target_channels) != tuple(scaler.active_channels):
                raise ColabRunnerError("Target channel order changed")
            if len(batch.X) == 0:
                print(
                    f"epoch={epoch_index + 1}/{total_epochs} source={source_index + 1}/{len(sources)} "
                    f"rows={len(features)} windows=0 feature_s={feature_seconds:.2f}",
                    flush=True,
                )
                source_index += 1
                continue

            if saved_permutation is not None:
                permutation = saved_permutation
                if len(permutation) != len(batch.X):
                    raise ColabRunnerError("Saved permutation length differs from rebuilt source batch")
                saved_permutation = None
            else:
                permutation = torch.randperm(len(batch.X))
                next_batch_start = 0
            source_train_started = time.perf_counter()
            source_batch_count = math.ceil(len(batch.X) / batch_size)
            for start in range(next_batch_start, len(batch.X), batch_size):
                index = train_module._numpy_batch_indices(permutation, start, batch_size)
                xb, mb, yb, ymb = [
                    torch.from_numpy(array[index]).to(device, non_blocking=True)
                    for array in (batch.X, batch.X_mask, batch.y, batch.y_mask)
                ]
                optimizer.zero_grad()
                location, scale = model(xb, mb)
                nll, _ = channelwise_gaussian_nll(yb, location, scale, ymb)
                loss = weighted_masked_channel_loss(nll, ymb, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(training["gradient_clip_norm"])
                )
                optimizer.step()
                count = len(index)
                epoch_loss_sum += float(loss.detach().cpu()) * count
                epoch_windows += count
                epoch_batches += 1
                next_start = min(start + batch_size, len(batch.X))
                local_batch = start // batch_size + 1
                if epoch_batches % progress_every_batches == 0 or next_start == len(batch.X):
                    if device.type == "cuda":
                        allocated = torch.cuda.memory_allocated(device) / 2**30
                        reserved = torch.cuda.memory_reserved(device) / 2**30
                    else:
                        allocated = reserved = 0.0
                    print(
                        f"epoch={epoch_index + 1}/{total_epochs} source={source_index + 1}/{len(sources)} "
                        f"source_batch={local_batch}/{source_batch_count} epoch_batches={epoch_batches} "
                        f"epoch_windows={epoch_windows} loss={float(loss.detach().cpu()):.6f} "
                        f"gpu_alloc_gib={allocated:.2f} gpu_reserved_gib={reserved:.2f}",
                        flush=True,
                    )
                if epoch_batches % checkpoint_every_batches == 0 and next_start < len(batch.X):
                    _torch_save_atomic(
                        resume_path,
                        _resume_payload(
                            bundle_sha256=bundle_hash,
                            config_sha256=config_hash,
                            model=model,
                            optimizer=optimizer,
                            history=history,
                            epoch_index=epoch_index,
                            source_index=source_index,
                            next_batch_start=next_start,
                            permutation=permutation,
                            epoch_loss_sum=epoch_loss_sum,
                            epoch_windows=epoch_windows,
                            epoch_batches=epoch_batches,
                        ),
                    )
            next_batch_start = 0
            source_index += 1
            _torch_save_atomic(
                resume_path,
                _resume_payload(
                    bundle_sha256=bundle_hash,
                    config_sha256=config_hash,
                    model=model,
                    optimizer=optimizer,
                    history=history,
                    epoch_index=epoch_index,
                    source_index=source_index,
                    next_batch_start=0,
                    permutation=None,
                    epoch_loss_sum=epoch_loss_sum,
                    epoch_windows=epoch_windows,
                    epoch_batches=epoch_batches,
                ),
            )
            print(
                f"source_done epoch={epoch_index + 1}/{total_epochs} "
                f"source={source_index}/{len(sources)} rows={len(features)} windows={len(batch.X)} "
                f"feature_s={feature_seconds:.2f} train_s={time.perf_counter() - source_train_started:.2f}",
                flush=True,
            )

        if epoch_windows == 0:
            raise ColabRunnerError("No scoreable training windows were produced")
        record = {
            "epoch": epoch_index + 1,
            "mean_weighted_gaussian_nll": epoch_loss_sum / epoch_windows,
            "windows": epoch_windows,
            "batches": epoch_batches,
        }
        history.append(record)
        _torch_save_atomic(
            epoch_checkpoint_path,
            {
                "schema_version": 1,
                "artifact_role": "incomplete_training_recovery_only",
                "completed_epochs": len(history),
                "history": history,
                "model_config": model.config.__dict__,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            },
        )
        print(
            f"epoch_done epoch={record['epoch']}/{total_epochs} windows={epoch_windows} "
            f"batches={epoch_batches} mean_nll={record['mean_weighted_gaussian_nll']:.8f}",
            flush=True,
        )
        epoch_index += 1
        source_index = 0
        epoch_loss_sum = 0.0
        epoch_windows = epoch_batches = 0
        _torch_save_atomic(
            resume_path,
            _resume_payload(
                bundle_sha256=bundle_hash,
                config_sha256=config_hash,
                model=model,
                optimizer=optimizer,
                history=history,
                epoch_index=epoch_index,
                source_index=0,
                next_batch_start=0,
                permutation=None,
                epoch_loss_sum=0.0,
                epoch_windows=0,
                epoch_batches=0,
            ),
        )

    model_cpu = model.to("cpu")
    model_path = destination / "model_state.pt"
    if not model_path.exists():
        torch.save(model_cpu.state_dict(), model_path)
    print("training_complete; starting natural calibration diagnostic", flush=True)
    diagnostics = train_module._natural_diagnostics(
        model_cpu,
        calibration_sources,
        scaler=scaler,
        config=config,
        seed_offset=1000,
    )
    report = {
        "run_id": destination.name,
        "status": "trained_not_thresholded",
        "elapsed_seconds_this_session": time.perf_counter() - overall_started,
        "synthetic_training_rows": 0,
        "threshold_selection_performed": False,
        "fit_scaler": scaler.to_dict(),
        "training": {
            "epochs": history,
            "derived_training_config": derived_path.name,
            "derived_training_config_sha256": _sha256_file(derived_path),
            "epoch_checkpoint": {
                "path": epoch_checkpoint_path.name,
                "completed_epochs": len(history),
                "bytes": epoch_checkpoint_path.stat().st_size,
                "sha256": _sha256_file(epoch_checkpoint_path),
            },
        },
        "natural_calibration_diagnostic": diagnostics,
        "model_checkpoint": {
            "path": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": _sha256_file(model_path),
        },
        "provenance": {
            "execution_engine": "colab_cuda_resumable_v1",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "bundle_manifest_sha256": bundle_hash,
            "config_sha256": config_hash,
            "frozen_training_parameters_changed": False,
            "mixed_precision_used": False,
        },
        "next_gate": "conformal + CUSUM + persistence_v2 calibration (Faz D)",
    }
    _write_json_atomic(report_path, report)
    train_module._write_checksums(destination)
    print(
        json.dumps(
            {
                "status": "complete",
                "training_report": str(report_path),
                "magnitude_domination_flagged_at_0_8": diagnostics[
                    "magnitude_domination_flagged_at_0_8"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "benchmark", "train"):
        item = sub.add_parser(name)
        item.add_argument("--repo-root", type=Path, default=Path.cwd())
        item.add_argument("--bundle-manifest", type=Path, required=True)
        item.add_argument("--verification", choices=("sizes", "full"), default="full")
        if name in {"benchmark", "train"}:
            item.add_argument("--device", default="auto")
        if name == "benchmark":
            item.add_argument("--parts", type=int, default=1)
        if name == "train":
            item.add_argument("--run-dir", type=Path, required=True)
            item.add_argument("--progress-every-batches", type=int, default=200)
            item.add_argument("--checkpoint-every-batches", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    bundle_path = args.bundle_manifest.resolve(strict=True)
    if args.command == "verify":
        _verify_bundle(root, bundle_path, args.verification)
    elif args.command == "benchmark":
        if args.parts < 1:
            raise ColabRunnerError("--parts must be >= 1")
        benchmark(
            repo_root=root,
            bundle_path=bundle_path,
            parts=args.parts,
            verification=args.verification,
            device_name=args.device,
        )
    else:
        if args.progress_every_batches < 1 or args.checkpoint_every_batches < 1:
            raise ColabRunnerError("Progress/checkpoint batch intervals must be >= 1")
        train(
            repo_root=root,
            bundle_path=bundle_path,
            run_dir=args.run_dir,
            verification=args.verification,
            device_name=args.device,
            progress_every_batches=args.progress_every_batches,
            checkpoint_every_batches=args.checkpoint_every_batches,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
