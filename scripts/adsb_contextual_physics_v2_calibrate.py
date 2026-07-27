"""Calibrate contextual_physics_v2 on the frozen natural calibration role.

Outputs conformal calibration scores, natural-burden mappings for instant and
cumulative-persistence profiles, and the unchanged two-axis Page-CUSUM mapping.
No truth-v2, development, rehearsal, or holdout data is read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.conditional_calibration import (  # noqa: E402
    ConditionalCalibrationConfig,
    HierarchicalConformalCalibrator,
    NATURAL_CALIBRATION_ROLE,
)
from adsb.contextual_scaling import StrictNaturalRobustScaler, StrictScalingConfig  # noqa: E402
from adsb.cusum import CusumConfig, VectorPageCUSUM  # noqa: E402
from adsb.evaluation import EpisodeContract, natural_alert_burden  # noqa: E402
from adsb.features import VECTOR_RESIDUAL_FEATURES  # noqa: E402
from adsb.models.contextual_persistence_v2 import (  # noqa: E402
    NULL_MEAN_SURPRISE,
    CumulativeConformalPersistence,
    PersistenceV2Config,
)
from adsb.models.contextual_residual_forecaster import (  # noqa: E402
    ContextualForecasterConfig,
    ContextualResidualForecaster,
    contextual_channel_scores,
)
from scripts.adsb_train_contextual_physics_v2 import (  # noqa: E402
    STEP5_FIT_DAY,
    FitSource,
    _canonical_json_sha256,
    _iter_fit_features,
    _make_batch,
    _sample_flights,
    _sha256_file,
    _write_checksums,
    _write_json_exclusive,
)

EXPECTED_EXPANSION_DAYS = ("2024-09-01", "2025-02-15", "2025-06-15")
FROZEN_TRAIN_CONFIG_PATH = Path("configs/adsb_contextual_physics_v2_train.json")
MIN_GROUP_SIZE = 1000
SCORE_BATCH_SIZE = 20_000
CUSUM_FIT_PARTS = 20
CUSUM_TARGET_VECTOR_SHIFT_MPS = 2.0
CUSUM_MAX_GAP_S = 60.0
CUSUM_MISSING_RESET_S = 60.0
CUSUM_Z_CLIP = 3.0
PLACEHOLDER_THRESHOLD_H = 1.0
PERSISTENCE_MAX_GAP_S = 30.0
PERSISTENCE_MISSING_RESET_S = 60.0
PERSISTENCE_SURPRISE_CLIP = 6.0
ALPHA_GRID = tuple(float(v) for v in np.geomspace(1e-5, 0.5, 12))
QUANTILE_GRID = (
    0.50,
    0.80,
    0.90,
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
    0.9995,
    0.9999,
    0.99995,
    0.99999,
    0.999995,
    0.999999,
    0.9999995,
    0.9999999,
)
EPISODE_CONTRACT = EpisodeContract(merge_gap_s=60.0, emission_time_col="t_end")


class CalibrationContractError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint(
    run_dir: Path,
) -> tuple[ContextualResidualForecaster, StrictNaturalRobustScaler, tuple[str, ...]]:
    derived = _load_json(run_dir / "derived_training_config.json")
    model = ContextualResidualForecaster(ContextualForecasterConfig(**derived["model_config"]))
    model.load_state_dict(
        torch.load(run_dir / "model_state.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    scaler_payload = derived["scaler"]
    scaler = StrictNaturalRobustScaler(
        StrictScalingConfig(clip=float(scaler_payload["clip"]))
    )
    scaler.calibration_ = scaler_payload["calibration"]
    scaler.excluded_channels_ = tuple(scaler_payload["excluded_channels"])
    return model, scaler, tuple(derived["target_channels"])


def _score_batched(model, batch) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(batch.X), SCORE_BATCH_SIZE):
        scores, _, _ = contextual_channel_scores(
            model,
            batch.X[start : start + SCORE_BATCH_SIZE],
            batch.X_mask[start : start + SCORE_BATCH_SIZE],
            batch.y[start : start + SCORE_BATCH_SIZE],
            batch.y_mask[start : start + SCORE_BATCH_SIZE],
        )
        parts.append(scores)
    return np.concatenate(parts) if parts else np.zeros_like(batch.y)


def _target_ground(features: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    source = features[["flight_id", "timestamp_utc", "on_ground"]].copy()
    source["timestamp_utc"] = pd.to_numeric(source["timestamp_utc"], errors="coerce")
    grouped = source.groupby(["flight_id", "timestamp_utc"], sort=False, dropna=False)
    if grouped["on_ground"].nunique(dropna=False).gt(1).any():
        raise CalibrationContractError("Conflicting on_ground values at a target timestamp")
    lookup = grouped["on_ground"].last()
    keys = pd.MultiIndex.from_arrays(
        [meta["flight_id"], meta["target_timestamp_utc"]],
        names=["flight_id", "timestamp_utc"],
    )
    if not keys.isin(lookup.index).all():
        raise CalibrationContractError("Model window target is missing from source features")
    return lookup.reindex(keys).to_numpy(dtype=object)


def _audit_on_ground_conflicts(
    sources: Iterable[FitSource],
) -> tuple[pd.DataFrame, set[str], dict[str, int]]:
    """Find contradictory ground-state keys and quarantine their whole flights."""

    parts: list[pd.DataFrame] = []
    source_parts = 0
    source_rows = 0
    for source, features in _iter_fit_features(sources):
        source_parts += 1
        source_rows += len(features)
        frame = features[["flight_id", "timestamp_utc", "on_ground"]].copy()
        frame["timestamp_utc"] = pd.to_numeric(frame["timestamp_utc"], errors="coerce")
        frame["source_path"] = source.path.as_posix()
        parts.append(frame)
    columns = [
        "flight_id",
        "timestamp_utc",
        "observed_on_ground_values",
        "source_paths",
        "source_rows",
    ]
    if not parts:
        return pd.DataFrame(columns=columns), set(), {
            "audited_parts": 0,
            "audited_feature_rows": 0,
        }
    audit = pd.concat(parts, ignore_index=True)
    grouped = audit.groupby(["flight_id", "timestamp_utc"], sort=False, dropna=False)
    conflicting = grouped["on_ground"].nunique(dropna=False)
    conflicting = conflicting.loc[conflicting.gt(1)]
    records: list[dict[str, Any]] = []
    for flight_id, timestamp_utc in conflicting.index:
        group = grouped.get_group((flight_id, timestamp_utc))
        values = sorted(
            {
                "<missing>" if pd.isna(value) else str(value)
                for value in group["on_ground"].tolist()
            }
        )
        records.append(
            {
                "flight_id": str(flight_id),
                "timestamp_utc": float(timestamp_utc),
                "observed_on_ground_values": json.dumps(values, separators=(",", ":")),
                "source_paths": json.dumps(
                    sorted(set(group["source_path"].astype(str))), separators=(",", ":")
                ),
                "source_rows": int(len(group)),
            }
        )
    conflicts = pd.DataFrame.from_records(records, columns=columns)
    quarantined_flights = set(conflicts["flight_id"].astype(str))
    return conflicts, quarantined_flights, {
        "audited_parts": source_parts,
        "audited_feature_rows": source_rows,
    }


def _score_calibration_sources(
    sources: Iterable[FitSource],
    *,
    model: ContextualResidualForecaster,
    scaler: StrictNaturalRobustScaler,
    target_channels: tuple[str, ...],
    config: dict[str, Any],
    quarantined_flights: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    long_parts: list[pd.DataFrame] = []
    cusum_parts: list[pd.DataFrame] = []
    counts = {
        "parts_with_rows": 0,
        "feature_rows_before_quarantine": 0,
        "quarantined_feature_rows": 0,
        "feature_rows": 0,
        "windows": 0,
    }
    for part_number, (_, features) in enumerate(_iter_fit_features(sources), start=1):
        counts["parts_with_rows"] += 1
        counts["feature_rows_before_quarantine"] += len(features)
        quarantine_mask = features["flight_id"].astype(str).isin(quarantined_flights)
        counts["quarantined_feature_rows"] += int(quarantine_mask.sum())
        features = features.loc[~quarantine_mask].reset_index(drop=True)
        counts["feature_rows"] += len(features)
        if features.empty:
            continue
        batch = _make_batch(features, scaler=scaler, config=config)
        cusum_parts.append(
            features[
                ["flight_id", "timestamp_utc", "on_ground", *VECTOR_RESIDUAL_FEATURES]
            ].copy()
        )
        if len(batch.X) == 0:
            continue
        scores = _score_batched(model, batch)
        grounds = _target_ground(features, batch.meta)
        counts["windows"] += len(batch.X)
        base = batch.meta.copy()
        base["on_ground"] = grounds
        for channel_number, channel in enumerate(target_channels):
            valid = batch.y_mask[:, channel_number] > 0
            if not valid.any():
                continue
            long_parts.append(
                pd.DataFrame(
                    {
                        "flight_id": base.loc[valid, "flight_id"].to_numpy(),
                        "timestamp_utc": base.loc[
                            valid, "target_timestamp_utc"
                        ].to_numpy(float),
                        "on_ground": base.loc[valid, "on_ground"].to_numpy(dtype=object),
                        "channel": channel,
                        "context_phase": base.loc[valid, "context_phase"].to_numpy(),
                        "context_cadence": base.loc[valid, "context_cadence"].to_numpy(),
                        "score": scores[valid, channel_number],
                    }
                )
            )
        if part_number % 20 == 0:
            print(f"calibration part={part_number} windows={counts['windows']}", flush=True)
    if not long_parts or not cusum_parts:
        raise CalibrationContractError("Natural calibration selection produced no scores")
    long_frame = pd.concat(long_parts, ignore_index=True)
    long_frame.sort_values(["channel", "flight_id", "timestamp_utc"], inplace=True)
    long_frame.reset_index(drop=True, inplace=True)
    long_frame["t_start"] = long_frame.groupby(["channel", "flight_id"], sort=False)[
        "timestamp_utc"
    ].shift(1)
    long_frame["t_start"] = long_frame["t_start"].fillna(long_frame["timestamp_utc"])
    long_frame["t_end"] = long_frame["timestamp_utc"]
    cusum_frame = pd.concat(cusum_parts, ignore_index=True)
    cusum_frame.sort_values(["flight_id", "timestamp_utc"], inplace=True)
    cusum_frame.reset_index(drop=True, inplace=True)
    return long_frame, cusum_frame, counts


def _derive_reference_multiplier(p_values: pd.DataFrame) -> tuple[float, dict[str, float]]:
    means: dict[str, float] = {}
    for channel, group in p_values.groupby("channel", sort=True):
        p = pd.to_numeric(group["conformal_p_value"], errors="coerce").to_numpy(float)
        valid = np.isfinite(p) & (p > 0.0)
        if not valid.any():
            raise CalibrationContractError(f"No finite conformal p-values for {channel}")
        surprise = np.minimum(-np.log10(p[valid]), PERSISTENCE_SURPRISE_CLIP)
        means[str(channel)] = float(np.mean(surprise))
    ratio = max(means.values()) / NULL_MEAN_SURPRISE
    multiplier = math.ceil(100.0 * max(1.01, 1.10 * ratio)) / 100.0
    if not np.isfinite(multiplier) or multiplier <= 1.0:
        raise CalibrationContractError("Invalid derived persistence multiplier")
    return multiplier, means


def _burden(frame: pd.DataFrame, alarm: np.ndarray) -> dict[str, Any]:
    return natural_alert_burden(frame, alarm, contract=EPISODE_CONTRACT)


def _rate(row: dict[str, Any]) -> float:
    value = row.get("alert_episodes_per_scoreable_flight_hour")
    return float(value) if value is not None and np.isfinite(value) else float("inf")


def _select_nearest(
    curve: list[dict[str, Any]], *, target_per_hour: float, parameter: str, higher_is_conservative: bool
) -> dict[str, Any]:
    if not curve:
        raise CalibrationContractError("Cannot select from an empty natural-burden curve")
    direction = -1.0 if higher_is_conservative else 1.0
    return min(
        curve,
        key=lambda row: (
            abs(_rate(row) - target_per_hour),
            direction * float(row[parameter]),
        ),
    )


def _profile_specs(budget: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    for channel, payload in budget["temporal_profiles"].items():
        if "mode" in payload:
            yield channel, "default", payload
        else:
            for profile, spec in payload.items():
                yield channel, profile, spec


def _derive_model_budgets(
    scored: pd.DataFrame, budget: dict[str, Any], multiplier: float
) -> dict[str, Any]:
    pareto = budget["budget_grid_episodes_per_100_scoreable_flight_hours"]
    shares = budget["budget_shares_of_total"]
    result: dict[str, Any] = {"alpha_grid": list(ALPHA_GRID), "channels": {}}
    persistence_channels = {
        channel
        for channel, _, spec in _profile_specs(budget)
        if spec["mode"] == "persistence_v2_cumulative"
    }
    persistence_input = scored.loc[scored["channel"].isin(persistence_channels)].copy()
    # The frozen detector writes by positional order and therefore requires a dense index.
    persistence_input.reset_index(drop=True, inplace=True)
    persistence = CumulativeConformalPersistence(
        PersistenceV2Config(
            reference_shift_multiplier=multiplier,
            threshold_h=PLACEHOLDER_THRESHOLD_H,
            max_gap_s=PERSISTENCE_MAX_GAP_S,
            missing_reset_s=PERSISTENCE_MISSING_RESET_S,
            surprise_clip=PERSISTENCE_SURPRISE_CLIP,
        )
    )
    persistence_scored = persistence.score(persistence_input)
    persistence_input = pd.concat([persistence_input, persistence_scored], axis=1)

    for channel, profile_name, spec in _profile_specs(budget):
        if spec["mode"] == "accumulation":
            continue
        channel_frame = scored.loc[scored["channel"] == channel].copy()
        if channel_frame.empty:
            raise CalibrationContractError(f"No calibration scores for budget channel {channel}")
        profile_fraction = float(spec.get("budget_fraction_of_channel", 1.0))
        target_scale = float(shares[channel]) * profile_fraction / 100.0
        channel_result = result["channels"].setdefault(channel, {"profiles": {}})
        if spec["mode"] == "instant":
            curve = [
                {
                    "alpha": alpha,
                    **_burden(
                        channel_frame,
                        channel_frame["conformal_p_value"].le(alpha).to_numpy(bool),
                    ),
                }
                for alpha in ALPHA_GRID
            ]
            selected = {
                str(v): _select_nearest(
                    curve,
                    target_per_hour=float(v) * target_scale,
                    parameter="alpha",
                    higher_is_conservative=False,
                )["alpha"]
                for v in pareto
            }
            channel_result["profiles"][profile_name] = {
                "mode": "instant",
                "natural_curve": curve,
                "selected_alpha_by_budget": selected,
            }
        elif spec["mode"] == "persistence_v2_cumulative":
            state_frame = persistence_input.loc[persistence_input["channel"] == channel].copy()
            evaluable = state_frame["persistence_v2_evaluable"].to_numpy(bool)
            states = state_frame.loc[evaluable, "persistence_v2_state"].to_numpy(float)
            candidates = sorted(set(float(np.quantile(states, q)) for q in QUANTILE_GRID))
            curve = [
                {
                    "threshold_h": h,
                    **_burden(
                        state_frame,
                        evaluable & (state_frame["persistence_v2_state"].to_numpy(float) > h),
                    ),
                }
                for h in candidates
            ]
            selected = {
                str(v): _select_nearest(
                    curve,
                    target_per_hour=float(v) * target_scale,
                    parameter="threshold_h",
                    higher_is_conservative=True,
                )["threshold_h"]
                for v in pareto
            }
            channel_result["profiles"][profile_name] = {
                "mode": "persistence_v2_cumulative",
                "natural_curve": curve,
                "selected_threshold_h_by_budget": selected,
            }
        else:
            raise CalibrationContractError(f"Unexpected temporal mode: {spec['mode']}")
    result["persistence_detector_template"] = persistence.to_dict()
    return result


def _fit_cusum(fit_sources: list[FitSource]) -> VectorPageCUSUM:
    parts = [features for _, features in _iter_fit_features(fit_sources[:CUSUM_FIT_PARTS])]
    if not parts:
        raise CalibrationContractError("CUSUM natural fit subset is empty")
    features = pd.concat(parts, ignore_index=True)
    features.sort_values(["flight_id", "timestamp_utc"], inplace=True)
    detector = VectorPageCUSUM(
        CusumConfig(
            target_vector_shift_mps=CUSUM_TARGET_VECTOR_SHIFT_MPS,
            threshold_h=PLACEHOLDER_THRESHOLD_H,
            max_gap_s=CUSUM_MAX_GAP_S,
            missing_reset_s=CUSUM_MISSING_RESET_S,
            z_clip=CUSUM_Z_CLIP,
            channels=tuple(VECTOR_RESIDUAL_FEATURES),
        )
    ).fit(features)
    return detector


def _derive_cusum_budgets(
    detector: VectorPageCUSUM, calibration: pd.DataFrame, budget: dict[str, Any]
) -> dict[str, Any]:
    scored = detector.score_rows(calibration)
    frame = calibration[["flight_id", "timestamp_utc"]].copy()
    frame["t_start"] = frame.groupby("flight_id", sort=False)["timestamp_utc"].shift(1)
    frame["t_start"] = frame["t_start"].fillna(frame["timestamp_utc"])
    frame["t_end"] = frame["timestamp_utc"]
    evaluable = scored["cusum_evaluable"].to_numpy(bool)
    states = scored.loc[evaluable, "cusum_joint_score"].to_numpy(float)
    candidates = sorted(set(float(np.quantile(states, q)) for q in QUANTILE_GRID))
    curve = [
        {
            "threshold_h": h,
            **_burden(frame, evaluable & (scored["cusum_joint_score"].to_numpy(float) > h)),
        }
        for h in candidates
    ]
    shares = budget["budget_shares_of_total"]
    combined_share = sum(float(shares[channel]) for channel in VECTOR_RESIDUAL_FEATURES)
    selected = {
        str(v): _select_nearest(
            curve,
            target_per_hour=float(v) * combined_share / 100.0,
            parameter="threshold_h",
            higher_is_conservative=True,
        )["threshold_h"]
        for v in budget["budget_grid_episodes_per_100_scoreable_flight_hours"]
    }
    return {
        "detector_template": detector.to_dict(),
        "combined_budget_share": combined_share,
        "natural_curve": curve,
        "selected_threshold_h_by_budget": selected,
    }


def run(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    report = _load_json(run_dir / "training_report.json")
    diagnostic = report["natural_calibration_diagnostic"]
    if diagnostic["magnitude_domination_flagged_at_0_8"] is not False:
        raise CalibrationContractError("Magnitude-domination gate is not false")
    run_manifest = _load_json(run_dir / "run_manifest.json")
    if tuple(run_manifest["fit_expansion_days"]) != EXPECTED_EXPANSION_DAYS:
        raise CalibrationContractError("Unexpected fit-expansion day set")
    if out_dir.exists():
        raise FileExistsError(out_dir)

    root = Path.cwd().resolve()
    train_config_path = root / run_manifest.get(
        "config_path", FROZEN_TRAIN_CONFIG_PATH.as_posix()
    )
    if _sha256_file(train_config_path) != run_manifest.get("config_sha256"):
        raise CalibrationContractError("Frozen training config SHA-256 mismatch")
    config = _load_json(train_config_path)
    step5_manifest_path = root / config["source_step5_manifest"]
    observed_step5_hash = _sha256_file(step5_manifest_path)
    if (
        observed_step5_hash != config.get("source_step5_manifest_sha256")
        or observed_step5_hash != run_manifest.get("source_step5_manifest_sha256")
    ):
        raise CalibrationContractError("Frozen Step-5 manifest SHA-256 mismatch")
    step5 = _load_json(step5_manifest_path)
    split = step5["split_contract"]["splits"]
    calibration_selected = _sample_flights(
        split["calibration"]["flight_ids"],
        probability=float(config["data"]["calibration_diagnostic_sample_probability"]),
        seed=int(config["data"]["calibration_diagnostic_sample_seed"]),
        purpose="contextual_physics_v2_calibration_diagnostic",
    )
    selected_hash = _canonical_json_sha256(list(calibration_selected))
    expected_selected_hash = run_manifest.get("calibration_diagnostic_flight_ids_sha256")
    if expected_selected_hash is not None and selected_hash != expected_selected_hash:
        raise CalibrationContractError("Calibration-selected flight hash differs from training")
    if expected_selected_hash is None:
        if run_manifest.get("execution_engine") != "colab_cuda_resumable_v1":
            raise CalibrationContractError(
                "Calibration-selected flight hash missing from non-Colab manifest"
            )
        if len(calibration_selected) != int(
            run_manifest.get("calibration_diagnostic_flights_selected", -1)
        ):
            raise CalibrationContractError(
                "Calibration-selected flight count differs from Colab training"
            )
    fit_ids = tuple(split["fit"]["flight_ids"])
    if set(fit_ids) & set(calibration_selected):
        raise CalibrationContractError("Fit and calibration flights overlap")
    input_paths = [root / row["path"] for row in step5["inputs"] if row["role"] == "fit"]
    calibration_sources = [
        FitSource(
            path,
            STEP5_FIT_DAY,
            selected_flights=set(calibration_selected),
            selected_sources=None,
        )
        for path in input_paths
    ]
    fit_sources = [
        FitSource(path, STEP5_FIT_DAY, selected_flights=set(fit_ids), selected_sources=None)
        for path in input_paths
    ]
    conflicts, quarantined_flights, audit_counts = _audit_on_ground_conflicts(
        calibration_sources
    )

    out_dir.mkdir(parents=True, exist_ok=False)
    quarantine_path = out_dir / "calibration_on_ground_quarantine.parquet"
    conflicts.to_parquet(quarantine_path, index=False)
    model, scaler, target_channels = _load_checkpoint(run_dir)
    long_frame, cusum_calibration, counts = _score_calibration_sources(
        calibration_sources,
        model=model,
        scaler=scaler,
        target_channels=target_channels,
        config=config,
        quarantined_flights=quarantined_flights,
    )
    calibrator = HierarchicalConformalCalibrator(
        ConditionalCalibrationConfig(min_group_size=MIN_GROUP_SIZE)
    ).fit(long_frame, data_role=NATURAL_CALIBRATION_ROLE, contains_synthetic=False)
    transformed = calibrator.transform(long_frame)
    long_frame["conformal_p_value"] = transformed["conformal_p_value"].to_numpy(float)
    long_frame["calibration_level"] = transformed["calibration_level"].to_numpy()
    long_frame["calibration_n"] = transformed["calibration_n"].to_numpy(int)
    score_path = out_dir / "natural_conformal_scores.parquet"
    long_frame.to_parquet(score_path, index=False)

    budget = _load_json(root / "configs/adsb_contextual_physics_v2_alarm_budget.json")
    persistence_channels = {
        channel
        for channel, _, spec in _profile_specs(budget)
        if spec["mode"] == "persistence_v2_cumulative"
    }
    multiplier, channel_means = _derive_reference_multiplier(
        long_frame.loc[long_frame["channel"].isin(persistence_channels)]
    )
    model_budgets = _derive_model_budgets(long_frame, budget, multiplier)
    cusum = _fit_cusum(fit_sources)
    cusum_budgets = _derive_cusum_budgets(cusum, cusum_calibration, budget)

    coverage = {
        channel: {
            "rows": int(len(group)),
            "levels": {
                str(level): int(count)
                for level, count in group["calibration_level"].value_counts().items()
            },
        }
        for channel, group in long_frame.groupby("channel", sort=True)
    }
    result = {
        "schema_version": 1,
        "candidate_namespace": "contextual_physics_v2",
        "data_role": NATURAL_CALIBRATION_ROLE,
        "training_run": str(run_dir),
        "training_report_sha256": _sha256_file(run_dir / "training_report.json"),
        "magnitude_domination_gate": False,
        "calibration_selected_flights": len(calibration_selected),
        "calibration_retained_flights": len(calibration_selected)
        - len(quarantined_flights),
        "calibration_selected_flight_ids_sha256": _canonical_json_sha256(
            list(calibration_selected)
        ),
        "data_quality_quarantine": {
            "policy": "exclude_entire_flight_on_conflicting_on_ground_at_same_timestamp",
            "conflict_keys": int(len(conflicts)),
            "quarantined_flights": int(len(quarantined_flights)),
            "quarantined_flight_ids_sha256": _canonical_json_sha256(
                sorted(quarantined_flights)
            ),
            "audit_counts": audit_counts,
            "artifact_path": quarantine_path.name,
            "artifact_sha256": _sha256_file(quarantine_path),
            "post_training_user_authorized_deviation": True,
        },
        "counts": counts,
        "coverage": coverage,
        "conformal": {
            "min_group_size": MIN_GROUP_SIZE,
            "scores_path": score_path.name,
            "scores_sha256": _sha256_file(score_path),
        },
        "persistence_v2": {
            "reference_shift_multiplier": multiplier,
            "channel_mean_clipped_surprise": channel_means,
            "null_mean_surprise": NULL_MEAN_SURPRISE,
            "selection_rule": "prereg_Ek_A_A.3",
            **model_budgets,
        },
        "cusum": cusum_budgets,
        "budget_grid": budget["budget_grid_episodes_per_100_scoreable_flight_hours"],
        "synthetic_rows": 0,
        "truth_v2_accessed": False,
        "thresholds_selected_from_natural_calibration_only": True,
    }
    _write_json_exclusive(out_dir / "calibration_report.json", result)
    _write_checksums(out_dir)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/adsb/runs/20260724_contextual_physics_v2_train_v4"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/adsb/runs/20260724_contextual_physics_v2_calibration_v1"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args.run_dir.resolve(), args.out_dir.resolve())
    print(
        json.dumps(
            {
                "calibration_selected_flights": result["calibration_selected_flights"],
                "reference_shift_multiplier": result["persistence_v2"][
                    "reference_shift_multiplier"
                ],
                "budget_grid": result["budget_grid"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
