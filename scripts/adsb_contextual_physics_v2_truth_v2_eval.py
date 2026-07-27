"""Evaluate frozen contextual_physics_v2 detectors on the existing truth-v2 corpus.

All model alpha and cumulative/CUSUM ``h`` values come from the natural-only
Phase-D calibration artifact.  This script never regenerates truth-v2 and never
selects a threshold from truth labels or recall.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.conditional_calibration import (  # noqa: E402
    ConditionalCalibrationConfig,
    HierarchicalConformalCalibrator,
    NATURAL_CALIBRATION_ROLE,
)
from adsb.cusum import VectorPageCUSUM  # noqa: E402
from adsb.evaluation import (  # noqa: E402
    EpisodeContract,
    active_interval_coverage,
    event_detection_metrics,
    event_observability_denominators,
    natural_alert_burden,
    truth_event_table,
)
from adsb.features import VECTOR_RESIDUAL_FEATURES, build_feature_table  # noqa: E402
from adsb.models.contextual_persistence_v2 import (  # noqa: E402
    CumulativeConformalPersistence,
    PersistenceV2Config,
)
from scripts.adsb_contextual_physics_v2_calibrate import (  # noqa: E402
    EXPECTED_EXPANSION_DAYS,
    MIN_GROUP_SIZE,
    PERSISTENCE_MAX_GAP_S,
    PERSISTENCE_MISSING_RESET_S,
    PERSISTENCE_SURPRISE_CLIP,
    _load_checkpoint,
    _score_batched,
    _target_ground,
)
from scripts.adsb_train_contextual_physics_v2 import (  # noqa: E402
    _canonical_json_sha256,
    _make_batch,
    _sha256_file,
    _write_checksums,
    _write_json_exclusive,
)

CORPUS_DIR = Path("data/objectstore/synthetic/adsb_v2_20260713_01")
BUDGET_CONFIG_PATH = Path("configs/adsb_contextual_physics_v2_alarm_budget.json")
FROZEN_TRAIN_CONFIG_PATH = Path("configs/adsb_contextual_physics_v2_train.json")
EXPECTED_CORPUS_FILES = (
    "clean.parquet",
    "vertical_rate_frozen.parquet",
    "ground_speed_biased.parquet",
    "track_frozen.parquet",
    "position_ramp_stealthy.parquet",
    "altitude_dropout.parquet",
)
FLIGHT_CHUNK_SIZE = 1000
EPISODE_CONTRACT = EpisodeContract(merge_gap_s=60.0, emission_time_col="t_end")

SOURCE_COLUMNS = [
    "flight_id",
    "timestamp_utc",
    "lat",
    "lon",
    "alt",
    "alt_geom_m",
    "on_ground",
    "ground_speed_ms",
    "track_deg",
    "vertical_rate_ms",
    "roll_deg",
    "event_id",
    "event_type",
    "attack_onset",
    "observable_onset",
    "event_end",
    "injection_active",
    "observable_changed",
    "evaluable_truth",
]

MODEL_RECIPES: dict[str, str] = {
    "vertical_rate_frozen": "vertical_rate_residual",
    "ground_speed_biased": "speed_residual",
    "track_frozen": "heading_residual",
}
CUSUM_RECIPE = "position_ramp_stealthy"
OUT_OF_SCOPE_RECIPES = ("altitude_dropout",)


class TruthV2EvaluationContractError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TruthV2EvaluationContractError(f"Expected JSON object: {path}")
    return value


def _with_exposure_bounds(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    result = frame.copy()
    group_columns = ["channel", "flight_id"] if "channel" in result else ["flight_id"]
    result["t_start"] = result.groupby(group_columns, sort=False)[time_col].shift(1)
    result["t_start"] = result["t_start"].fillna(result[time_col])
    result["t_end"] = result[time_col]
    return result


def _audit_truth_v2_ground_conflicts(
    corpus_dir: Path, filenames: tuple[str, ...]
) -> tuple[pd.DataFrame, set[str], dict[str, int]]:
    records: list[dict[str, Any]] = []
    audited_rows = 0
    clean_flights = 0
    for filename in filenames:
        frame = pd.read_parquet(
            corpus_dir / filename,
            columns=["flight_id", "timestamp_utc", "on_ground"],
        )
        audited_rows += len(frame)
        if filename == "clean.parquet":
            clean_flights = int(frame["flight_id"].nunique())
        frame["timestamp_utc"] = pd.to_numeric(frame["timestamp_utc"], errors="coerce")
        duplicate_mask = frame.duplicated(
            ["flight_id", "timestamp_utc"], keep=False
        )
        duplicates = frame.loc[duplicate_mask]
        del duplicate_mask
        del frame
        grouped = duplicates.groupby(
            ["flight_id", "timestamp_utc"], sort=False, dropna=False
        )
        conflicting = grouped["on_ground"].nunique(dropna=False)
        for flight_id, timestamp_utc in conflicting.loc[conflicting.gt(1)].index:
            group = grouped.get_group((flight_id, timestamp_utc))
            values = sorted(
                {
                    "<missing>" if pd.isna(value) else str(value)
                    for value in group["on_ground"].tolist()
                }
            )
            records.append(
                {
                    "corpus_file": filename,
                    "flight_id": str(flight_id),
                    "timestamp_utc": float(timestamp_utc),
                    "observed_on_ground_values": json.dumps(
                        values, separators=(",", ":")
                    ),
                    "source_rows": int(len(group)),
                }
            )
        del grouped
        del duplicates
        del conflicting
        gc.collect()
    columns = [
        "corpus_file",
        "flight_id",
        "timestamp_utc",
        "observed_on_ground_values",
        "source_rows",
    ]
    conflicts = pd.DataFrame.from_records(records, columns=columns)
    quarantined = set(conflicts["flight_id"].astype(str))
    unique_keys = (
        conflicts[["flight_id", "timestamp_utc"]].drop_duplicates()
        if not conflicts.empty
        else conflicts
    )
    return conflicts, quarantined, {
        "audited_files": len(filenames),
        "audited_rows": audited_rows,
        "conflict_records_across_files": len(conflicts),
        "unique_conflict_keys": len(unique_keys),
        "clean_flights_before_quarantine": clean_flights,
    }


def _load_corpus_features(
    path: Path, quarantined_flights: set[str] | None = None
) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=SOURCE_COLUMNS)
    if raw["flight_id"].isna().any():
        raise TruthV2EvaluationContractError(f"{path}: null flight_id")
    if quarantined_flights:
        raw = raw.loc[
            ~raw["flight_id"].astype(str).isin(quarantined_flights)
        ].reset_index(drop=True)
    return build_feature_table(raw)


def _score_model_long(
    features: pd.DataFrame,
    *,
    model,
    scaler,
    target_channels: tuple[str, ...],
    train_config: dict[str, Any],
    output_channels: set[str] | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    flight_ids = features["flight_id"].drop_duplicates().to_numpy()
    for start in range(0, len(flight_ids), FLIGHT_CHUNK_SIZE):
        selected = set(flight_ids[start : start + FLIGHT_CHUNK_SIZE])
        chunk = features.loc[features["flight_id"].isin(selected)]
        batch = _make_batch(chunk, scaler=scaler, config=train_config)
        if len(batch.X) == 0:
            continue
        scores = _score_batched(model, batch)
        grounds = _target_ground(chunk, batch.meta)
        for channel_index, channel in enumerate(target_channels):
            if output_channels is not None and channel not in output_channels:
                continue
            valid = batch.y_mask[:, channel_index] > 0
            if not valid.any():
                continue
            parts.append(
                pd.DataFrame(
                    {
                        "flight_id": batch.meta.loc[valid, "flight_id"].to_numpy(),
                        "timestamp_utc": batch.meta.loc[
                            valid, "target_timestamp_utc"
                        ].to_numpy(float),
                        "on_ground": grounds[valid],
                        "channel": channel,
                        "context_phase": batch.meta.loc[
                            valid, "context_phase"
                        ].to_numpy(),
                        "context_cadence": batch.meta.loc[
                            valid, "context_cadence"
                        ].to_numpy(),
                        "score": scores[valid, channel_index],
                    }
                )
            )
    if not parts:
        raise TruthV2EvaluationContractError("Corpus produced no scoreable model windows")
    result = pd.concat(parts, ignore_index=True)
    result.sort_values(["channel", "flight_id", "timestamp_utc"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    result = _with_exposure_bounds(result, "timestamp_utc")
    return result


def _fit_frozen_calibrator(calibration_dir: Path, report: dict[str, Any]):
    score_path = calibration_dir / report["conformal"]["scores_path"]
    if _sha256_file(score_path) != report["conformal"]["scores_sha256"]:
        raise TruthV2EvaluationContractError("Natural conformal score artifact hash mismatch")
    natural = pd.read_parquet(score_path)
    calibrator = HierarchicalConformalCalibrator(
        ConditionalCalibrationConfig(min_group_size=int(report["conformal"]["min_group_size"]))
    ).fit(natural, data_role=NATURAL_CALIBRATION_ROLE, contains_synthetic=False)
    return calibrator


def _conformal_transform(frame: pd.DataFrame, calibrator) -> pd.DataFrame:
    result = frame.copy()
    transformed = calibrator.transform(result)
    result["conformal_p_value"] = transformed["conformal_p_value"].to_numpy(float)
    return result


def _detection_point(
    events: pd.DataFrame,
    scored: pd.DataFrame,
    alarm: np.ndarray,
    clean_scored: pd.DataFrame,
    clean_alarm: np.ndarray,
) -> dict[str, Any]:
    detection = event_detection_metrics(events, scored, alarm)
    coverage = active_interval_coverage(events, scored, alarm)
    burden = natural_alert_burden(clean_scored, clean_alarm, contract=EPISODE_CONTRACT)
    return {
        "event_recall": detection["event_recall"],
        "n_events": detection["n_events"],
        "n_detected_events": detection["n_detected_events"],
        "first_alarm_delay_s_median": detection["first_alarm_delay_s"]["median"],
        "first_alarm_delay_s_p95": detection["first_alarm_delay_s"]["p95"],
        "active_interval_coverage_micro_fraction": coverage["micro_fraction"],
        "paired_clean_natural_burden_per_hour": burden[
            "alert_episodes_per_scoreable_flight_hour"
        ],
    }


def _persistence_detector(report: dict[str, Any]) -> CumulativeConformalPersistence:
    multiplier = float(report["persistence_v2"]["reference_shift_multiplier"])
    return CumulativeConformalPersistence(
        PersistenceV2Config(
            reference_shift_multiplier=multiplier,
            threshold_h=1.0,
            max_gap_s=PERSISTENCE_MAX_GAP_S,
            missing_reset_s=PERSISTENCE_MISSING_RESET_S,
            surprise_clip=PERSISTENCE_SURPRISE_CLIP,
        )
    )


def _evaluate_model_recipe(
    recipe: str,
    channel: str,
    corrupt: pd.DataFrame,
    clean: pd.DataFrame,
    events: pd.DataFrame,
    calibration_report: dict[str, Any],
    pareto: list[float],
) -> dict[str, Any]:
    corrupt_channel = corrupt.loc[corrupt["channel"] == channel].reset_index(drop=True)
    clean_channel = clean.loc[clean["channel"] == channel].reset_index(drop=True)
    persistence = _persistence_detector(calibration_report)
    corrupt_state = persistence.score(corrupt_channel)
    clean_state = persistence.score(clean_channel)
    profile_reports = calibration_report["persistence_v2"]["channels"][channel][
        "profiles"
    ]
    profiles: dict[str, Any] = {}
    for profile_name, profile_report in profile_reports.items():
        points: list[dict[str, Any]] = []
        for budget_value in pareto:
            key = str(budget_value)
            if profile_report["mode"] == "instant":
                parameter = float(profile_report["selected_alpha_by_budget"][key])
                corrupt_alarm = corrupt_channel["conformal_p_value"].le(parameter).to_numpy(bool)
                clean_alarm = clean_channel["conformal_p_value"].le(parameter).to_numpy(bool)
                parameter_payload = {"frozen_alpha": parameter}
            elif profile_report["mode"] == "persistence_v2_cumulative":
                parameter = float(profile_report["selected_threshold_h_by_budget"][key])
                corrupt_alarm = corrupt_state["persistence_v2_evaluable"].to_numpy(bool) & (
                    corrupt_state["persistence_v2_state"].to_numpy(float) > parameter
                )
                clean_alarm = clean_state["persistence_v2_evaluable"].to_numpy(bool) & (
                    clean_state["persistence_v2_state"].to_numpy(float) > parameter
                )
                parameter_payload = {"frozen_threshold_h": parameter}
            else:
                raise TruthV2EvaluationContractError(
                    f"Unexpected calibrated profile mode: {profile_report['mode']}"
                )
            point = {
                "pareto_v": float(budget_value),
                **parameter_payload,
                **_detection_point(
                    events, corrupt_channel, corrupt_alarm, clean_channel, clean_alarm
                ),
            }
            points.append(point)
            print(
                f"  {recipe}/{profile_name} V={budget_value}: "
                f"recall={point['event_recall']} burden={point['paired_clean_natural_burden_per_hour']}",
                flush=True,
            )
        profiles[profile_name] = {"mode": profile_report["mode"], "points": points}
    return {"recipe": recipe, "channel": channel, "profiles": profiles}


def _score_cusum(features: pd.DataFrame, detector: VectorPageCUSUM) -> pd.DataFrame:
    raw = detector.score_rows(features)
    result = _with_exposure_bounds(
        features[["flight_id", "timestamp_utc"]].copy(), "timestamp_utc"
    )
    result["cusum_joint_score"] = raw["cusum_joint_score"].to_numpy(float)
    result["cusum_evaluable"] = raw["cusum_evaluable"].to_numpy(bool)
    return result


def _evaluate_cusum_recipe(
    corrupt: pd.DataFrame,
    clean: pd.DataFrame,
    events: pd.DataFrame,
    calibration_report: dict[str, Any],
    pareto: list[float],
) -> dict[str, Any]:
    thresholds = calibration_report["cusum"]["selected_threshold_h_by_budget"]
    points: list[dict[str, Any]] = []
    for budget_value in pareto:
        threshold = float(thresholds[str(budget_value)])
        corrupt_alarm = corrupt["cusum_evaluable"].to_numpy(bool) & (
            corrupt["cusum_joint_score"].to_numpy(float) > threshold
        )
        clean_alarm = clean["cusum_evaluable"].to_numpy(bool) & (
            clean["cusum_joint_score"].to_numpy(float) > threshold
        )
        point = {
            "pareto_v": float(budget_value),
            "frozen_threshold_h": threshold,
            **_detection_point(events, corrupt, corrupt_alarm, clean, clean_alarm),
        }
        points.append(point)
        print(
            f"  {CUSUM_RECIPE}/east_north_cusum V={budget_value}: "
            f"recall={point['event_recall']} burden={point['paired_clean_natural_burden_per_hour']}",
            flush=True,
        )
    return {
        "recipe": CUSUM_RECIPE,
        "channels": list(VECTOR_RESIDUAL_FEATURES),
        "points": points,
    }


def run(run_dir: Path, calibration_dir: Path, corpus_dir: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    training_report = _load_json(run_dir / "training_report.json")
    if training_report["natural_calibration_diagnostic"][
        "magnitude_domination_flagged_at_0_8"
    ] is not False:
        raise TruthV2EvaluationContractError("Magnitude-domination gate is not false")
    run_manifest = _load_json(run_dir / "run_manifest.json")
    if tuple(run_manifest["fit_expansion_days"]) != EXPECTED_EXPANSION_DAYS:
        raise TruthV2EvaluationContractError("Unexpected fit-expansion day set")
    calibration_report_path = calibration_dir / "calibration_report.json"
    calibration_report = _load_json(calibration_report_path)
    if calibration_report["training_report_sha256"] != _sha256_file(
        run_dir / "training_report.json"
    ):
        raise TruthV2EvaluationContractError("Calibration belongs to another training report")
    if calibration_report.get("truth_v2_accessed") is not False:
        raise TruthV2EvaluationContractError("Calibration artifact is not natural-only")

    missing = [name for name in EXPECTED_CORPUS_FILES if not (corpus_dir / name).is_file()]
    if missing:
        raise TruthV2EvaluationContractError(f"Existing truth-v2 corpus is incomplete: {missing}")
    budget_path = Path.cwd() / BUDGET_CONFIG_PATH
    budget = _load_json(budget_path)
    pareto = budget["budget_grid_episodes_per_100_scoreable_flight_hours"]
    if pareto != calibration_report["budget_grid"]:
        raise TruthV2EvaluationContractError("Frozen budget grid differs from calibration")

    train_config_path = Path.cwd() / run_manifest.get(
        "config_path", FROZEN_TRAIN_CONFIG_PATH.as_posix()
    )
    if _sha256_file(train_config_path) != run_manifest.get("config_sha256"):
        raise TruthV2EvaluationContractError("Frozen training config SHA-256 mismatch")
    train_config = _load_json(train_config_path)
    model, scaler, target_channels = _load_checkpoint(run_dir)
    calibrator = _fit_frozen_calibrator(calibration_dir, calibration_report)
    conflicts, quarantined_flights, audit_counts = _audit_truth_v2_ground_conflicts(
        corpus_dir, EXPECTED_CORPUS_FILES
    )

    results: dict[str, Any] = {}
    observability: dict[str, Any] = {}
    for recipe, channel in MODEL_RECIPES.items():
        print(f"truth-v2 paired clean channel={channel}", flush=True)
        clean_features = _load_corpus_features(
            corpus_dir / "clean.parquet", quarantined_flights
        )
        clean_model = _conformal_transform(
            _score_model_long(
                clean_features,
                model=model,
                scaler=scaler,
                target_channels=target_channels,
                train_config=train_config,
                output_channels={channel},
            ),
            calibrator,
        )
        del clean_features
        gc.collect()

        print(f"truth-v2 injected recipe={recipe}", flush=True)
        corrupt_features = _load_corpus_features(
            corpus_dir / f"{recipe}.parquet", quarantined_flights
        )
        events = truth_event_table(corrupt_features)
        observability[recipe] = event_observability_denominators(events)
        eligible = events.loc[events["observable_eligible"].fillna(False)]
        corrupt_model = _conformal_transform(
            _score_model_long(
                corrupt_features,
                model=model,
                scaler=scaler,
                target_channels=target_channels,
                train_config=train_config,
                output_channels={channel},
            ),
            calibrator,
        )
        del corrupt_features
        gc.collect()
        results[recipe] = _evaluate_model_recipe(
            recipe,
            channel,
            corrupt_model,
            clean_model,
            eligible,
            calibration_report,
            pareto,
        )
        del clean_model, corrupt_model, events, eligible
        gc.collect()

    cusum = VectorPageCUSUM.from_dict(calibration_report["cusum"]["detector_template"])
    print("truth-v2 paired clean CUSUM", flush=True)
    clean_features = _load_corpus_features(
        corpus_dir / "clean.parquet", quarantined_flights
    )
    clean_cusum = _score_cusum(clean_features, cusum)
    del clean_features
    gc.collect()

    print(f"truth-v2 injected recipe={CUSUM_RECIPE}", flush=True)
    cusum_features = _load_corpus_features(
        corpus_dir / f"{CUSUM_RECIPE}.parquet", quarantined_flights
    )
    cusum_events = truth_event_table(cusum_features)
    observability[CUSUM_RECIPE] = event_observability_denominators(cusum_events)
    cusum_eligible = cusum_events.loc[cusum_events["observable_eligible"].fillna(False)]
    corrupt_cusum = _score_cusum(cusum_features, cusum)
    del cusum_features
    gc.collect()
    results[CUSUM_RECIPE] = _evaluate_cusum_recipe(
        corrupt_cusum,
        clean_cusum,
        cusum_eligible,
        calibration_report,
        pareto,
    )
    del clean_cusum, corrupt_cusum, cusum_events, cusum_eligible
    gc.collect()

    out_dir.mkdir(parents=True, exist_ok=False)
    quarantine_path = out_dir / "truth_v2_on_ground_quarantine.parquet"
    conflicts.to_parquet(quarantine_path, index=False)
    report = {
        "schema_version": 1,
        "candidate_namespace": "contextual_physics_v2",
        "corpus_dir": str(corpus_dir),
        "corpus_reused_not_regenerated": True,
        "training_report_sha256": _sha256_file(run_dir / "training_report.json"),
        "calibration_report_sha256": _sha256_file(calibration_report_path),
        "budget_config_sha256": _sha256_file(budget_path),
        "budget_grid": pareto,
        "threshold_selection_performed_on_truth_v2": False,
        "out_of_scope_recipes": list(OUT_OF_SCOPE_RECIPES),
        "out_of_scope_reason": (
            "altitude_dropout belongs to the S2 data-quality layer, not the frozen "
            "contextual model/CUSUM physics channels"
        ),
        "data_quality_quarantine": {
            "policy": (
                "exclude_union_of_entire_flights_with_conflicting_on_ground_"
                "at_same_timestamp_from_all_paired_corpus_files"
            ),
            **audit_counts,
            "quarantined_flights": len(quarantined_flights),
            "quarantined_flight_ids_sha256": _canonical_json_sha256(
                sorted(quarantined_flights)
            ),
            "retained_paired_flights": (audit_counts["clean_flights_before_quarantine"] - len(quarantined_flights)),
            "artifact_path": quarantine_path.name,
            "artifact_sha256": _sha256_file(quarantine_path),
            "post_training_user_authorized_deviation": True,
        },
        "event_observability_denominators": observability,
        "results": results,
    }
    _write_json_exclusive(out_dir / "truth_v2_eval_report.json", report)
    _write_checksums(out_dir)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/adsb/runs/20260724_contextual_physics_v2_train_v4"),
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("artifacts/adsb/runs/20260724_contextual_physics_v2_calibration_v1"),
    )
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/adsb/runs/20260724_contextual_physics_v2_truth_v2_eval_v1"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(
        args.run_dir.resolve(),
        args.calibration_dir.resolve(),
        args.corpus_dir.resolve(),
        args.out_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "budget_grid": report["budget_grid"],
                "recipes": sorted(report["results"]),
                "report": str(args.out_dir / "truth_v2_eval_report.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
