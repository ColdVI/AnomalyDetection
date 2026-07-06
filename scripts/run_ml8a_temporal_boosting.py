"""ML-8A temporal boosting runner (development data only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.artifacts import load_lstm_bundle, load_modular_iforest_bundle
from src.ml.data.scaling import apply_scaler_params
from src.ml.data.windowing import build_windows
from src.ml.decision.decision_layers import (
    fit_cusum_policy,
    fit_k_of_n_policy,
    fit_threshold_policy,
    policy_from_dict,
)
from src.ml.evaluation.events import (
    event_metrics,
    load_uav_sead_ranges,
    range_mask,
    uav_sead_absolute_us,
)
from src.ml.features.window_descriptors import (
    build_window_descriptors,
    descriptor_schema_sha256,
    label_windows_from_intervals,
)
from src.ml.models.temporal_boosting import (
    binary_window_metrics,
    fit_temporal_boosting,
    gain_importance,
    predict_temporal_boosting,
)
from src.ml.models.lstm_autoencoder import reconstruction_scores
from src.ml.models.modular_iforest import anomaly_scores

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_filtered_parquet(path: Path, columns: list[str], ids: set[str]) -> pd.DataFrame:
    if not ids:
        raise ValueError("Cannot read an empty flight set")
    return pd.read_parquet(path, columns=columns, filters=[("source_id", "in", sorted(ids))])


def _training_rows(labeled: pd.DataFrame, normal_ids: set[str]) -> pd.DataFrame:
    """Normal windows are negative; anomalous range-outside windows are excluded."""

    normal = labeled[labeled["source_id"].isin(normal_ids)].copy()
    normal["target"] = 0
    positive = labeled[
        ~labeled["source_id"].isin(normal_ids) & (labeled["train_label"] == "positive")
    ].copy()
    positive["target"] = 1
    return pd.concat([normal, positive], ignore_index=True)


def _save_smoke_bundle(run_dir: Path, fit, metrics: dict, schema_path: Path, counts: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.txt"
    scaler_path = run_dir / "scaler.json"
    metrics_path = run_dir / "metrics.json"
    importance_path = run_dir / "feature_importance.json"
    schema_copy = run_dir / "descriptor_schema_v1.json"
    fit.model.booster_.save_model(str(model_path))
    scaler_path.write_text(json.dumps(fit.scaler, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps({**metrics, "counts": counts}, indent=2), encoding="utf-8")
    importance_path.write_text(json.dumps(gain_importance(fit), indent=2), encoding="utf-8")
    schema_copy.write_bytes(schema_path.read_bytes())
    files = [model_path, scaler_path, metrics_path, importance_path, schema_copy]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "split_00_smoke",
        "blind_holdout_read": False,
        "scaler_fit": "train_only",
        "descriptor_schema_sha256": descriptor_schema_sha256(schema_copy),
        "files": {path.name: _sha256(path) for path in files},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_sead_smoke(split_name: str = "split_00") -> dict:
    feature_root = ROOT / "data/gold/ml_features"
    manifest = json.loads((feature_root / "split_manifest.json").read_text(encoding="utf-8"))
    config = manifest["sources"]["uav_sead"]
    split = config["supervised_splits"][split_name]
    parts = {name: set(split[name]) for name in ("train", "val", "test")}
    holdout = set(split["final_holdout"])
    requested = set().union(*parts.values())
    assert not requested & holdout, "Blind holdout attempted to enter an ML-8A stream"

    schema_path = ROOT / "artifacts/ml8a/descriptor_schema_v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    channels = schema["sources"]["uav_sead"]["channels"]
    feature_path = feature_root / "uav_sead/uav_sead_ml_features.parquet"
    raw = _read_filtered_parquet(feature_path, ["source_id", "t_rel_s", *channels], requested)
    assert set(raw["source_id"].unique()) <= requested
    assert not set(raw["source_id"].unique()) & holdout

    silver_path = ROOT / "data/silver/uav_sead_silver.parquet"
    silver_time = _read_filtered_parquet(silver_path, ["source_id", "timestamp"], requested)
    t0 = silver_time.groupby("source_id")["timestamp"].min().to_dict()
    ranges = load_uav_sead_ranges(ROOT / "data/objectstore/bronze/uav_sead/labels.json")

    descriptors = build_window_descriptors(raw, channels)
    labeled = label_windows_from_intervals(
        descriptors,
        {sid: ranges.get(sid, []) for sid in requested},
        t0_by_source=t0,
        interval_unit="absolute_us",
    )
    prepared = {}
    for part, ids in parts.items():
        frame = labeled[labeled["source_id"].isin(ids)].copy()
        prepared[part] = _training_rows(frame, set(split[f"{part}_normal"])) if part != "test" else frame

    fit = fit_temporal_boosting(prepared["train"], prepared["val"], seed=int(split["seed"]))
    test = prepared["test"].copy()
    test["target"] = (test["anomaly_overlap_fraction"] >= 0.5).astype(int)
    test["score"] = predict_temporal_boosting(fit, test)
    metrics = binary_window_metrics(test["target"], test["score"])
    metrics["validation_auprc"] = fit.validation_auprc
    metrics["best_iteration"] = int(fit.model.best_iteration_)
    counts = {
        part: {
            "rows": len(frame),
            "positive": int(frame.get("target", pd.Series(dtype=int)).sum()) if "target" in frame else None,
        }
        for part, frame in prepared.items()
    }
    run_dir = ROOT / f"artifacts/ml8a/uav_sead/{split_name}_smoke"
    _save_smoke_bundle(run_dir, fit, metrics, schema_path, counts)
    return {**metrics, "counts": counts, "run_dir": str(run_dir)}


def _align_score(endpoints: pd.DataFrame, scored: pd.DataFrame, score_col: str) -> np.ndarray:
    values = np.full(len(endpoints), np.nan)
    for sid, index in endpoints.groupby("source_id", sort=False).groups.items():
        left = endpoints.loc[index, ["t_rel_s"]].sort_values("t_rel_s")
        right = scored[scored["source_id"] == sid][["t_rel_s", score_col]].sort_values("t_rel_s")
        if right.empty:
            continue
        merged = pd.merge_asof(left, right, on="t_rel_s", direction="backward")
        values[left.index] = merged[score_col].to_numpy()
    return values


def _if_scores_for(raw: pd.DataFrame, bundle: Path, score_name: str = "if_fusion") -> pd.DataFrame:
    fitted, _ = load_modular_iforest_bundle(bundle)
    scaler = json.loads((bundle / "scaler.json").read_text(encoding="utf-8"))
    scaled = apply_scaler_params(raw, scaler)
    module_scores = [
        anomaly_scores(item["model"], scaled[item["feature_columns"]])
        for item in fitted.values()
    ]
    return pd.DataFrame({
        "source_id": raw["source_id"].to_numpy(),
        "t_rel_s": raw["t_rel_s"].to_numpy(),
        score_name: np.max(np.column_stack(module_scores), axis=1),
    })


def _lstm_scores_for(raw: pd.DataFrame, bundle: Path, score_name: str) -> pd.DataFrame:
    model, scaler, _, manifest = load_lstm_bundle(bundle)
    cols = manifest["feature_columns"]
    scaled = apply_scaler_params(raw, scaler)
    sequence = scaled[["source_id", "t_rel_s", "label"]].copy()
    for col in cols:
        sequence[col] = scaled[col].where(raw[col].notna())
    x, mask, meta = build_windows(
        sequence,
        cols,
        window=int(manifest["window"]),
        stride=int(manifest["stride"]),
        max_gap_s=float(manifest["max_gap_s"]),
    )
    return pd.DataFrame({
        "source_id": meta["source_id"],
        "t_rel_s": meta["t_end"],
        score_name: reconstruction_scores(model, x, mask),
    })


def _empirical_probability(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.sort(reference[np.isfinite(reference)])
    if not len(finite):
        raise ValueError("Normal validation reference is empty")
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    result[valid] = (np.searchsorted(finite, values[valid], side="right") + 0.5) / (len(finite) + 1.0)
    return result


def _streams(frame: pd.DataFrame, ids: set[str], score_col: str) -> list[np.ndarray]:
    return [
        group.sort_values("t_rel_s")[score_col].to_numpy()
        for sid, group in frame[frame["source_id"].isin(ids)].groupby("source_id")
    ]


def _evaluate_policy(frame: pd.DataFrame, ids: set[str], score_col: str, policy) -> dict:
    totals = {"n_events": 0, "detected_events": 0, "false_alarm_events": 0,
              "alarm_onsets": 0, "normal_hours": 0.0, "preexisting_alarm_events": 0}
    delays: list[float] = []
    true_alarm_onsets = 0
    families: dict[str, dict[str, int]] = {}
    for _, group in frame[frame["source_id"].isin(ids)].groupby("source_id"):
        group = group.sort_values("t_rel_s")
        onsets = policy.apply(group[score_col].to_numpy())
        y = group["is_anomaly"].to_numpy(dtype=bool)
        metrics = event_metrics(
            group["t_rel_s"].to_numpy(), y, onsets.astype(float), 0.5, max_gap_s=2.0
        )
        for key in totals:
            totals[key] += metrics[key]
        delays.extend(metrics["detection_delays_s"])
        true_alarm_onsets += int((onsets & y).sum())
        if metrics["n_events"]:
            family = str(group["anomaly_family"].iloc[0]) if "anomaly_family" in group else "unknown"
            item = families.setdefault(family, {"events": 0, "detected": 0})
            item["events"] += int(metrics["n_events"])
            item["detected"] += int(metrics["detected_events"])
    return {
        **totals,
        "event_onset_recall": (totals["detected_events"] / totals["n_events"]
                               if totals["n_events"] else np.nan),
        "event_precision": (true_alarm_onsets / totals["alarm_onsets"]
                            if totals["alarm_onsets"] else np.nan),
        "median_detection_delay_s": float(np.median(delays)) if delays else np.nan,
        "p90_detection_delay_s": float(np.quantile(delays, 0.9)) if delays else np.nan,
        "avg_detection_time_s": float(np.mean(delays)) if delays else np.nan,
        "max_detection_time_s": float(np.max(delays)) if delays else np.nan,
        "false_alarms_per_hour": (totals["false_alarm_events"] / totals["normal_hours"]
                                  if totals["normal_hours"] else np.nan),
        "family_onset_recall": {
            family: {
                **counts,
                "recall": counts["detected"] / counts["events"] if counts["events"] else np.nan,
            }
            for family, counts in sorted(families.items())
        },
    }


def run_sead_matrix() -> dict:
    feature_root = ROOT / "data/gold/ml_features"
    manifest = json.loads((feature_root / "split_manifest.json").read_text(encoding="utf-8"))
    config = manifest["sources"]["uav_sead"]
    folds = config["supervised_splits"]
    holdout = set(folds["split_00"]["final_holdout"])
    development = set().union(*(set(fold[p]) for fold in folds.values() for p in ("train", "val", "test")))
    assert not development & holdout

    schema_path = ROOT / "artifacts/ml8a/descriptor_schema_v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    channels = schema["sources"]["uav_sead"]["channels"]
    raw = pd.read_parquet(
        feature_root / "uav_sead/uav_sead_ml_features.parquet",
        filters=[("source_id", "in", sorted(development))],
    )
    assert not set(raw["source_id"].unique()) & holdout
    budgets = {"critical": 2.0, "advisory": 12.0}
    all_results: list[dict] = []
    output = ROOT / "artifacts/ml8a/uav_sead/full_matrix_gapfix"
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "development_score_frame.parquet"
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
    else:
        descriptors = build_window_descriptors(raw[["source_id", "t_rel_s", *channels]], channels)
        silver = _read_filtered_parquet(
            ROOT / "data/silver/uav_sead_silver.parquet", ["source_id", "timestamp"], development
        )
        t0 = silver.groupby("source_id")["timestamp"].min().to_dict()
        ranges = load_uav_sead_ranges(ROOT / "data/objectstore/bronze/uav_sead/labels.json")
        label_meta = json.loads(
            (ROOT / "data/objectstore/bronze/uav_sead/labels.json").read_text(encoding="utf-8")
        )
        frame = label_windows_from_intervals(
            descriptors, {sid: ranges.get(sid, []) for sid in development},
            t0_by_source=t0, interval_unit="absolute_us"
        )
        anomaly = np.zeros(len(frame), dtype=bool)
        for sid, index in frame.groupby("source_id").groups.items():
            anomaly[index] = range_mask(
                uav_sead_absolute_us(frame.loc[index, "t_rel_s"], t0[sid]), ranges.get(sid, [])
            )
        frame["is_anomaly"] = anomaly
        frame["anomaly_family"] = frame["source_id"].map(
            {
                sid: str(meta.get("label", "unknown"))
                .replace("_position_anomaly", "_position")
                .replace("_anomaly", "")
                .replace("_fault", "")
                for sid, meta in label_meta.items()
            }
        )
        frame["if_fusion"] = _align_score(
            frame,
            _if_scores_for(raw, ROOT / "artifacts/models/uav_sead/ml7_candidate_iforest"),
            "if_fusion",
        )
        frame["lstm_retrained"] = _align_score(
            frame,
            _lstm_scores_for(
                raw, ROOT / "artifacts/models/uav_sead/ml8a_retrained_lstm_ae", "lstm_retrained"
            ),
            "lstm_retrained",
        )
        frame.to_parquet(cache_path, index=False)
    for split_name, split in folds.items():
        ids = {part: set(split[part]) for part in ("train", "val", "test")}
        labeled = frame[frame["source_id"].isin(ids["train"] | ids["val"] | ids["test"])].copy()
        train = _training_rows(labeled[labeled["source_id"].isin(ids["train"])], set(split["train_normal"]))
        val = _training_rows(labeled[labeled["source_id"].isin(ids["val"])], set(split["val_normal"]))
        fit = fit_temporal_boosting(train, val, seed=int(split["seed"]))
        labeled["lightgbm"] = predict_temporal_boosting(fit, labeled)
        seed_dir = output / split_name
        seed_dir.mkdir(parents=True, exist_ok=True)
        fit.model.booster_.save_model(str(seed_dir / "model.txt"))
        (seed_dir / "scaler.json").write_text(json.dumps(fit.scaler, indent=2), encoding="utf-8")
        (seed_dir / "feature_importance.json").write_text(
            json.dumps(gain_importance(fit), indent=2), encoding="utf-8"
        )

        policies_path = seed_dir / "policy.json"
        policies_json: dict = (
            json.loads(policies_path.read_text(encoding="utf-8")) if policies_path.exists() else {}
        )
        seed_results: list[dict] = []
        for score_source in ("lightgbm", "if_fusion", "lstm_retrained"):
            score_col = score_source
            if score_source != "lightgbm":
                reference_mask = labeled["source_id"].isin(set(split["val_normal"]))
                transformed = _empirical_probability(
                    labeled.loc[reference_mask, score_source].to_numpy(), labeled[score_source].to_numpy()
                )
                score_col = f"{score_source}_prob"
                labeled[score_col] = transformed
            val_streams = _streams(labeled, set(split["val_normal"]), score_col)
            test_frame = labeled[labeled["source_id"].isin(ids["test"])].copy()
            target = (test_frame["anomaly_overlap_fraction"] >= 0.5).astype(int)
            window = binary_window_metrics(target, test_frame[score_col])
            for budget_name, budget in budgets.items():
                fitters = {
                    "threshold": lambda: fit_threshold_policy(val_streams, budget),
                    "k_of_n": lambda: fit_k_of_n_policy(val_streams, budget),
                    "cusum": lambda: fit_cusum_policy(
                        val_streams, budget, bootstrap_hours=200.0, seed=int(split["seed"])
                    ),
                }
                policies = {}
                for decision, fitter in fitters.items():
                    key = f"{score_source}:{budget_name}:{decision}"
                    policies[decision] = (
                        policy_from_dict(policies_json[key]) if key in policies_json else fitter()
                    )
                    policies_json[key] = policies[decision].to_dict()
                    policies_path.write_text(json.dumps(policies_json, indent=2), encoding="utf-8")
                for decision, policy in policies.items():
                    result = _evaluate_policy(labeled, ids["test"], score_col, policy)
                    row = {
                        "split": split_name, "seed": int(split["seed"]),
                        "score_source": score_source, "decision": decision,
                        "budget": budget_name, **window, **result,
                    }
                    all_results.append(row)
                    seed_results.append(row)
        policies_json["existing_lstm"] = {
            "status": "N/A", "reason": "No pre-existing UAV-SEAD LSTM-AE artifact"
        }
        policies_path.write_text(json.dumps(policies_json, indent=2), encoding="utf-8")
        (seed_dir / "metrics.json").write_text(
            json.dumps(seed_results, indent=2, allow_nan=True), encoding="utf-8"
        )

    metrics_path = output / "metrics.json"
    metrics_payload = {
        "matrix_status": "3 available score rows x 3 decisions; original LSTM baseline N/A",
        "original_lstm_baseline": "N/A",
        "results": all_results,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, allow_nan=True), encoding="utf-8")
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    matrix_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "blind_holdout_read": False,
        "descriptor_schema_sha256": descriptor_schema_sha256(schema_path),
        "retrained_lstm_is_not_original_baseline": True,
        "files": {str(path.relative_to(output)).replace("\\", "/"): _sha256(path) for path in files},
    }
    (output / "manifest.json").write_text(json.dumps(matrix_manifest, indent=2), encoding="utf-8")
    return metrics_payload


def run_alfa_matrix() -> dict:
    """Run the frozen five-seed ALFA protocol without post-SEAD tuning."""

    feature_root = ROOT / "data/gold/ml_features"
    manifest = json.loads((feature_root / "split_manifest.json").read_text(encoding="utf-8"))
    config = manifest["sources"]["alfa"]
    folds = config["supervised_splits"]
    development = set().union(*(set(fold[p]) for fold in folds.values() for p in ("train", "val", "test")))
    exploration = set(config["splits"]["split_00"].get("exploration", []))
    assert not development & exploration

    schema_path = ROOT / "artifacts/ml8a/descriptor_schema_v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    channels = schema["sources"]["alfa"]["channels"]
    raw = pd.read_parquet(
        feature_root / "alfa/alfa_ml_features.parquet",
        filters=[("source_id", "in", sorted(development))],
    )
    assert not set(raw["source_id"].unique()) & exploration

    output = ROOT / "artifacts/ml8a/alfa/full_matrix_gapfix"
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "development_score_frame.parquet"
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
    else:
        descriptors = build_window_descriptors(raw[["source_id", "t_rel_s", *channels]], channels)
        intervals: dict[str, list[tuple[float, float]]] = {}
        onset: dict[str, float] = {}
        for sid, group in raw.groupby("source_id"):
            group = group.sort_values("t_rel_s")
            anomalous = group[~group["label"].isin(["normal", "unknown"])]
            if anomalous.empty:
                intervals[sid] = []
                continue
            start = float(anomalous["t_rel_s"].iloc[0])
            end = float(group["t_rel_s"].max())
            dt = float(group["t_rel_s"].diff().median())
            intervals[sid] = [(start, end + (dt if np.isfinite(dt) else 0.0))]
            onset[sid] = start
        frame = label_windows_from_intervals(descriptors, intervals)
        frame["is_anomaly"] = [
            sid in onset and float(t) >= onset[sid]
            for sid, t in frame[["source_id", "t_rel_s"]].itertuples(index=False, name=None)
        ]
        frame["anomaly_family"] = frame["source_id"].map(config["flight_labels"])
        frame["if_fusion"] = _align_score(
            frame,
            _if_scores_for(raw, ROOT / "artifacts/models/alfa/ml6_modular_iforest"),
            "if_fusion",
        )
        frame["lstm_existing"] = _align_score(
            frame,
            _lstm_scores_for(raw, ROOT / "artifacts/models/alfa/ml6_lstm_ae", "lstm_existing"),
            "lstm_existing",
        )
        frame.to_parquet(cache_path, index=False)

    budgets = {"critical": 2.0, "advisory": 12.0}
    all_results: list[dict] = []
    for split_name, split in folds.items():
        ids = {part: set(split[part]) for part in ("train", "val", "test")}
        labeled = frame[frame["source_id"].isin(ids["train"] | ids["val"] | ids["test"])].copy()
        train = _training_rows(
            labeled[labeled["source_id"].isin(ids["train"])], set(split["train_normal"])
        )
        val = _training_rows(
            labeled[labeled["source_id"].isin(ids["val"])], set(split["val_normal"])
        )
        fit = fit_temporal_boosting(train, val, seed=int(split["seed"]))
        labeled["lightgbm"] = predict_temporal_boosting(fit, labeled)
        seed_dir = output / split_name
        seed_dir.mkdir(parents=True, exist_ok=True)
        fit.model.booster_.save_model(str(seed_dir / "model.txt"))
        (seed_dir / "scaler.json").write_text(json.dumps(fit.scaler, indent=2), encoding="utf-8")
        (seed_dir / "feature_importance.json").write_text(
            json.dumps(gain_importance(fit), indent=2), encoding="utf-8"
        )
        policies_path = seed_dir / "policy.json"
        policies_json = json.loads(policies_path.read_text()) if policies_path.exists() else {}
        seed_results: list[dict] = []
        for score_source in ("lightgbm", "if_fusion", "lstm_existing"):
            score_col = score_source
            if score_source != "lightgbm":
                reference_mask = labeled["source_id"].isin(set(split["val_normal"]))
                score_col = f"{score_source}_prob"
                labeled[score_col] = _empirical_probability(
                    labeled.loc[reference_mask, score_source].to_numpy(),
                    labeled[score_source].to_numpy(),
                )
            val_streams = _streams(labeled, set(split["val_normal"]), score_col)
            test_frame = labeled[labeled["source_id"].isin(ids["test"])]
            window = binary_window_metrics(
                (test_frame["anomaly_overlap_fraction"] >= 0.5).astype(int),
                test_frame[score_col],
            )
            for budget_name, budget in budgets.items():
                fitters = {
                    "threshold": lambda: fit_threshold_policy(val_streams, budget),
                    "k_of_n": lambda: fit_k_of_n_policy(val_streams, budget),
                    "cusum": lambda: fit_cusum_policy(
                        val_streams, budget, bootstrap_hours=200.0, seed=int(split["seed"])
                    ),
                }
                for decision, fitter in fitters.items():
                    key = f"{score_source}:{budget_name}:{decision}"
                    policy = policy_from_dict(policies_json[key]) if key in policies_json else fitter()
                    policies_json[key] = policy.to_dict()
                    policies_path.write_text(json.dumps(policies_json, indent=2), encoding="utf-8")
                    result = _evaluate_policy(labeled, ids["test"], score_col, policy)
                    row = {
                        "split": split_name,
                        "seed": int(split["seed"]),
                        "score_source": score_source,
                        "decision": decision,
                        "budget": budget_name,
                        **window,
                        **result,
                    }
                    seed_results.append(row)
                    all_results.append(row)
        (seed_dir / "metrics.json").write_text(
            json.dumps(seed_results, indent=2, allow_nan=True), encoding="utf-8"
        )

    metrics_payload = {"matrix_status": "complete 3 score x 3 decision x 5 seed", "results": all_results}
    (output / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "evaluation_status": "development_only",
                "descriptor_schema_sha256": descriptor_schema_sha256(schema_path),
                "files": {
                    str(path.relative_to(output)).replace("\\", "/"): _sha256(path) for path in files
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["uav_sead"], default="uav_sead")
    parser.add_argument("--split", default="split_00")
    parser.add_argument("--smoke", action="store_true", help="Run descriptor -> LightGBM window sanity")
    parser.add_argument("--full-matrix", action="store_true")
    parser.add_argument("--alfa-matrix", action="store_true")
    args = parser.parse_args()
    if args.alfa_matrix:
        result = run_alfa_matrix()
    elif args.full_matrix:
        result = run_sead_matrix()
    elif args.smoke:
        result = run_sead_smoke(args.split)
    else:
        parser.error("Choose --smoke or --full-matrix")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
