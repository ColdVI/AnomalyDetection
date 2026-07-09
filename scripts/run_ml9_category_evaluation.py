"""ML-9 category-matched residual evaluation on frozen SEAD development splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.data.scaling import apply_scaler_params, fit_scaler_params
from src.ml.decision.decision_layers import (
    fit_cusum_policy,
    fit_k_of_n_policy,
    fit_threshold_policy,
)
from src.ml.evaluation.events import (
    event_metrics,
    load_uav_sead_ranges,
    load_uav_sead_ranges_by_category,
    range_mask,
    uav_sead_absolute_us,
)
from src.ml.evaluation.score_fusion import (
    empirical_probability as _empirical_probability,
    last_causal_per_bucket,
    max_score_fusion,
)
from src.ml.features.uav_attack_features import feature_columns
from src.ml.models.modular_iforest import (
    PX4_ML7_CANDIDATE_MODULES,
    PX4_ML9_CANDIDATE_MODULES,
    PX4_ML9_POOLED_EKF_REFERENCE,
    anomaly_scores,
    fit_modular_iforest,
)

FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SILVER_PATH = ROOT / "data/silver/uav_sead_silver.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
LABEL_PATH = ROOT / "data/objectstore/bronze/uav_sead/labels.json"
CUSUM_PATH = ROOT / "artifacts/cusum/uav_sead_cusum_baseline.json"

SCORE_SOURCES = (
    "existing_fusion", "ml9_fusion", "pooled_ekf",
    "dikey_tutarlilik", "kontrol_cevabi", "motor_simetrisi",
)
BUDGETS = {"critical": 2.0, "advisory": 12.0}
MIN_RECALL = {"critical": 0.30, "advisory": 0.50}
MEANINGFUL_RECALL_GAIN = 0.05
DECISION_STRIDE_S = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _score_modules(fitted: dict, scaled: pd.DataFrame, val_ids: set[str]) -> pd.DataFrame:
    out = scaled[["source_id", "t_rel_s", "label"]].copy()
    val_mask = out["source_id"].isin(val_ids).to_numpy()
    for name, item in fitted.items():
        cols = item["feature_columns"]
        values = scaled[cols]
        finite_mask = values.notna().all(axis=1).to_numpy()
        raw = np.full(len(scaled), np.nan)
        if finite_mask.any():
            raw[finite_mask] = anomaly_scores(item["model"], values.loc[finite_mask])
        out[name] = _empirical_probability(raw[val_mask], raw)

    existing = [name for name in PX4_ML7_CANDIDATE_MODULES if name in out]
    ml9 = [name for name in PX4_ML9_CANDIDATE_MODULES if name in out]
    out["existing_fusion"] = max_score_fusion(out, existing)
    out["ml9_fusion"] = max_score_fusion(out, ml9)
    return out


def _one_second_streams(scored: pd.DataFrame) -> pd.DataFrame:
    """Use the last causally available row in each elapsed one-second bucket."""
    keep = ["source_id", "t_rel_s", "label", *SCORE_SOURCES]
    return last_causal_per_bucket(
        scored, stride_seconds=DECISION_STRIDE_S, columns=keep,
    )


def _streams(frame: pd.DataFrame, ids: set[str], score_col: str) -> list[np.ndarray]:
    return [
        group.sort_values("t_rel_s")[score_col].to_numpy(dtype=float)
        for _, group in frame[frame["source_id"].isin(ids)].groupby("source_id")
    ]


def _combined_categories(categories: dict[str, list[tuple[float, float]]]) -> dict:
    result = {name: list(spans) for name, spans in categories.items()}
    actuator = result.get("Actuator Outputs", []) + result.get("Actuator Controls", [])
    if actuator:
        result["Actuator Outputs+Controls"] = actuator
    return result


def _aggregate(details: list[dict]) -> dict:
    events = sum(int(row["n_events"]) for row in details)
    detected = sum(int(row["detected_events"]) for row in details)
    false_alarms = sum(int(row["false_alarm_events"]) for row in details)
    normal_hours = sum(float(row["normal_hours"]) for row in details)
    delays = [delay for row in details for delay in row["detection_delays_s"]]
    return {
        "n_events": events,
        "detected_events": detected,
        "event_onset_recall": detected / events if events else np.nan,
        "false_alarm_events": false_alarms,
        "normal_hours": normal_hours,
        "false_alarms_per_hour": false_alarms / normal_hours if normal_hours else np.nan,
        "avg_detection_time_s": float(np.mean(delays)) if delays else np.nan,
        "max_detection_time_s": float(np.max(delays)) if delays else np.nan,
    }


def _evaluate(
    frame: pd.DataFrame,
    ids: set[str],
    score_col: str,
    policy,
    *,
    t0: dict[str, float],
    ranges: dict[str, list[tuple[float, float]]],
    ranges_by_category: dict[str, dict[str, list[tuple[float, float]]]],
    flight_labels: dict[str, str],
) -> tuple[dict, list[dict], list[dict]]:
    overall_details: list[dict] = []
    label_details: dict[str, list[dict]] = {}
    category_details: dict[str, list[dict]] = {}
    subset = frame[frame["source_id"].isin(ids)]
    for source_id, group in subset.groupby("source_id"):
        group = group.sort_values("t_rel_s")
        times = group["t_rel_s"].to_numpy(dtype=float)
        absolute = uav_sead_absolute_us(times, t0[source_id])
        onsets = policy.apply(group[score_col].to_numpy(dtype=float))

        truth = range_mask(absolute, ranges.get(source_id, []))
        metrics = event_metrics(
            times, truth, onsets.astype(float), 0.5, max_gap_s=2.0,
        )
        overall_details.append(metrics)
        label_details.setdefault(flight_labels[source_id], []).append(metrics)

        for category, spans in _combined_categories(
            ranges_by_category.get(source_id, {})
        ).items():
            category_truth = range_mask(absolute, spans)
            category_metrics = event_metrics(
                times, category_truth, onsets.astype(float), 0.5, max_gap_s=2.0,
            )
            category_details.setdefault(category, []).append(category_metrics)

    by_label = [{"flight_label": key, **_aggregate(value)}
                for key, value in sorted(label_details.items())]
    by_category = [{"annotation_category": key, **_aggregate(value)}
                   for key, value in sorted(category_details.items())]
    return _aggregate(overall_details), by_label, by_category


def _fit_policies(val_streams: list[np.ndarray], budget: float, seed: int) -> dict:
    return {
        "threshold": fit_threshold_policy(
            val_streams, budget, stride_seconds=DECISION_STRIDE_S),
        "k_of_n": fit_k_of_n_policy(
            val_streams, budget, stride_seconds=DECISION_STRIDE_S),
        "cusum": fit_cusum_policy(
            val_streams, budget, stride_seconds=DECISION_STRIDE_S,
            bootstrap_hours=200.0, seed=seed),
    }


def _gate_b(category: pd.DataFrame) -> dict:
    seed_count = int(category["seed"].nunique())
    comparisons = [
        ("Position.Z", "dikey_tutarlilik", "pooled_ekf"),
        ("Actuator Outputs+Controls", "motor_simetrisi", "kontrol_cevabi"),
    ]
    rows = []
    for category_name, candidate, baseline in comparisons:
        for (decision, budget), group in category[
            (category["annotation_category"] == category_name)
            & category["score_source"].isin([candidate, baseline])
        ].groupby(["decision", "budget"]):
            pivot = group.pivot(index="seed", columns="score_source", values="event_onset_recall")
            if candidate not in pivot or baseline not in pivot:
                continue
            gains = pivot[candidate] - pivot[baseline]
            row = {
                "annotation_category": category_name,
                "candidate": candidate,
                "baseline": baseline,
                "decision": decision,
                "budget": budget,
                "candidate_mean_recall": float(pivot[candidate].mean()),
                "baseline_mean_recall": float(pivot[baseline].mean()),
                "mean_recall_gain": float(gains.mean()),
                "positive_seed_count": int((gains > 0).sum()),
            }
            row["meaningful_gain"] = (
                row["mean_recall_gain"] >= MEANINGFUL_RECALL_GAIN
                and row["positive_seed_count"] >= 3
            )
            rows.append(row)
    passed = seed_count >= 5 and any(row["meaningful_gain"] for row in rows)
    return {
        "status": ("passed" if passed else "failed") if seed_count >= 5 else "smoke_only",
        "evaluated_seed_count": seed_count,
        "rule": "matching policy/budget mean recall gain >=0.05 and positive in >=3/5 seeds",
        "comparisons": rows,
    }


def _gate_c(metrics: pd.DataFrame) -> dict:
    seed_count = int(metrics["seed"].nunique())
    rows = []
    selected = metrics[metrics["score_source"] == "ml9_fusion"]
    for (decision, budget), group in selected.groupby(["decision", "budget"]):
        recall = float(group["event_onset_recall"].mean())
        fa = float(group["false_alarms_per_hour"].mean())
        row = {
            "decision": decision,
            "budget": budget,
            "mean_event_onset_recall": recall,
            "mean_false_alarms_per_hour": fa,
            "passed": recall >= MIN_RECALL[budget] and fa <= BUDGETS[budget],
        }
        rows.append(row)
    passed = seed_count >= 5 and any(row["passed"] for row in rows)
    return {"status": ("passed" if passed else "failed") if seed_count >= 5 else "smoke_only",
            "evaluated_seed_count": seed_count,
            "candidates": rows}


def run(run_name: str, split_names: tuple[str, ...] | None = None,
        reuse_run: str | None = None) -> Path:
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = manifest["sources"]["uav_sead"]
    folds = config["splits"]
    if split_names is not None:
        unknown = set(split_names) - set(folds)
        if unknown:
            raise ValueError(f"Unknown split names: {sorted(unknown)}")
        folds = {name: folds[name] for name in split_names}
    holdout = set(folds["split_00"]["final_holdout"])
    development = set().union(*(
        set(split[part]) for split in folds.values() for part in ("train", "val", "test")
    ))
    if development & holdout:
        raise AssertionError("Blind holdout entered the ML-9 telemetry request")

    raw = pd.read_parquet(FEATURE_PATH, filters=[("source_id", "in", sorted(development))])
    if set(raw["source_id"].unique()) & holdout:
        raise AssertionError("Blind holdout rows were read from the feature table")
    silver_time = pd.read_parquet(
        SILVER_PATH, columns=["source_id", "timestamp"],
        filters=[("source_id", "in", sorted(development))],
    )
    if set(silver_time["source_id"].unique()) & holdout:
        raise AssertionError("Blind holdout rows were read from Silver")
    t0 = silver_time.groupby("source_id")["timestamp"].min().to_dict()
    ranges = load_uav_sead_ranges(LABEL_PATH)
    categories = load_uav_sead_ranges_by_category(LABEL_PATH)

    output = ROOT / "artifacts/ml9/uav_sead" / run_name
    output.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    all_labels: list[dict] = []
    all_categories: list[dict] = []
    module_definitions = {
        **PX4_ML9_CANDIDATE_MODULES,
        **PX4_ML9_POOLED_EKF_REFERENCE,
    }

    reused_splits: set[str] = set()
    if reuse_run:
        reuse = ROOT / "artifacts/ml9/uav_sead" / reuse_run
        reuse_manifest = json.loads((reuse / "manifest.json").read_text(encoding="utf-8"))
        if reuse_manifest["split_manifest_sha256"] != _sha256(SPLIT_PATH):
            raise ValueError("Reusable ML-9 run uses a different split manifest")
        if reuse_manifest.get("blind_holdout_read") is not False:
            raise ValueError("Reusable ML-9 run does not prove holdout isolation")
        for relative, expected in reuse_manifest["files"].items():
            if _sha256(reuse / relative) != expected:
                raise ValueError(f"Reusable ML-9 checksum mismatch: {relative}")
        for filename, target in [
            ("metrics.csv", all_metrics),
            ("flight_label_metrics.csv", all_labels),
            ("category_metrics.csv", all_categories),
        ]:
            target.extend(pd.read_csv(reuse / filename).to_dict(orient="records"))
        reused_splits = {str(row["split"]) for row in all_metrics}
        for split_name in reused_splits:
            shutil.copytree(reuse / split_name, output / split_name, dirs_exist_ok=True)

    for split_name, split in folds.items():
        if split_name in reused_splits:
            continue
        seed = int(split["seed"])
        parts = {name: set(split[name]) for name in ("train", "val", "test")}
        if set().union(*parts.values()) & holdout:
            raise AssertionError(f"{split_name}: holdout entered train/val/test")
        scaler = fit_scaler_params(
            raw[raw["source_id"].isin(parts["train"])], feature_columns(raw),
        )
        scaled = apply_scaler_params(raw, scaler)
        fitted = fit_modular_iforest(
            scaled, split, module_definitions, seed=seed, n_jobs=1,
        )
        scored = _one_second_streams(_score_modules(fitted, scaled, parts["val"]))

        seed_dir = output / split_name
        models_dir = seed_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, item in fitted.items():
            joblib.dump(item["model"], models_dir / f"{name}.joblib")
        (seed_dir / "scaler.json").write_text(
            json.dumps(scaler, indent=2), encoding="utf-8")
        (seed_dir / "calibration.json").write_text(json.dumps({
            name: {key: value for key, value in item.items() if key != "model"}
            for name, item in fitted.items()
        }, indent=2), encoding="utf-8")

        policies_json = {}
        for score_source in SCORE_SOURCES:
            val_streams = _streams(scored, parts["val"], score_source)
            for budget_name, budget in BUDGETS.items():
                policies = _fit_policies(val_streams, budget, seed)
                for decision, policy in policies.items():
                    policies_json[f"{score_source}:{budget_name}:{decision}"] = policy.to_dict()
                    overall, by_label, by_category = _evaluate(
                        scored, parts["test"], score_source, policy,
                        t0=t0, ranges=ranges, ranges_by_category=categories,
                        flight_labels=config["flight_labels"],
                    )
                    common = {
                        "split": split_name, "seed": seed,
                        "score_source": score_source, "decision": decision,
                        "budget": budget_name,
                    }
                    all_metrics.append({**common, **overall})
                    all_labels.extend({**common, **row} for row in by_label)
                    all_categories.extend({**common, **row} for row in by_category)
        (seed_dir / "policies.json").write_text(
            json.dumps(policies_json, indent=2), encoding="utf-8")

    metrics = pd.DataFrame(all_metrics)
    label_metrics = pd.DataFrame(all_labels)
    category_metrics = pd.DataFrame(all_categories)
    metrics.to_csv(output / "metrics.csv", index=False)
    label_metrics.to_csv(output / "flight_label_metrics.csv", index=False)
    category_metrics.to_csv(output / "category_metrics.csv", index=False)
    gates = {
        "gate_a": {
            "status": "passed",
            "blind_holdout_telemetry_or_scores_read": False,
            "holdout_flights_locked": len(holdout),
            "scaler_fit": "per-seed normal train only",
            "cusum_feature_baseline_fit": "frozen split_00 normal train only",
            "prefix_invariance": "unit + real-development-flight passed",
        },
        "gate_b": _gate_b(category_metrics),
        "gate_c": _gate_c(metrics),
    }
    (output / "gates.json").write_text(
        json.dumps(_jsonable(gates), indent=2, allow_nan=True), encoding="utf-8")
    shutil.copy2(CUSUM_PATH, output / "cusum_feature_baseline.json")

    files = [path for path in output.rglob("*")
             if path.is_file() and path.name != "manifest.json"]
    artifact_manifest = {
        "artifact_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ML-9 development category evaluation",
        "source": "uav_sead",
        "actual_feature_flights": int(raw["source_id"].nunique()),
        "plan_declared_flights": 412,
        "data_drift_note": "current frozen manifest has 611 parseable flights; existing split retained",
        "blind_holdout_read": False,
        "blind_holdout_flights": len(holdout),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "feature_table_sha256": _sha256(FEATURE_PATH),
        "silver_table_sha256": _sha256(SILVER_PATH),
        "decision_layers": "imported unchanged from src/ml/decision/decision_layers.py",
        "gate_status": {name: value["status"] for name, value in gates.items()},
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): _sha256(path)
            for path in files
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_matrix")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Optional smoke subset, e.g. --splits split_00")
    parser.add_argument("--reuse-run", default=None,
                        help="Reuse completed splits from another checksum-verified ML-9 run")
    args = parser.parse_args()
    output = run(
        args.run_name, tuple(args.splits) if args.splits else None,
        reuse_run=args.reuse_run,
    )
    print(f"ML-9 artifact: {output}")
    print((output / "gates.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
