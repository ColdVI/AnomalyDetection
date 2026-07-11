"""Evaluate ML-10 zero-shot Chronos residuals on frozen SEAD development splits."""

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

from scripts.run_ml9_category_evaluation import (
    BUDGETS,
    MIN_RECALL,
    _evaluate,
    _jsonable,
    _one_second_streams,
    _score_modules,
    _streams,
)
from src.ml.data.scaling import apply_scaler_params
from src.ml.decision.decision_layers import (
    fit_cusum_policy,
    fit_k_of_n_policy,
    fit_threshold_policy,
)
from src.ml.evaluation.events import (
    load_uav_sead_ranges,
    load_uav_sead_ranges_by_category,
)
from src.ml.evaluation.score_fusion import (
    empirical_probability as _empirical_probability,
    last_causal_per_bucket,
    max_score_fusion,
)

FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SCORE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml10_forecast_residual.parquet"
SILVER_PATH = ROOT / "data/silver/uav_sead_silver.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
LABEL_PATH = ROOT / "data/objectstore/bronze/uav_sead/labels.json"
PREFLIGHT_PATH = ROOT / "artifacts/ml10/uav_sead/preflight_check.json"
FEASIBILITY_PATH = ROOT / "artifacts/ml10/uav_sead/feasibility_check.json"
PRECOMPUTE_DIR = ROOT / "artifacts/ml10/uav_sead/precompute"
ML9_DIR = ROOT / "artifacts/ml9/uav_sead/full_matrix"

NEW_SCORE_SOURCES = ("chronos_dikey", "chronos_motor", "ml10_fusion")
ALL_SCORE_SOURCES = (
    "existing_fusion", "ml9_fusion", "dikey_tutarlilik", "motor_simetrisi",
    *NEW_SCORE_SOURCES,
)
DECISION_STRIDE_S = 1.0
MEANINGFUL_RECALL_GAIN = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest.get("files", {}).items():
        path = directory / relative
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"Checksum mismatch in {directory}: {relative}")
    return manifest


def _fit_policies(val_streams: list[np.ndarray], budget: float, seed: int) -> dict:
    return {
        "threshold": fit_threshold_policy(
            val_streams, budget, stride_seconds=DECISION_STRIDE_S,
        ),
        "k_of_n": fit_k_of_n_policy(
            val_streams, budget, stride_seconds=DECISION_STRIDE_S,
        ),
        "cusum": fit_cusum_policy(
            val_streams,
            budget,
            stride_seconds=DECISION_STRIDE_S,
            bootstrap_hours=200.0,
            seed=seed,
        ),
    }


def _load_ml9_models(split_name: str) -> tuple[dict, dict]:
    split_dir = ML9_DIR / split_name
    scaler = json.loads((split_dir / "scaler.json").read_text(encoding="utf-8"))
    calibration = json.loads(
        (split_dir / "calibration.json").read_text(encoding="utf-8")
    )
    fitted = {
        name: {
            **values,
            "model": joblib.load(split_dir / "models" / f"{name}.joblib"),
        }
        for name, values in calibration.items()
    }
    return scaler, fitted


def _chronos_streams(
    module_streams: pd.DataFrame,
    residual_rows: pd.DataFrame,
    val_ids: set[str],
) -> tuple[pd.DataFrame, dict]:
    residual_streams = last_causal_per_bucket(
        residual_rows,
        stride_seconds=DECISION_STRIDE_S,
        columns=[
            "source_id", "t_rel_s", "chronos_alt_residual",
            "chronos_actuator_std_residual",
        ],
    )
    out = module_streams.merge(
        residual_streams, on=["source_id", "t_rel_s"], how="left", validate="one_to_one",
    )
    val_mask = out["source_id"].isin(val_ids).to_numpy()
    mapping = {
        "chronos_dikey": "chronos_alt_residual",
        "chronos_motor": "chronos_actuator_std_residual",
    }
    calibration = {}
    for target, raw_column in mapping.items():
        reference = out.loc[val_mask, raw_column].to_numpy(dtype=float)
        out[target] = _empirical_probability(
            reference, out[raw_column].to_numpy(dtype=float),
        )
        finite = reference[np.isfinite(reference)]
        calibration[target] = {
            "raw_column": raw_column,
            "normal_validation_count": len(finite),
            "normal_validation_min": float(np.min(finite)),
            "normal_validation_median": float(np.median(finite)),
            "normal_validation_max": float(np.max(finite)),
        }
    out["ml10_fusion"] = max_score_fusion(
        out, ["existing_fusion", "chronos_dikey", "chronos_motor"],
    )
    return out, calibration


def _gate_b(category: pd.DataFrame) -> dict:
    seed_count = int(category["seed"].nunique())
    comparisons = [
        ("Position.Z", "chronos_dikey", "dikey_tutarlilik"),
        ("Actuator Outputs+Controls", "chronos_motor", "motor_simetrisi"),
    ]
    rows = []
    for category_name, candidate, baseline in comparisons:
        selected = category[
            (category["annotation_category"] == category_name)
            & category["score_source"].isin([candidate, baseline])
        ]
        for (decision, budget), group in selected.groupby(["decision", "budget"]):
            pivot = group.pivot(
                index="seed", columns="score_source", values="event_onset_recall",
            )
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
    selected = metrics[metrics["score_source"] == "ml10_fusion"]
    for (decision, budget), group in selected.groupby(["decision", "budget"]):
        recall = float(group["event_onset_recall"].mean())
        false_alarms = float(group["false_alarms_per_hour"].mean())
        rows.append({
            "decision": decision,
            "budget": budget,
            "mean_event_onset_recall": recall,
            "mean_false_alarms_per_hour": false_alarms,
            "passed": recall >= MIN_RECALL[budget] and false_alarms <= BUDGETS[budget],
        })
    passed = seed_count >= 5 and any(row["passed"] for row in rows)
    return {
        "status": ("passed" if passed else "failed") if seed_count >= 5 else "smoke_only",
        "evaluated_seed_count": seed_count,
        "candidates": rows,
    }


def run(run_name: str, split_names: tuple[str, ...] | None = None) -> Path:
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    feasibility = json.loads(FEASIBILITY_PATH.read_text(encoding="utf-8"))
    precompute = _verify_manifest(PRECOMPUTE_DIR)
    ml9_manifest = _verify_manifest(ML9_DIR)
    if preflight["model"]["status"] != "passed":
        raise RuntimeError("ML-10 preflight did not pass")
    if feasibility.get("full_run_authorized") is not True:
        raise RuntimeError("ML-10 feasibility did not authorize the full run")
    if precompute.get("zero_shot") is not True or precompute.get("training_or_gradient_updates") is not False:
        raise RuntimeError("Precompute does not prove zero-shot inference")
    if _sha256(SCORE_PATH) != precompute["score_table_sha256"]:
        raise ValueError("Forecast-residual score table checksum mismatch")
    if _sha256(FEATURE_PATH) != ml9_manifest["feature_table_sha256"]:
        raise ValueError("ML-9 frozen model input feature checksum mismatch")

    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = manifest["sources"]["uav_sead"]
    folds = config["splits"]
    if split_names is not None:
        unknown = set(split_names) - set(folds)
        if unknown:
            raise ValueError(f"Unknown split names: {sorted(unknown)}")
        folds = {name: folds[name] for name in split_names}
    holdout = set(config["splits"]["split_00"]["final_holdout"])
    development = set().union(*(
        set(split[part]) for split in folds.values() for part in ("train", "val", "test")
    ))
    if development & holdout:
        raise AssertionError("Blind holdout entered the ML-10 development request")

    raw = pd.read_parquet(FEATURE_PATH, filters=[("source_id", "in", sorted(development))])
    residual = pd.read_parquet(SCORE_PATH, filters=[("source_id", "in", sorted(development))])
    silver_time = pd.read_parquet(
        SILVER_PATH,
        columns=["source_id", "timestamp"],
        filters=[("source_id", "in", sorted(development))],
    )
    for name, frame in (("feature", raw), ("residual", residual), ("silver", silver_time)):
        if set(frame["source_id"].unique()) & holdout:
            raise AssertionError(f"Blind holdout rows were read from {name} table")
    t0 = silver_time.groupby("source_id")["timestamp"].min().to_dict()
    ranges = load_uav_sead_ranges(LABEL_PATH)
    categories = load_uav_sead_ranges_by_category(LABEL_PATH)

    output = ROOT / "artifacts/ml10/uav_sead" / run_name
    output.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    all_labels: list[dict] = []
    all_categories: list[dict] = []

    # Reuse the checksum-verified ML-9 rows as the predeclared Gate-B baselines.
    selected_splits = set(folds)
    old_metrics = pd.read_csv(ML9_DIR / "metrics.csv")
    old_labels = pd.read_csv(ML9_DIR / "flight_label_metrics.csv")
    old_categories = pd.read_csv(ML9_DIR / "category_metrics.csv")
    all_metrics.extend(old_metrics[
        old_metrics["split"].isin(selected_splits)
        & old_metrics["score_source"].isin(["existing_fusion", "ml9_fusion"])
    ].to_dict(orient="records"))
    all_labels.extend(old_labels[
        old_labels["split"].isin(selected_splits)
        & old_labels["score_source"].isin(["existing_fusion", "ml9_fusion"])
    ].to_dict(orient="records"))
    all_categories.extend(old_categories[
        old_categories["split"].isin(selected_splits)
        & old_categories["score_source"].isin([
            "existing_fusion", "ml9_fusion", "dikey_tutarlilik", "motor_simetrisi",
        ])
    ].to_dict(orient="records"))

    for split_name, split in folds.items():
        seed = int(split["seed"])
        parts = {name: set(split[name]) for name in ("train", "val", "test")}
        if set().union(*parts.values()) & holdout:
            raise AssertionError(f"{split_name}: holdout entered train/val/test")
        scaler, fitted = _load_ml9_models(split_name)
        scaled = apply_scaler_params(raw, scaler)
        module_streams = _one_second_streams(
            _score_modules(fitted, scaled, parts["val"]),
        )
        scored, calibration = _chronos_streams(module_streams, residual, parts["val"])

        split_dir = output / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "chronos_calibration.json").write_text(
            json.dumps(calibration, indent=2), encoding="utf-8",
        )
        policies_json = {}
        for score_source in NEW_SCORE_SOURCES:
            val_streams = _streams(scored, parts["val"], score_source)
            for budget_name, budget in BUDGETS.items():
                policies = _fit_policies(val_streams, budget, seed)
                for decision, policy in policies.items():
                    policies_json[f"{score_source}:{budget_name}:{decision}"] = policy.to_dict()
                    overall, by_label, by_category = _evaluate(
                        scored,
                        parts["test"],
                        score_source,
                        policy,
                        t0=t0,
                        ranges=ranges,
                        ranges_by_category=categories,
                        flight_labels=config["flight_labels"],
                    )
                    common = {
                        "split": split_name,
                        "seed": seed,
                        "score_source": score_source,
                        "decision": decision,
                        "budget": budget_name,
                    }
                    all_metrics.append({**common, **overall})
                    all_labels.extend({**common, **row} for row in by_label)
                    all_categories.extend({**common, **row} for row in by_category)
        (split_dir / "policies.json").write_text(
            json.dumps(policies_json, indent=2), encoding="utf-8",
        )

    metrics = pd.DataFrame(all_metrics)
    label_metrics = pd.DataFrame(all_labels)
    category_metrics = pd.DataFrame(all_categories)
    metrics.to_csv(output / "metrics.csv", index=False)
    label_metrics.to_csv(output / "flight_label_metrics.csv", index=False)
    category_metrics.to_csv(output / "category_metrics.csv", index=False)
    gates = {
        "gate_a": {
            "status": "passed",
            "mandatory_preflight": "passed",
            "mandatory_feasibility": "passed; full development at 1s stride",
            "causal_future_leak_test": "passed before evaluation",
            "zero_shot_no_training_step_test": "passed before evaluation",
            "blind_holdout_telemetry_or_scores_read": False,
            "blind_holdout_flights_locked": len(holdout),
            "decision_layers": "imported unchanged from src/ml/decision/decision_layers.py",
            "score_fusion": "shared src/ml/evaluation/score_fusion.py helper",
        },
        "gate_b": _gate_b(category_metrics),
        "gate_c": _gate_c(metrics),
    }
    (output / "gates.json").write_text(
        json.dumps(_jsonable(gates), indent=2, allow_nan=True), encoding="utf-8",
    )
    shutil.copy2(PREFLIGHT_PATH, output / "preflight_check.json")
    shutil.copy2(FEASIBILITY_PATH, output / "feasibility_check.json")
    shutil.copy2(PRECOMPUTE_DIR / "manifest.json", output / "precompute_manifest.json")

    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    artifact_manifest = {
        "artifact_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ML-10 development forecast-residual evaluation",
        "source": "uav_sead",
        "model_id": precompute["model_id"],
        "model_revision": precompute["model_revision"],
        "zero_shot": True,
        "blind_holdout_read": False,
        "blind_holdout_flights": len(holdout),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "feature_table_sha256": _sha256(FEATURE_PATH),
        "forecast_score_table_sha256": _sha256(SCORE_PATH),
        "silver_table_sha256": _sha256(SILVER_PATH),
        "ml9_baseline_manifest_sha256": _sha256(ML9_DIR / "manifest.json"),
        "decision_layers": "imported unchanged from src/ml/decision/decision_layers.py",
        "gate_status": {name: value["status"] for name, value in gates.items()},
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): _sha256(path)
            for path in files
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2), encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_matrix")
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args()
    output = run(
        args.run_name, tuple(args.splits) if args.splits else None,
    )
    print(f"ML-10 artifact: {output}")
    print((output / "gates.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
