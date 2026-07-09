"""ML-15 drift-calibrated SEAD evaluation on ML-14 refreshed splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ml9_category_evaluation import (  # noqa: E402
    BUDGETS,
    MIN_RECALL,
    _evaluate,
    _fit_policies,
    _jsonable,
    _score_modules,
    _streams,
)
from src.ml.data.scaling import apply_scaler_params, fit_scaler_params  # noqa: E402
from src.ml.data.splits import session_of  # noqa: E402
from src.ml.decision import decision_layers  # noqa: E402
from src.ml.decision.drift_calibration import fit_drift_corrected_policy  # noqa: E402
from src.ml.evaluation.events import load_uav_sead_ranges, load_uav_sead_ranges_by_category  # noqa: E402
from src.ml.evaluation.score_fusion import last_causal_per_bucket, max_score_fusion  # noqa: E402
from src.ml.features.uav_attack_features import feature_columns  # noqa: E402
from src.ml.models.modular_iforest import (  # noqa: E402
    PX4_ML12_THIN_MODULES,
    PX4_ML7_CANDIDATE_MODULES,
    fit_modular_iforest,
)

FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SILVER_PATH = ROOT / "data/silver/uav_sead_silver.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
LABEL_PATH = ROOT / "data/objectstore/bronze/uav_sead/labels.json"
ML14_FULL = ROOT / "artifacts/ml14/uav_sead/full_matrix"

SCORE_SOURCES = ("existing_fusion", "itki_komutu", "ml14_fusion")
DECISION_STRIDE_S = 1.0

DECISION_FITS = {
    "threshold": decision_layers.fit_threshold_policy,
    "k_of_n": decision_layers.fit_k_of_n_policy,
    "cusum": decision_layers.fit_cusum_policy,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _one_second_streams(scored: pd.DataFrame) -> pd.DataFrame:
    return last_causal_per_bucket(
        scored,
        stride_seconds=DECISION_STRIDE_S,
        columns=["source_id", "t_rel_s", "label", *SCORE_SOURCES],
    )


def _streams_by_session(frame: pd.DataFrame, ids: set[str], score_col: str) -> dict[str, list[np.ndarray]]:
    sessions: dict[str, list[np.ndarray]] = {}
    subset = frame[frame["source_id"].isin(ids)]
    for source_id, group in subset.groupby("source_id"):
        sessions.setdefault(session_of(str(source_id)), []).append(
            group.sort_values("t_rel_s")[score_col].to_numpy(dtype=float)
        )
    return sessions


def _fallback_shift_ratio() -> float:
    gates_path = ML14_FULL / "gates.json"
    if not gates_path.exists():
        return 1.0
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    value = gates.get("gate_d2", {}).get("new_median_shift_ratio", 1.0)
    return float(value) if np.isfinite(value) else 1.0


def _gate_b(metrics: pd.DataFrame) -> dict:
    corrected = metrics[
        (metrics["calibration"] == "drift_corrected")
        & (metrics["decision"] == "cusum")
        & (metrics["score_source"].isin(["existing_fusion", "itki_komutu"]))
    ]
    seed_count = int(corrected["seed"].nunique())
    rows = []
    for (source, budget), group in corrected.groupby(["score_source", "budget"]):
        target = BUDGETS[budget]
        fa = group["false_alarms_per_hour"]
        rows.append({
            "score_source": source,
            "budget": budget,
            "median_fa_per_hour": float(fa.median()),
            "seeds_within_1_25_budget": int((fa <= 1.25 * target).sum()),
            "seed_count": int(group["seed"].nunique()),
            "passed": float(fa.median()) <= target and int((fa <= 1.25 * target).sum()) >= 4,
        })
    passed_cells = sum(1 for row in rows if row["passed"])
    return {
        "status": ("passed" if passed_cells >= 3 else "failed") if seed_count >= 5 else "smoke_only",
        "rule": ">=3/4 {existing_fusion,itki_komutu} x {critical,advisory} CUSUM cells meet median FA<=budget and >=4/5 seeds <=1.25x budget",
        "evaluated_seed_count": seed_count,
        "passed_cells": passed_cells,
        "cells": rows,
    }


def _gate_c(metrics: pd.DataFrame) -> dict:
    corrected = metrics[metrics["calibration"] == "drift_corrected"]
    seed_count = int(corrected["seed"].nunique())
    rows = []
    for (source, decision, budget), group in corrected.groupby(["score_source", "decision", "budget"]):
        recall = float(group["event_onset_recall"].mean())
        fa = float(group["false_alarms_per_hour"].mean())
        rows.append({
            "score_source": source,
            "decision": decision,
            "budget": budget,
            "mean_event_onset_recall": recall,
            "mean_false_alarms_per_hour": fa,
            "passed": recall >= MIN_RECALL[budget] and fa <= BUDGETS[budget],
        })
    return {
        "status": ("passed" if any(row["passed"] for row in rows) else "failed") if seed_count >= 5 else "smoke_only",
        "rule": "any drift-corrected row critical recall >=0.30 @ FA<=2 or advisory recall >=0.50 @ FA<=12",
        "evaluated_seed_count": seed_count,
        "candidates": rows,
    }


def run(run_name: str, split_names: tuple[str, ...] | None = None) -> Path:
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
        raise AssertionError("Blind holdout entered the ML-15 development request")

    raw = pd.read_parquet(FEATURE_PATH, filters=[("source_id", "in", sorted(development))])
    silver_time = pd.read_parquet(
        SILVER_PATH,
        columns=["source_id", "timestamp"],
        filters=[("source_id", "in", sorted(development))],
    )
    if set(raw["source_id"].unique()) & holdout:
        raise AssertionError("Blind holdout rows were read from feature table")
    if set(silver_time["source_id"].unique()) & holdout:
        raise AssertionError("Blind holdout rows were read from Silver")

    t0 = silver_time.groupby("source_id")["timestamp"].min().to_dict()
    ranges = load_uav_sead_ranges(LABEL_PATH)
    categories = load_uav_sead_ranges_by_category(LABEL_PATH)
    output = ROOT / "artifacts/ml15/uav_sead" / run_name
    output.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    all_labels: list[dict] = []
    all_categories: list[dict] = []
    module_definitions = {
        **PX4_ML7_CANDIDATE_MODULES,
        "itki_komutu": PX4_ML12_THIN_MODULES["itki_komutu"],
    }
    fallback = _fallback_shift_ratio()

    for split_name, split in folds.items():
        seed = int(split["seed"])
        parts = {name: set(split[name]) for name in ("train", "val", "test")}
        if set().union(*parts.values()) & holdout:
            raise AssertionError(f"{split_name}: holdout entered train/val/test")

        scaler = fit_scaler_params(
            raw[raw["source_id"].isin(parts["train"])], feature_columns(raw),
        )
        scaled = apply_scaler_params(raw, scaler)
        fitted = fit_modular_iforest(scaled, split, module_definitions, seed=seed, n_jobs=1)
        scored = _score_modules(fitted, scaled, parts["val"])
        scored["ml14_fusion"] = max_score_fusion(scored, ["existing_fusion", "itki_komutu"])
        streams = _one_second_streams(scored)

        split_dir = output / split_name
        models_dir = split_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, item in fitted.items():
            joblib.dump(item["model"], models_dir / f"{name}.joblib")
        (split_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
        (split_dir / "calibration.json").write_text(json.dumps({
            name: {key: value for key, value in item.items() if key != "model"}
            for name, item in fitted.items()
        }, indent=2), encoding="utf-8")

        policies_json = {}
        drift_reports = {}
        for score_source in SCORE_SOURCES:
            val_streams = _streams(streams, parts["val"], score_source)
            val_by_session = _streams_by_session(streams, parts["val"], score_source)
            for budget_name, budget in BUDGETS.items():
                uncorrected = _fit_policies(val_streams, budget, seed)
                for decision, policy in uncorrected.items():
                    policies_json[f"none:{score_source}:{budget_name}:{decision}"] = policy.to_dict()
                    overall, by_label, by_category = _evaluate(
                        streams,
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
                        "calibration": "none",
                        "score_source": score_source,
                        "decision": decision,
                        "budget": budget_name,
                    }
                    all_metrics.append({**common, **overall})
                    all_labels.extend({**common, **row} for row in by_label)
                    all_categories.extend({**common, **row} for row in by_category)

                for decision, fit_fn in DECISION_FITS.items():
                    policy, report = fit_drift_corrected_policy(
                        val_by_session,
                        budget,
                        fit_fn,
                        seed,
                        stride_seconds=DECISION_STRIDE_S,
                        fallback_drift_multiplier=fallback,
                    )
                    key = f"drift_corrected:{score_source}:{budget_name}:{decision}"
                    policies_json[key] = policy.to_dict()
                    drift_reports[key] = report
                    overall, by_label, by_category = _evaluate(
                        streams,
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
                        "calibration": "drift_corrected",
                        "score_source": score_source,
                        "decision": decision,
                        "budget": budget_name,
                    }
                    all_metrics.append({**common, **overall})
                    all_labels.extend({**common, **row} for row in by_label)
                    all_categories.extend({**common, **row} for row in by_category)
        (split_dir / "policies.json").write_text(json.dumps(policies_json, indent=2), encoding="utf-8")
        (split_dir / "drift_reports.json").write_text(
            json.dumps(_jsonable(drift_reports), indent=2, allow_nan=True), encoding="utf-8",
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
            "blind_holdout_read": False,
            "blind_holdout_flights": len(holdout),
            "decision_layers": "imported unchanged from src/ml/decision/decision_layers.py",
            "policy_objects": "existing decision layer classes only",
        },
        "gate_b": _gate_b(metrics),
        "gate_c": _gate_c(metrics),
    }
    (output / "gates.json").write_text(json.dumps(_jsonable(gates), indent=2, allow_nan=True), encoding="utf-8")

    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    artifact_manifest = {
        "artifact_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ML-15 drift-calibrated SEAD evaluation",
        "source": "uav_sead",
        "plan": "docs/ML15_KALIBRASYON_PLAN.md",
        "blind_holdout_read": False,
        "blind_holdout_flights": len(holdout),
        "development_flights": int(raw["source_id"].nunique()),
        "development_source_ids_sha256": _ids_sha256(development),
        "evaluated_splits": sorted(folds),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "feature_table_sha256": _sha256(FEATURE_PATH),
        "silver_table_sha256": _sha256(SILVER_PATH),
        "ml14_full_manifest_sha256": _sha256(ML14_FULL / "manifest.json"),
        "fallback_drift_multiplier": fallback,
        "gate_status": {name: value["status"] for name, value in gates.items()},
        "files": {str(path.relative_to(output)).replace("\\", "/"): _sha256(path) for path in files},
    }
    (output / "manifest.json").write_text(json.dumps(_jsonable(artifact_manifest), indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_matrix")
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args()
    output = run(args.run_name, tuple(args.splits) if args.splits else None)
    print(f"ML-15 artifact: {output}")
    print((output / "gates.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
