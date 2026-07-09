"""ML-14 enriched SEAD evaluation on refreshed development splits."""

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

from scripts.run_ml9_category_evaluation import (
    BUDGETS,
    MIN_RECALL,
    _evaluate,
    _fit_policies,
    _jsonable,
    _score_modules,
    _streams,
)
from src.ml.data.scaling import apply_scaler_params, fit_scaler_params
from src.ml.evaluation.events import load_uav_sead_ranges, load_uav_sead_ranges_by_category
from src.ml.evaluation.score_fusion import last_causal_per_bucket, max_score_fusion
from src.ml.features.uav_attack_features import feature_columns
from src.ml.models.modular_iforest import (
    PX4_ML12_THIN_MODULES,
    PX4_ML7_CANDIDATE_MODULES,
    fit_modular_iforest,
)

FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SILVER_PATH = ROOT / "data/silver/uav_sead_silver.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
LABEL_PATH = ROOT / "data/objectstore/bronze/uav_sead/labels.json"
OLD_ML9_DIR = ROOT / "artifacts/ml9/uav_sead/full_matrix"
OLD_ML12_DIR = ROOT / "artifacts/ml12/uav_sead/full_matrix"
REBUILD_REPORT = ROOT / "artifacts/ml14/uav_sead/rebuild_report.json"

SCORE_SOURCES = ("existing_fusion", "itki_komutu", "ml14_fusion")
DECISION_STRIDE_S = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_second_streams(scored: pd.DataFrame) -> pd.DataFrame:
    return last_causal_per_bucket(
        scored,
        stride_seconds=DECISION_STRIDE_S,
        columns=["source_id", "t_rel_s", "label", *SCORE_SOURCES],
    )


def _gate_d2(metrics: pd.DataFrame) -> dict:
    old_rows = []
    references = [
        ("existing_fusion", OLD_ML9_DIR / "metrics.csv"),
        ("itki_komutu", OLD_ML12_DIR / "metrics.csv"),
    ]
    for source, path in references:
        frame = pd.read_csv(path)
        selected = frame[
            (frame["score_source"] == source)
            & (frame["decision"] == "cusum")
            & (frame["budget"].isin(BUDGETS))
        ]
        for budget, group in selected.groupby("budget"):
            old_rows.append({
                "score_source": source,
                "budget": budget,
                "old_mean_fa_per_hour": float(group["false_alarms_per_hour"].mean()),
                "old_shift_ratio": float(group["false_alarms_per_hour"].mean() / BUDGETS[budget]),
            })

    new_rows = []
    selected = metrics[
        (metrics["score_source"].isin(["existing_fusion", "itki_komutu"]))
        & (metrics["decision"] == "cusum")
        & (metrics["budget"].isin(BUDGETS))
    ]
    for (source, budget), group in selected.groupby(["score_source", "budget"]):
        new_rows.append({
            "score_source": source,
            "budget": budget,
            "new_mean_fa_per_hour": float(group["false_alarms_per_hour"].mean()),
            "new_shift_ratio": float(group["false_alarms_per_hour"].mean() / BUDGETS[budget]),
        })

    table = pd.DataFrame(old_rows).merge(pd.DataFrame(new_rows), on=["score_source", "budget"])
    old_median = float(table["old_shift_ratio"].median())
    new_median = float(table["new_shift_ratio"].median())
    relative_drop = (old_median - new_median) / old_median if old_median else np.nan
    passed = int(metrics["seed"].nunique()) >= 5 and relative_drop >= 0.15
    return {
        "status": ("passed" if passed else "failed") if int(metrics["seed"].nunique()) >= 5 else "smoke_only",
        "rule": "median CUSUM test_FA_per_hour/budget drops by >=15% vs frozen ML9/ML12 references",
        "old_median_shift_ratio": old_median,
        "new_median_shift_ratio": new_median,
        "relative_drop": float(relative_drop),
        "cells": table.to_dict(orient="records"),
    }


def _gate_d3(metrics: pd.DataFrame) -> dict:
    seed_count = int(metrics["seed"].nunique())
    rows = []
    for (source, decision, budget), group in metrics.groupby(["score_source", "decision", "budget"]):
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
    passed = seed_count >= 5 and any(row["passed"] for row in rows)
    return {
        "status": ("passed" if passed else "failed") if seed_count >= 5 else "smoke_only",
        "rule": "any row critical recall >=0.30 @ FA<=2 or advisory recall >=0.50 @ FA<=12",
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
        raise AssertionError("Blind holdout entered the ML-14 development request")

    raw = pd.read_parquet(FEATURE_PATH, filters=[("source_id", "in", sorted(development))])
    silver_time = pd.read_parquet(
        SILVER_PATH, columns=["source_id", "timestamp"],
        filters=[("source_id", "in", sorted(development))],
    )
    for name, frame in (("feature", raw), ("silver", silver_time)):
        if set(frame["source_id"].unique()) & holdout:
            raise AssertionError(f"Blind holdout rows were read from {name} table")

    t0 = silver_time.groupby("source_id")["timestamp"].min().to_dict()
    ranges = load_uav_sead_ranges(LABEL_PATH)
    categories = load_uav_sead_ranges_by_category(LABEL_PATH)
    output = ROOT / "artifacts/ml14/uav_sead" / run_name
    output.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    all_labels: list[dict] = []
    all_categories: list[dict] = []
    module_definitions = {**PX4_ML7_CANDIDATE_MODULES, "itki_komutu": PX4_ML12_THIN_MODULES["itki_komutu"]}

    for split_name, split in folds.items():
        seed = int(split["seed"])
        parts = {name: set(split[name]) for name in ("train", "val", "test")}
        if set().union(*parts.values()) & holdout:
            raise AssertionError(f"{split_name}: holdout entered train/val/test")

        scaler = fit_scaler_params(raw[raw["source_id"].isin(parts["train"])], feature_columns(raw))
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
        for score_source in SCORE_SOURCES:
            val_streams = _streams(streams, parts["val"], score_source)
            for budget_name, budget in BUDGETS.items():
                for decision, policy in _fit_policies(val_streams, budget, seed).items():
                    policies_json[f"{score_source}:{budget_name}:{decision}"] = policy.to_dict()
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
                        "score_source": score_source,
                        "decision": decision,
                        "budget": budget_name,
                    }
                    all_metrics.append({**common, **overall})
                    all_labels.extend({**common, **row} for row in by_label)
                    all_categories.extend({**common, **row} for row in by_category)
        (split_dir / "policies.json").write_text(json.dumps(policies_json, indent=2), encoding="utf-8")

    metrics = pd.DataFrame(all_metrics)
    label_metrics = pd.DataFrame(all_labels)
    category_metrics = pd.DataFrame(all_categories)
    metrics.to_csv(output / "metrics.csv", index=False)
    label_metrics.to_csv(output / "flight_label_metrics.csv", index=False)
    category_metrics.to_csv(output / "category_metrics.csv", index=False)

    d1 = json.loads(REBUILD_REPORT.read_text(encoding="utf-8")) if REBUILD_REPORT.exists() else {}
    gates = {
        "gate_d1": d1.get("gate_d1", {"status": "unknown"}),
        "gate_d2": _gate_d2(metrics),
        "gate_d3": _gate_d3(metrics),
    }
    (output / "gates.json").write_text(json.dumps(_jsonable(gates), indent=2, allow_nan=True), encoding="utf-8")

    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    artifact_manifest = {
        "artifact_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ML-14 enriched SEAD evaluation",
        "source": "uav_sead",
        "plan": "docs/ML14_UYGULAMA_PLANI.md",
        "blind_holdout_read": False,
        "blind_holdout_flights": len(holdout),
        "development_flights": int(raw["source_id"].nunique()),
        "development_source_ids_sha256": hashlib.sha256("\n".join(sorted(development)).encode("utf-8")).hexdigest(),
        "evaluated_splits": sorted(folds),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "feature_table_sha256": _sha256(FEATURE_PATH),
        "silver_table_sha256": _sha256(SILVER_PATH),
        "old_ml9_manifest_sha256": _sha256(OLD_ML9_DIR / "manifest.json"),
        "old_ml12_manifest_sha256": _sha256(OLD_ML12_DIR / "manifest.json"),
        "gate_status": {name: value["status"] for name, value in gates.items()},
        "files": {str(path.relative_to(output)).replace("\\", "/"): _sha256(path) for path in files},
    }
    (output / "manifest.json").write_text(json.dumps(artifact_manifest, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="full_matrix")
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args()
    output = run(args.run_name, tuple(args.splits) if args.splits else None)
    print(f"ML-14 artifact: {output}")
    print((output / "gates.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
