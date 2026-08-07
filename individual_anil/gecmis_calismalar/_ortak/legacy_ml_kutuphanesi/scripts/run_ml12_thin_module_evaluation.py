"""Evaluate ML-12 thin-module candidates on frozen SEAD development splits.

On-kayitli plan: docs/ML12_INCE_MODUL_PLAN.md. Ince moduller (1-3 feature IF)
her split'in train-normalinde fit edilir, donmus ML-9 scaler'i ve degismeyen
karar katmanlariyla ML-9/ML-10 baseline satirlarina karsi olculur. Blind
holdout (131 ucus) hicbir asamada okunmaz.
"""

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
    _jsonable,
    _score_modules,
    _streams,
)
from scripts.run_ml10_forecast_evaluation import _fit_policies, _load_ml9_models
from src.ml.data.scaling import apply_scaler_params
from src.ml.evaluation.events import (
    load_uav_sead_ranges,
    load_uav_sead_ranges_by_category,
)
from src.ml.evaluation.score_fusion import last_causal_per_bucket, max_score_fusion
from src.ml.models.modular_iforest import (
    PX4_ML12_THIN_MODULES,
    fit_modular_iforest,
)

FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SILVER_PATH = ROOT / "data/silver/uav_sead_silver.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
LABEL_PATH = ROOT / "data/objectstore/bronze/uav_sead/labels.json"
ML9_DIR = ROOT / "artifacts/ml9/uav_sead/full_matrix"
ML10_DIR = ROOT / "artifacts/ml10/uav_sead/full_matrix"

THIN_SOURCES = tuple(PX4_ML12_THIN_MODULES)
FUSION_OF = {
    "ml12_fusion_itki": "itki_komutu",
    "ml12_fusion_ince": "itki_kontrol_ince",
}
NEW_SCORE_SOURCES = (*THIN_SOURCES, *FUSION_OF)
REUSED_ML9_SOURCES = ("existing_fusion", "ml9_fusion", "motor_simetrisi")
REUSED_ML10_SOURCES = ("chronos_motor", "ml10_fusion")
GATE_CATEGORY = "Actuator Outputs+Controls"
DECISION_STRIDE_S = 1.0
MEANINGFUL_RECALL_GAIN = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.get("files", {}).items():
        path = directory / relative
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"Checksum mismatch in {directory}: {relative}")
    return manifest


def _one_second_streams(scored: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keep = ["source_id", "t_rel_s", "label", *columns]
    return last_causal_per_bucket(
        scored, stride_seconds=DECISION_STRIDE_S, columns=keep,
    )


def _gate_b(category: pd.DataFrame) -> dict:
    """B1 (gate'i belirler): aday vs motor_simetrisi. B2 (bilgi): vs chronos_motor."""
    seed_count = int(category["seed"].nunique())
    comparisons = [
        *[("b1_gate", candidate, "motor_simetrisi") for candidate in THIN_SOURCES],
        *[("b2_informative", candidate, "chronos_motor") for candidate in THIN_SOURCES],
    ]
    rows = []
    for arm, candidate, baseline in comparisons:
        selected = category[
            (category["annotation_category"] == GATE_CATEGORY)
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
                "arm": arm,
                "annotation_category": GATE_CATEGORY,
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
    b1_passed = any(row["meaningful_gain"] for row in rows if row["arm"] == "b1_gate")
    b2_passed = any(
        row["meaningful_gain"] for row in rows if row["arm"] == "b2_informative"
    )
    return {
        "status": ("passed" if b1_passed else "failed") if seed_count >= 5 else "smoke_only",
        "evaluated_seed_count": seed_count,
        "rule": ("B1 decides the gate: matching policy/budget mean recall gain >=0.05 "
                 "and positive in >=3/5 seeds vs motor_simetrisi; B2 vs chronos_motor "
                 "is informative only (best-known claim requires B2)"),
        "b2_informative_passed": bool(b2_passed),
        "comparisons": rows,
    }


def _gate_c(metrics: pd.DataFrame) -> dict:
    seed_count = int(metrics["seed"].nunique())
    rows = []
    selected = metrics[metrics["score_source"].isin(FUSION_OF)]
    for (score_source, decision, budget), group in selected.groupby(
        ["score_source", "decision", "budget"]
    ):
        recall = float(group["event_onset_recall"].mean())
        false_alarms = float(group["false_alarms_per_hour"].mean())
        rows.append({
            "score_source": score_source,
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
    ml9_manifest = _verify_manifest(ML9_DIR)
    ml10_manifest = _verify_manifest(ML10_DIR)
    if _sha256(FEATURE_PATH) != ml9_manifest["feature_table_sha256"]:
        raise ValueError("Frozen ML-9 feature table checksum mismatch")
    if _sha256(SPLIT_PATH) != ml9_manifest["split_manifest_sha256"]:
        raise ValueError("Frozen ML-9 split manifest checksum mismatch")

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
        raise AssertionError("Blind holdout entered the ML-12 development request")

    raw = pd.read_parquet(
        FEATURE_PATH, filters=[("source_id", "in", sorted(development))],
    )
    silver_time = pd.read_parquet(
        SILVER_PATH,
        columns=["source_id", "timestamp"],
        filters=[("source_id", "in", sorted(development))],
    )
    for name, frame in (("feature", raw), ("silver", silver_time)):
        if set(frame["source_id"].unique()) & holdout:
            raise AssertionError(f"Blind holdout rows were read from {name} table")
    t0 = silver_time.groupby("source_id")["timestamp"].min().to_dict()
    ranges = load_uav_sead_ranges(LABEL_PATH)
    categories = load_uav_sead_ranges_by_category(LABEL_PATH)

    output = ROOT / "artifacts/ml12/uav_sead" / run_name
    output.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    all_labels: list[dict] = []
    all_categories: list[dict] = []

    # On-kayitli baseline satirlari: checksum'u dogrulanmis ML-9/ML-10 CSV'leri.
    selected_splits = set(folds)
    for directory, sources in ((ML9_DIR, REUSED_ML9_SOURCES),
                               (ML10_DIR, REUSED_ML10_SOURCES)):
        for filename, target in (
            ("metrics.csv", all_metrics),
            ("flight_label_metrics.csv", all_labels),
            ("category_metrics.csv", all_categories),
        ):
            frame = pd.read_csv(directory / filename)
            target.extend(frame[
                frame["split"].isin(selected_splits)
                & frame["score_source"].isin(sources)
            ].to_dict(orient="records"))

    for split_name, split in folds.items():
        seed = int(split["seed"])
        parts = {name: set(split[name]) for name in ("train", "val", "test")}
        if set().union(*parts.values()) & holdout:
            raise AssertionError(f"{split_name}: holdout entered train/val/test")

        scaler, ml9_fitted = _load_ml9_models(split_name)
        scaled = apply_scaler_params(raw, scaler)
        thin_fitted = fit_modular_iforest(
            scaled, split, PX4_ML12_THIN_MODULES, seed=seed, n_jobs=1,
        )
        if set(thin_fitted) != set(PX4_ML12_THIN_MODULES):
            raise RuntimeError("Thin module feature columns missing from table")
        # ML-9 modulleri existing_fusion'i yeniden uretir; ince moduller ayni
        # cagride val-normal CDF'ine kalibre edilir (ortak score_fusion yolu).
        scored = _score_modules({**ml9_fitted, **thin_fitted}, scaled, parts["val"])
        for fusion_name, thin_name in FUSION_OF.items():
            scored[fusion_name] = max_score_fusion(
                scored, ["existing_fusion", thin_name],
            )
        streams = _one_second_streams(
            scored, ["existing_fusion", *NEW_SCORE_SOURCES],
        )

        split_dir = output / split_name
        models_dir = split_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, item in thin_fitted.items():
            joblib.dump(item["model"], models_dir / f"{name}.joblib")
        (split_dir / "calibration.json").write_text(json.dumps({
            name: {key: value for key, value in item.items() if key != "model"}
            for name, item in thin_fitted.items()
        }, indent=2), encoding="utf-8")

        policies_json = {}
        for score_source in NEW_SCORE_SOURCES:
            val_streams = _streams(streams, parts["val"], score_source)
            for budget_name, budget in BUDGETS.items():
                policies = _fit_policies(val_streams, budget, seed)
                for decision, policy in policies.items():
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
            "blind_holdout_telemetry_or_scores_read": False,
            "blind_holdout_flights_locked": len(holdout),
            "frozen_ml9_scaler_reused": True,
            "thin_module_definitions": PX4_ML12_THIN_MODULES,
            "iforest_hyperparameters": "identical to ML-9 (n_estimators=300, max_samples=256)",
            "decision_layers": "imported unchanged from src/ml/decision/decision_layers.py",
            "score_fusion": "shared src/ml/evaluation/score_fusion.py helper",
        },
        "gate_b": _gate_b(category_metrics),
        "gate_c": _gate_c(metrics),
    }
    (output / "gates.json").write_text(
        json.dumps(_jsonable(gates), indent=2, allow_nan=True), encoding="utf-8",
    )

    files = [path for path in output.rglob("*")
             if path.is_file() and path.name != "manifest.json"]
    artifact_manifest = {
        "artifact_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ML-12 development thin-module evaluation",
        "source": "uav_sead",
        "plan": "docs/ML12_INCE_MODUL_PLAN.md",
        "blind_holdout_read": False,
        "blind_holdout_flights": len(holdout),
        "development_flights": int(raw["source_id"].nunique()),
        "development_source_ids_sha256": hashlib.sha256(
            "\n".join(sorted(development)).encode("utf-8")).hexdigest(),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "feature_table_sha256": _sha256(FEATURE_PATH),
        "silver_table_sha256": _sha256(SILVER_PATH),
        "ml9_baseline_manifest_sha256": _sha256(ML9_DIR / "manifest.json"),
        "ml10_baseline_manifest_sha256": _sha256(ML10_DIR / "manifest.json"),
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
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Optional smoke subset, e.g. --splits split_00")
    args = parser.parse_args()
    output = run(args.run_name, tuple(args.splits) if args.splits else None)
    print(f"ML-12 artifact: {output}")
    print((output / "gates.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
