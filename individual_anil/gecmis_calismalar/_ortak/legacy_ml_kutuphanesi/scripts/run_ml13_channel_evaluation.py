"""Evaluate ML-13 two-channel alarm architecture on frozen SEAD splits."""

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
    _aggregate,
    _combined_categories,
    _evaluate,
    _jsonable,
    _score_modules,
    _streams,
)
from scripts.run_ml10_forecast_evaluation import _fit_policies, _load_ml9_models
from src.ml.data.scaling import apply_scaler_params
from src.ml.decision.channel_union import union_onsets
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
from src.ml.evaluation.score_fusion import last_causal_per_bucket

FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SILVER_PATH = ROOT / "data/silver/uav_sead_silver.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
LABEL_PATH = ROOT / "data/objectstore/bronze/uav_sead/labels.json"
ML9_DIR = ROOT / "artifacts/ml9/uav_sead/full_matrix"
ML12_DIR = ROOT / "artifacts/ml12/uav_sead/full_matrix"

DECISION_STRIDE_S = 1.0
SYSTEM_SCORE = "existing_fusion"
MECHANICAL_SCORE = "itki_komutu"
UNION_SCORE_SOURCE = "channel_union"
GATE_CATEGORY = "Actuator Outputs+Controls"
MEANINGFUL_RECALL_GAIN = 0.05
BASELINE_FA_MULTIPLIER = 1.10
BUDGET_ALLOCATIONS = {
    "agirlikli_sistem": {
        "advisory": {"sistem": 10.0, "mekanik": 2.0},
        "critical": {"sistem": 1.67, "mekanik": 0.33},
    },
    "dengeli": {
        "advisory": {"sistem": 8.0, "mekanik": 4.0},
        "critical": {"sistem": 1.33, "mekanik": 0.67},
    },
    "esit": {
        "advisory": {"sistem": 6.0, "mekanik": 6.0},
        "critical": {"sistem": 1.0, "mekanik": 1.0},
    },
}
BASELINE_SOURCES = {
    "ml9": (SYSTEM_SCORE,),
    "ml12": ("ml12_fusion_itki",),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("blind_holdout_read") is not False:
        raise ValueError(f"{directory} does not prove holdout isolation")
    for relative, expected in manifest.get("files", {}).items():
        path = directory / relative
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"Checksum mismatch in {directory}: {relative}")
    return manifest


def _load_mechanical_model(split_name: str) -> dict:
    split_dir = ML12_DIR / split_name
    calibration = json.loads(
        (split_dir / "calibration.json").read_text(encoding="utf-8")
    )
    values = calibration[MECHANICAL_SCORE]
    return {
        MECHANICAL_SCORE: {
            **values,
            "model": joblib.load(split_dir / "models" / f"{MECHANICAL_SCORE}.joblib"),
        }
    }


def _one_second_streams(scored: pd.DataFrame) -> pd.DataFrame:
    keep = ["source_id", "t_rel_s", "label", SYSTEM_SCORE, MECHANICAL_SCORE]
    return last_causal_per_bucket(
        scored, stride_seconds=DECISION_STRIDE_S, columns=keep,
    )


def _evaluate_union(
    frame: pd.DataFrame,
    ids: set[str],
    system_policy,
    mechanical_policy,
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
        system_onsets = system_policy.apply(group[SYSTEM_SCORE].to_numpy(dtype=float))
        mechanical_onsets = mechanical_policy.apply(
            group[MECHANICAL_SCORE].to_numpy(dtype=float)
        )
        onsets = union_onsets([system_onsets, mechanical_onsets])

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

    by_label = [
        {"flight_label": key, **_aggregate(value)}
        for key, value in sorted(label_details.items())
    ]
    by_category = [
        {"annotation_category": key, **_aggregate(value)}
        for key, value in sorted(category_details.items())
    ]
    return _aggregate(overall_details), by_label, by_category


def _baseline_rows(filename: str, selected_splits: set[str]) -> pd.DataFrame:
    frames = []
    for family, sources in BASELINE_SOURCES.items():
        directory = ML9_DIR if family == "ml9" else ML12_DIR
        frame = pd.read_csv(directory / filename)
        frames.append(
            frame[
                frame["split"].isin(selected_splits)
                & frame["score_source"].isin(sources)
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _best_baseline(
    baseline: pd.DataFrame, decision: str, budget: str,
) -> pd.DataFrame:
    selected = baseline[
        (baseline["decision"] == decision) & (baseline["budget"] == budget)
    ].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(["split", "seed", "score_source"])
    idx = selected.groupby(["split", "seed"])["event_onset_recall"].idxmax()
    best = selected.loc[
        idx,
        [
            "split",
            "seed",
            "score_source",
            "event_onset_recall",
            "false_alarms_per_hour",
        ],
    ].copy()
    return best.rename(columns={
        "score_source": "baseline_source",
        "event_onset_recall": "baseline_event_onset_recall",
        "false_alarms_per_hour": "baseline_false_alarms_per_hour",
    })


def _gate_b(metrics: pd.DataFrame, baseline_metrics: pd.DataFrame) -> dict:
    seed_count = int(metrics["seed"].nunique())
    rows = []
    for (allocation, decision, budget), group in metrics.groupby(
        ["allocation", "decision", "budget"]
    ):
        baseline = _best_baseline(baseline_metrics, decision, budget)
        merged = group.merge(baseline, on=["split", "seed"], how="inner")
        if merged.empty:
            continue
        gains = (
            merged["event_onset_recall"]
            - merged["baseline_event_onset_recall"]
        )
        candidate_recall = float(merged["event_onset_recall"].mean())
        baseline_recall = float(merged["baseline_event_onset_recall"].mean())
        candidate_fa = float(merged["false_alarms_per_hour"].mean())
        baseline_fa = float(merged["baseline_false_alarms_per_hour"].mean())
        if baseline_fa > 0:
            fa_ratio = candidate_fa / baseline_fa
        else:
            fa_ratio = 1.0 if candidate_fa <= 0 else np.inf
        row = {
            "allocation": allocation,
            "candidate": UNION_SCORE_SOURCE,
            "baseline_rule": "per-split best of existing_fusion and ml12_fusion_itki",
            "decision": decision,
            "budget": budget,
            "candidate_mean_recall": candidate_recall,
            "baseline_mean_recall": baseline_recall,
            "mean_recall_gain": float(gains.mean()),
            "positive_seed_count": int((gains > 0).sum()),
            "candidate_mean_false_alarms_per_hour": candidate_fa,
            "baseline_mean_false_alarms_per_hour": baseline_fa,
            "fa_ratio": float(fa_ratio),
            "fa_not_inflated": bool(fa_ratio <= BASELINE_FA_MULTIPLIER),
            "baseline_source_counts": {
                str(key): int(value)
                for key, value in merged["baseline_source"].value_counts().items()
            },
        }
        row["meaningful_gain"] = (
            row["mean_recall_gain"] >= MEANINGFUL_RECALL_GAIN
            and row["positive_seed_count"] >= 3
            and row["fa_not_inflated"]
        )
        rows.append(row)
    passed = seed_count >= 5 and any(row["meaningful_gain"] for row in rows)
    return {
        "status": ("passed" if passed else "failed") if seed_count >= 5 else "smoke_only",
        "evaluated_seed_count": seed_count,
        "rule": (
            "matching decision/budget mean recall gain >=0.05, positive in >=3/5 "
            "seeds, and union FA <=1.10x frozen best single-channel baseline"
        ),
        "comparisons": rows,
    }


def _gate_c1(metrics: pd.DataFrame) -> dict:
    seed_count = int(metrics["seed"].nunique())
    rows = []
    for (allocation, decision, budget), group in metrics.groupby(
        ["allocation", "decision", "budget"]
    ):
        recall = float(group["event_onset_recall"].mean())
        false_alarms = float(group["false_alarms_per_hour"].mean())
        rows.append({
            "allocation": allocation,
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


def _gate_c2(
    channel_metrics: pd.DataFrame, channel_category_metrics: pd.DataFrame,
) -> dict:
    seed_count = int(channel_metrics["seed"].nunique())
    key = [
        "split", "seed", "allocation", "decision", "budget", "channel",
        "score_source", "channel_budget_per_hour",
    ]
    overall = channel_metrics[
        channel_metrics["channel"] == "mekanik"
    ][key + ["false_alarms_per_hour"]].rename(columns={
        "false_alarms_per_hour": "channel_false_alarms_per_hour",
    })
    category = channel_category_metrics[
        (channel_category_metrics["channel"] == "mekanik")
        & (channel_category_metrics["annotation_category"] == GATE_CATEGORY)
    ][key + ["event_onset_recall"]].rename(columns={
        "event_onset_recall": "category_event_onset_recall",
    })
    merged = category.merge(overall, on=key, how="inner")
    rows = []
    for (allocation, decision, budget), group in merged.groupby(
        ["allocation", "decision", "budget"]
    ):
        recall = float(group["category_event_onset_recall"].mean())
        false_alarms = float(group["channel_false_alarms_per_hour"].mean())
        critical = recall >= MIN_RECALL["critical"] and false_alarms <= BUDGETS["critical"]
        advisory = recall >= MIN_RECALL["advisory"] and false_alarms <= BUDGETS["advisory"]
        rows.append({
            "allocation": allocation,
            "decision": decision,
            "budget": budget,
            "annotation_category": GATE_CATEGORY,
            "mean_category_event_onset_recall": recall,
            "mean_channel_false_alarms_per_hour": false_alarms,
            "critical_equivalent_passed": bool(critical),
            "advisory_equivalent_passed": bool(advisory),
            "passed": bool(critical or advisory),
        })
    passed = seed_count >= 5 and any(row["passed"] for row in rows)
    return {
        "status": ("passed" if passed else "failed") if seed_count >= 5 else "smoke_only",
        "evaluated_seed_count": seed_count,
        "rule": (
            "mechanical channel only: Actuator Outputs+Controls recall >=0.30 "
            "@ channel FA <=2, or recall >=0.50 @ channel FA <=12"
        ),
        "candidates": rows,
    }


def _append_rows(target: list[dict], common: dict, rows: list[dict]) -> None:
    target.extend({**common, **row} for row in rows)


def run(
    run_name: str,
    split_names: tuple[str, ...] | None = None,
) -> Path:
    ml9_manifest = _verify_manifest(ML9_DIR)
    ml12_manifest = _verify_manifest(ML12_DIR)
    if _sha256(FEATURE_PATH) != ml9_manifest["feature_table_sha256"]:
        raise ValueError("Frozen ML-9 feature table checksum mismatch")
    if _sha256(FEATURE_PATH) != ml12_manifest["feature_table_sha256"]:
        raise ValueError("Frozen ML-12 feature table checksum mismatch")
    if _sha256(SILVER_PATH) != ml12_manifest["silver_table_sha256"]:
        raise ValueError("Frozen ML-12 silver table checksum mismatch")
    if _sha256(SPLIT_PATH) != ml9_manifest["split_manifest_sha256"]:
        raise ValueError("Frozen ML-9 split manifest checksum mismatch")
    if _sha256(SPLIT_PATH) != ml12_manifest["split_manifest_sha256"]:
        raise ValueError("Frozen ML-12 split manifest checksum mismatch")

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
        raise AssertionError("Blind holdout entered the ML-13 development request")

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

    output = ROOT / "artifacts/ml13/uav_sead" / run_name
    output.mkdir(parents=True, exist_ok=True)
    selected_splits = set(folds)
    baseline_metrics = _baseline_rows("metrics.csv", selected_splits)
    baseline_labels = _baseline_rows("flight_label_metrics.csv", selected_splits)
    baseline_categories = _baseline_rows("category_metrics.csv", selected_splits)

    all_metrics: list[dict] = []
    all_labels: list[dict] = []
    all_categories: list[dict] = []
    channel_metrics: list[dict] = []
    channel_labels: list[dict] = []
    channel_categories: list[dict] = []

    for split_name, split in folds.items():
        seed = int(split["seed"])
        parts = {name: set(split[name]) for name in ("train", "val", "test")}
        if set().union(*parts.values()) & holdout:
            raise AssertionError(f"{split_name}: holdout entered train/val/test")

        scaler, ml9_fitted = _load_ml9_models(split_name)
        mechanical_fitted = _load_mechanical_model(split_name)
        scaled = apply_scaler_params(raw, scaler)
        scored = _score_modules(
            {**ml9_fitted, **mechanical_fitted}, scaled, parts["val"],
        )
        streams = _one_second_streams(scored)
        val_streams = {
            "sistem": _streams(streams, parts["val"], SYSTEM_SCORE),
            "mekanik": _streams(streams, parts["val"], MECHANICAL_SCORE),
        }

        split_dir = output / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        policies_json = {}
        for allocation, budget_map in BUDGET_ALLOCATIONS.items():
            for budget_name in BUDGETS:
                system_budget = budget_map[budget_name]["sistem"]
                mechanical_budget = budget_map[budget_name]["mekanik"]
                system_policies = _fit_policies(
                    val_streams["sistem"], system_budget, seed,
                )
                mechanical_policies = _fit_policies(
                    val_streams["mekanik"], mechanical_budget, seed,
                )
                for decision, system_policy in system_policies.items():
                    mechanical_policy = mechanical_policies[decision]
                    policies_json[
                        f"{allocation}:{budget_name}:{decision}:sistem"
                    ] = system_policy.to_dict()
                    policies_json[
                        f"{allocation}:{budget_name}:{decision}:mekanik"
                    ] = mechanical_policy.to_dict()

                    common = {
                        "split": split_name,
                        "seed": seed,
                        "score_source": UNION_SCORE_SOURCE,
                        "allocation": allocation,
                        "decision": decision,
                        "budget": budget_name,
                        "system_budget_per_hour": system_budget,
                        "mechanical_budget_per_hour": mechanical_budget,
                    }
                    overall, by_label, by_category = _evaluate_union(
                        streams,
                        parts["test"],
                        system_policy,
                        mechanical_policy,
                        t0=t0,
                        ranges=ranges,
                        ranges_by_category=categories,
                        flight_labels=config["flight_labels"],
                    )
                    all_metrics.append({**common, **overall})
                    _append_rows(all_labels, common, by_label)
                    _append_rows(all_categories, common, by_category)

                    for channel, score_col, policy, channel_budget in (
                        ("sistem", SYSTEM_SCORE, system_policy, system_budget),
                        ("mekanik", MECHANICAL_SCORE, mechanical_policy, mechanical_budget),
                    ):
                        channel_common = {
                            "split": split_name,
                            "seed": seed,
                            "score_source": score_col,
                            "channel": channel,
                            "allocation": allocation,
                            "decision": decision,
                            "budget": budget_name,
                            "channel_budget_per_hour": channel_budget,
                        }
                        ch_overall, ch_labels, ch_categories = _evaluate(
                            streams,
                            parts["test"],
                            score_col,
                            policy,
                            t0=t0,
                            ranges=ranges,
                            ranges_by_category=categories,
                            flight_labels=config["flight_labels"],
                        )
                        channel_metrics.append({**channel_common, **ch_overall})
                        _append_rows(channel_labels, channel_common, ch_labels)
                        _append_rows(channel_categories, channel_common, ch_categories)
        (split_dir / "policies.json").write_text(
            json.dumps(policies_json, indent=2), encoding="utf-8",
        )

    metrics = pd.DataFrame(all_metrics)
    label_metrics = pd.DataFrame(all_labels)
    category_metrics = pd.DataFrame(all_categories)
    channel_metrics_frame = pd.DataFrame(channel_metrics)
    channel_label_metrics = pd.DataFrame(channel_labels)
    channel_category_metrics = pd.DataFrame(channel_categories)
    metrics.to_csv(output / "metrics.csv", index=False)
    label_metrics.to_csv(output / "flight_label_metrics.csv", index=False)
    category_metrics.to_csv(output / "category_metrics.csv", index=False)
    channel_metrics_frame.to_csv(output / "channel_metrics.csv", index=False)
    channel_label_metrics.to_csv(output / "channel_flight_label_metrics.csv", index=False)
    channel_category_metrics.to_csv(output / "channel_category_metrics.csv", index=False)
    baseline_metrics.to_csv(output / "baseline_metrics.csv", index=False)
    baseline_labels.to_csv(output / "baseline_flight_label_metrics.csv", index=False)
    baseline_categories.to_csv(output / "baseline_category_metrics.csv", index=False)

    gates = {
        "gate_a": {
            "status": "passed",
            "blind_holdout_telemetry_or_scores_read": False,
            "blind_holdout_flights_locked": len(holdout),
            "frozen_ml9_manifest_verified": True,
            "frozen_ml12_manifest_verified": True,
            "scorer_models_loaded_from_artifacts": True,
            "scorer_model_training_performed": False,
            "system_channel": SYSTEM_SCORE,
            "mechanical_channel": MECHANICAL_SCORE,
            "budget_allocations": BUDGET_ALLOCATIONS,
            "decision_layers": "imported unchanged from src/ml/decision/decision_layers.py",
            "event_metrics": "imported unchanged from src/ml/evaluation/events.py",
            "union_semantics": "boolean OR over aligned per-flight one-second onset masks",
        },
        "gate_b": _gate_b(metrics, baseline_metrics),
        "gate_c1": _gate_c1(metrics),
        "gate_c2": _gate_c2(channel_metrics_frame, channel_category_metrics),
    }
    (output / "gates.json").write_text(
        json.dumps(_jsonable(gates), indent=2, allow_nan=True), encoding="utf-8",
    )

    files = [
        path for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    artifact_manifest = {
        "artifact_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ML-13 development two-channel alarm evaluation",
        "source": "uav_sead",
        "plan": "docs/ML13_KANAL_MIMARISI_PLAN.md",
        "blind_holdout_read": False,
        "blind_holdout_flights": len(holdout),
        "development_flights": int(raw["source_id"].nunique()),
        "development_source_ids_sha256": hashlib.sha256(
            "\n".join(sorted(development)).encode("utf-8")
        ).hexdigest(),
        "evaluated_splits": sorted(folds),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "feature_table_sha256": _sha256(FEATURE_PATH),
        "silver_table_sha256": _sha256(SILVER_PATH),
        "ml9_baseline_manifest_sha256": _sha256(ML9_DIR / "manifest.json"),
        "ml12_baseline_manifest_sha256": _sha256(ML12_DIR / "manifest.json"),
        "score_channels": {
            "sistem": {
                "score_source": SYSTEM_SCORE,
                "artifact_dir": str(ML9_DIR.relative_to(ROOT)).replace("\\", "/"),
            },
            "mekanik": {
                "score_source": MECHANICAL_SCORE,
                "artifact_dir": str(ML12_DIR.relative_to(ROOT)).replace("\\", "/"),
            },
        },
        "budget_allocations": BUDGET_ALLOCATIONS,
        "decision_layers": "imported unchanged from src/ml/decision/decision_layers.py",
        "event_metrics": "imported unchanged from src/ml/evaluation/events.py",
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
    output = run(args.run_name, tuple(args.splits) if args.splits else None)
    print(f"ML-13 artifact: {output}")
    print((output / "gates.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
