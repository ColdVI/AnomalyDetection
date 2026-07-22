"""RFLY-0 exploratory RflyMAD and SEAD+RFLY pooled-normal evaluation.

This runner is intentionally not an official gate.  RflyMAD currently has only
51 normal flights, so the pre-registered 30/30 normal quota cannot be used.  The
outputs are early-signal artifacts for planning the next registered phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.simplefilter("ignore", PerformanceWarning)

from scripts.run_ml9_category_evaluation import (  # noqa: E402
    BUDGETS,
    _aggregate,
    _fit_policies,
    _jsonable,
    _score_modules,
    _streams,
)
from src.ml.data.scaling import (  # noqa: E402
    apply_scaler_params,
    fit_scaler_params,
    infer_source_column_presence,
    infer_source_schema_groups,
)
from src.ml.data.splits import flight_label_table, make_group_split  # noqa: E402
from src.ml.evaluation.events import (  # noqa: E402
    event_metrics,
    load_uav_sead_ranges,
    range_mask,
    uav_sead_absolute_us,
)
from src.ml.evaluation.score_fusion import last_causal_per_bucket, max_score_fusion  # noqa: E402
from src.ml.features.temporal import fit_cusum_baselines  # noqa: E402
from src.ml.features.uav_attack_features import (  # noqa: E402
    CUSUM_SOURCE_COLUMNS as PX4_CUSUM_COLUMNS,
)
from src.ml.features.uav_attack_features import build_px4_features, feature_columns  # noqa: E402
from src.ml.models.modular_iforest import (  # noqa: E402
    PX4_ML12_THIN_MODULES,
    PX4_ML9_CANDIDATE_MODULES,
    fit_modular_iforest,
)

RFLY_SILVER = ROOT / "data/silver/rflymad_silver.parquet"
SEAD_SILVER = ROOT / "data/silver/uav_sead_silver.parquet"
RFLY_GOLD = ROOT / "data/gold/ml_features/rflymad/rflymad_ml_features.parquet"
SEAD_GOLD = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SPLIT_MANIFEST = ROOT / "data/gold/ml_features/split_manifest.json"
SEAD_LABELS = ROOT / "data/objectstore/bronze/uav_sead/labels.json"

OUT_RFLY = ROOT / "artifacts/rfly0/rflymad"
OUT_POOLED = ROOT / "artifacts/rfly0/pooled_sead_rfly"
DECISION_STRIDE_S = 1.0
RFLY_EXPLORATORY_QUOTA = (15, 15)
RFLY_OFFICIAL_QUOTA = (12, 12)
SCORE_SOURCES = ("existing_fusion", "ml9_fusion", "itki_komutu", "rfly0_fusion")
INVALID_RFLY_INTERVAL_REASON = (
    "rfly_ctrl_lxl_no_active_fault: internal RFLY control message indicates no "
    "active fault trigger, contradicting the folder label"
)

MODULE_HYPOTHESES = {
    "itki_komutu": "actuator/motor hypothesis",
    "itki_kontrol_ince": "actuator/motor hypothesis",
    "kontrol_cevabi": "actuator/motor hypothesis",
    "motor_simetrisi": "actuator/motor hypothesis",
    "nav_butunlugu": "GPS/navigation hypothesis",
    "ekf_redleri": "GPS/navigation hypothesis",
    "irtifa_tutarliligi": "baro/altitude hypothesis",
    "dikey_tutarlilik": "baro/altitude hypothesis",
    "sinyal_kalitesi": "gyro/mag/acce sensor hypothesis",
    "veri_kalitesi": "gyro/mag/acce sensor hypothesis",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _load_rfly_fault_intervals(
    source_ids: set[str],
    flight_labels: dict[str, str],
    *,
    silver_path: Path = RFLY_SILVER,
) -> dict[str, tuple[float, float, str]]:
    """Load interval truth for selected RFLY development flights only."""
    wanted = sorted(sid for sid in source_ids if sid.startswith("Real-"))
    if not wanted:
        return {}
    cols = [
        "source_id",
        "fault_onset_s",
        "fault_end_s",
        "fault_interval_source",
    ]
    try:
        rows = pd.read_parquet(silver_path, columns=cols, filters=[("source_id", "in", wanted)])
    except Exception as exc:
        raise RuntimeError(
            "RFLY interval truth requires a refreshed rflymad_silver.parquet with "
            "fault_onset_s/fault_end_s columns"
        ) from exc

    seen = set(rows["source_id"].unique())
    missing: list[str] = []
    intervals: dict[str, tuple[float, float, str]] = {}
    for source_id in wanted:
        label = flight_labels.get(source_id)
        if label == "normal":
            continue
        if source_id not in seen:
            missing.append(source_id)
            continue
        first = rows[rows["source_id"] == source_id].iloc[0]
        onset = first["fault_onset_s"]
        end = first["fault_end_s"]
        if pd.isna(onset) or pd.isna(end) or float(end) < float(onset):
            missing.append(source_id)
            continue
        intervals[source_id] = (
            float(onset),
            float(end),
            str(first.get("fault_interval_source", "unknown")),
        )
    if missing:
        raise AssertionError(
            "Missing RFLY interval truth for anomalous development flights: "
            + ", ".join(sorted(missing)[:10])
            + (" ..." if len(missing) > 10 else "")
        )
    return intervals



def _invalid_rfly_interval_truth(
    source_ids: set[str],
    *,
    silver_path: Path = RFLY_SILVER,
) -> dict[str, str]:
    """Return anomalous RFLY flights that must be excluded from official truth."""
    wanted = sorted(sid for sid in source_ids if sid.startswith("Real-"))
    if not wanted:
        return {}
    cols = [
        "source_id",
        "label",
        "fault_onset_s",
        "fault_end_s",
        "fault_interval_source",
    ]
    rows = pd.read_parquet(silver_path, columns=cols, filters=[("source_id", "in", wanted)])
    invalid: dict[str, str] = {}
    for _, row in rows.drop_duplicates("source_id").iterrows():
        if row["label"] == "normal":
            continue
        source = str(row.get("fault_interval_source", "unknown"))
        onset = row["fault_onset_s"]
        end = row["fault_end_s"]
        if source == "rfly_ctrl_lxl_no_active_fault":
            invalid[str(row["source_id"])] = INVALID_RFLY_INTERVAL_REASON
        elif pd.isna(onset) or pd.isna(end) or float(end) < float(onset):
            invalid[str(row["source_id"])] = (
                f"{source}: missing or invalid interval truth; excluded before recall/FA evaluation"
            )
    return invalid


def _exclude_invalid_ids_from_folds(
    folds: dict[str, dict],
    invalid_ids: set[str],
) -> dict[str, dict]:
    if not invalid_ids:
        return folds
    cleaned: dict[str, dict] = {}
    for split_name, split in folds.items():
        item = dict(split)
        for key, value in split.items():
            if isinstance(value, list):
                item[key] = [source_id for source_id in value if source_id not in invalid_ids]
        cleaned[split_name] = item
    return cleaned

def _build_rfly_splits(
    rfly_features: pd.DataFrame,
    *,
    quota: tuple[int, int] = RFLY_EXPLORATORY_QUOTA,
    final_holdout_fraction: float = 0.0,
) -> dict[str, dict]:
    flights = flight_label_table(rfly_features)
    return {
        f"split_{seed:02d}": make_group_split(
            flights,
            seed=seed,
            n_val=quota[0],
            n_test_normal=quota[1],
            by_session=True,
            final_holdout_fraction=final_holdout_fraction,
        )
        for seed in range(5)
    }


def _build_features_with_split0_cusum(
    silver: pd.DataFrame,
    split0_train: set[str],
) -> tuple[pd.DataFrame, dict]:
    provisional = build_px4_features(silver)
    train = provisional[provisional["source_id"].isin(split0_train)]
    baselines = fit_cusum_baselines(train, PX4_CUSUM_COLUMNS)
    features = build_px4_features(silver, cusum_baselines=baselines)
    return features, baselines


def _distribution_report(rfly_silver: pd.DataFrame) -> dict:
    flights = rfly_silver[
        ["source_id", "label", "rflymad_subdataset", "rflymad_flight_mode"]
    ].drop_duplicates()
    label_mode = (
        flights.groupby(["label", "rflymad_flight_mode"])
        .size()
        .reset_index(name="count")
        .to_dict("records")
    )
    return {
        "source": "rflymad",
        "n_rows": int(len(rfly_silver)),
        "n_flights": int(flights["source_id"].nunique()),
        "by_subdataset": flights.groupby("rflymad_subdataset").size().sort_index().to_dict(),
        "by_mode": flights.groupby("rflymad_flight_mode").size().sort_index().to_dict(),
        "by_label": flights.groupby("label").size().sort_values(ascending=False).to_dict(),
        "label_by_mode": label_mode,
        "normal_count": int((flights["label"] == "normal").sum()),
        "official_rfly_normal_minimum": 61,
        "official_amended_quota": {
            "val_normal": RFLY_OFFICIAL_QUOTA[0],
            "test_normal": RFLY_OFFICIAL_QUOTA[1],
            "basis": "R1 amend: floor(15 * 51 / 61) with 27 train normals retained",
        },
        "exploratory_quota": {
            "val_normal": RFLY_EXPLORATORY_QUOTA[0],
            "test_normal": RFLY_EXPLORATORY_QUOTA[1],
        },
    }


def _one_second_streams(scored: pd.DataFrame) -> pd.DataFrame:
    return last_causal_per_bucket(
        scored,
        stride_seconds=DECISION_STRIDE_S,
        columns=["source_id", "t_rel_s", "label", *SCORE_SOURCES],
    )


def _dataset_group(source_id: str, flight_label: str) -> str:
    if source_id.startswith("Real-"):
        if flight_label == "normal":
            return "rfly_normal"
        if flight_label == "motor_fault":
            return "rfly_motor"
        if flight_label.startswith("sensor_"):
            return "rfly_sensor"
        return "rfly_other"
    if flight_label == "normal":
        return "sead_normal"
    return "sead_anomaly"


def _truth_mask(
    source_id: str,
    group: pd.DataFrame,
    flight_label: str,
    *,
    sead_t0: dict[str, float],
    sead_ranges: dict[str, list[tuple[float, float]]],
    rfly_intervals: dict[str, tuple[float, float, str]],
) -> np.ndarray:
    times = group["t_rel_s"].to_numpy(dtype=float)
    if source_id.startswith("Real-"):
        if flight_label == "normal":
            return np.zeros(len(group), dtype=bool)
        if source_id not in rfly_intervals:
            raise AssertionError(f"Missing interval truth for RFLY anomalous flight: {source_id}")
        onset, end, _source = rfly_intervals[source_id]
        return (times >= onset) & (times <= end)
    absolute = uav_sead_absolute_us(times, sead_t0[source_id])
    return range_mask(absolute, sead_ranges.get(source_id, []))


def _evaluate_generic(
    frame: pd.DataFrame,
    ids: set[str],
    score_col: str,
    policy,
    *,
    flight_labels: dict[str, str],
    sead_t0: dict[str, float],
    sead_ranges: dict[str, list[tuple[float, float]]],
    rfly_intervals: dict[str, tuple[float, float, str]],
) -> tuple[dict, list[dict], list[dict]]:
    overall_details: list[dict] = []
    label_details: dict[str, list[dict]] = {}
    group_details: dict[str, list[dict]] = {}
    subset = frame[frame["source_id"].isin(ids)]
    for source_id, group in subset.groupby("source_id"):
        group = group.sort_values("t_rel_s")
        times = group["t_rel_s"].to_numpy(dtype=float)
        label = flight_labels[source_id]
        truth = _truth_mask(
            source_id,
            group,
            label,
            sead_t0=sead_t0,
            sead_ranges=sead_ranges,
            rfly_intervals=rfly_intervals,
        )
        onsets = policy.apply(group[score_col].to_numpy(dtype=float))
        metrics = event_metrics(times, truth, onsets.astype(float), 0.5, max_gap_s=2.0)
        overall_details.append(metrics)
        label_details.setdefault(label, []).append(metrics)
        group_details.setdefault(_dataset_group(source_id, label), []).append(metrics)

    by_label = [{"flight_label": key, **_aggregate(value)}
                for key, value in sorted(label_details.items())]
    by_group = [{"eval_group": key, **_aggregate(value)}
                for key, value in sorted(group_details.items())]
    return _aggregate(overall_details), by_label, by_group


def _diagnosis_rows(
    streams: pd.DataFrame,
    test_ids: set[str],
    flight_labels: dict[str, str],
    module_cols: list[str],
) -> list[dict]:
    rows = []
    subset = streams[streams["source_id"].isin(test_ids)]
    for source_id, group in subset.groupby("source_id"):
        maxima = group[module_cols].max(numeric_only=True).sort_values(ascending=False)
        if maxima.empty:
            continue
        dominant = str(maxima.index[0])
        rows.append({
            "source_id": source_id,
            "flight_label": flight_labels[source_id],
            "eval_group": _dataset_group(source_id, flight_labels[source_id]),
            "dominant_module": dominant,
            "dominant_module_score": float(maxima.iloc[0]),
            "hypothesis": MODULE_HYPOTHESES.get(dominant, "unknown anomaly"),
        })
    return rows


def _summarize_metrics(metrics: pd.DataFrame) -> dict:
    rows = []
    for (source, decision, budget), group in metrics.groupby(["score_source", "decision", "budget"]):
        rows.append({
            "score_source": source,
            "decision": decision,
            "budget": budget,
            "mean_event_onset_recall": float(group["event_onset_recall"].mean()),
            "mean_false_alarms_per_hour": float(group["false_alarms_per_hour"].mean()),
            "seed_count": int(group["seed"].nunique()),
        })
    return {
        "official_gate": False,
        "status": "exploratory_not_official_gate",
        "best_rows": sorted(
            rows,
            key=lambda row: (
                np.nan_to_num(row["mean_event_onset_recall"], nan=-1.0),
                -np.nan_to_num(row["mean_false_alarms_per_hour"], nan=1e9),
            ),
            reverse=True,
        )[:10],
    }


def _gate_rb(group_metrics: pd.DataFrame) -> dict:
    seed_count = int(group_metrics["seed"].nunique())
    rows = []
    candidates = ("itki_komutu", "rfly0_fusion", "ml9_fusion")
    for eval_group in ("rfly_motor", "rfly_sensor"):
        subset = group_metrics[
            (group_metrics["eval_group"] == eval_group)
            & (group_metrics["score_source"].isin(("existing_fusion", *candidates)))
        ]
        for (candidate, decision, budget), group in subset[
            subset["score_source"].isin(candidates)
        ].groupby(["score_source", "decision", "budget"]):
            baseline = subset[
                (subset["score_source"] == "existing_fusion")
                & (subset["decision"] == decision)
                & (subset["budget"] == budget)
            ]
            pivot = group[["seed", "event_onset_recall"]].merge(
                baseline[["seed", "event_onset_recall"]],
                on="seed",
                suffixes=("_candidate", "_baseline"),
            )
            if pivot.empty:
                continue
            gains = pivot["event_onset_recall_candidate"] - pivot["event_onset_recall_baseline"]
            rows.append({
                "eval_group": eval_group,
                "candidate": candidate,
                "baseline": "existing_fusion",
                "decision": decision,
                "budget": budget,
                "candidate_mean_recall": float(pivot["event_onset_recall_candidate"].mean()),
                "baseline_mean_recall": float(pivot["event_onset_recall_baseline"].mean()),
                "mean_recall_gain": float(gains.mean()),
                "positive_seed_count": int((gains > 0).sum()),
                "passed": float(gains.mean()) >= 0.05 and int((gains > 0).sum()) >= 3,
            })
    passed = seed_count >= 5 and any(row["passed"] for row in rows)
    return {
        "status": ("passed" if passed else "failed") if seed_count >= 5 else "smoke_only",
        "evaluated_seed_count": seed_count,
        "rule": "motor/sensor candidate recall gain >=0.05 vs existing_fusion and positive in >=3/5 seeds",
        "comparisons": rows,
    }


def _gate_rc(metrics: pd.DataFrame) -> dict:
    seed_count = int(metrics["seed"].nunique())
    rows = []
    for (source, decision, budget), group in metrics.groupby(["score_source", "decision", "budget"]):
        recall = float(group["event_onset_recall"].mean())
        fa = float(group["false_alarms_per_hour"].mean())
        target_recall = 0.30 if budget == "critical" else 0.50
        target_fa = 2.0 if budget == "critical" else 12.0
        rows.append({
            "score_source": source,
            "decision": decision,
            "budget": budget,
            "mean_event_onset_recall": recall,
            "mean_false_alarms_per_hour": fa,
            "passed": recall >= target_recall and fa <= target_fa,
        })
    passed = seed_count >= 5 and any(row["passed"] for row in rows)
    return {
        "status": ("passed" if passed else "failed") if seed_count >= 5 else "smoke_only",
        "evaluated_seed_count": seed_count,
        "rule": "any row critical recall >=0.30 @ FA<=2 or advisory recall >=0.50 @ FA<=12",
        "candidates": rows,
    }


def _run_matrix(
    *,
    raw: pd.DataFrame,
    folds: dict[str, dict],
    selected_splits: tuple[str, ...],
    output: Path,
    flight_labels: dict[str, str],
    sead_t0: dict[str, float],
    sead_ranges: dict[str, list[tuple[float, float]]],
    rfly_intervals: dict[str, tuple[float, float, str]],
    invalid_rfly_interval_truth: dict[str, str] | None,
    source_note: str,
    input_hashes: dict[str, str],
    official_gate: bool = False,
    blind_holdout_count: int = 0,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    invalid_rfly_interval_truth = invalid_rfly_interval_truth or {}
    unknown = set(selected_splits) - set(folds)
    if unknown:
        raise ValueError(f"Unknown split names: {sorted(unknown)}")
    folds = {name: folds[name] for name in selected_splits}

    all_metrics: list[dict] = []
    all_labels: list[dict] = []
    all_groups: list[dict] = []
    all_diagnosis: list[dict] = []
    module_definitions = {**PX4_ML9_CANDIDATE_MODULES, **PX4_ML12_THIN_MODULES}
    feature_cols = feature_columns(raw)
    schema_groups = infer_source_schema_groups(raw)
    source_presence = infer_source_column_presence(raw, feature_cols, schema_groups)

    for split_name, split in folds.items():
        seed = int(split["seed"])
        parts = {name: set(split[name]) for name in ("train", "val", "test")}
        train_mask = raw["source_id"].isin(parts["train"])
        scaler = fit_scaler_params(
            raw[train_mask],
            feature_cols,
            source_groups=schema_groups.loc[train_mask],
            source_presence=source_presence,
        )
        scaled = apply_scaler_params(raw, scaler, source_groups=schema_groups)
        fitted = fit_modular_iforest(
            scaled,
            split,
            module_definitions,
            seed=seed,
            n_jobs=1,
        )
        scored = _score_modules(fitted, scaled, parts["val"])
        scored["rfly0_fusion"] = max_score_fusion(
            scored,
            [col for col in ("ml9_fusion", "itki_komutu") if col in scored.columns],
        )
        streams = _one_second_streams(scored)
        module_cols = [name for name in module_definitions if name in streams.columns]
        all_diagnosis.extend(_diagnosis_rows(
            streams, parts["test"], flight_labels, module_cols,
        ))

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
            if score_source not in streams.columns:
                continue
            val_streams = _streams(streams, parts["val"], score_source)
            for budget_name, budget in BUDGETS.items():
                for decision, policy in _fit_policies(val_streams, budget, seed).items():
                    policies_json[f"{score_source}:{budget_name}:{decision}"] = policy.to_dict()
                    overall, by_label, by_group = _evaluate_generic(
                        streams,
                        parts["test"],
                        score_source,
                        policy,
                        flight_labels=flight_labels,
                        sead_t0=sead_t0,
                        sead_ranges=sead_ranges,
                        rfly_intervals=rfly_intervals,
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
                    all_groups.extend({**common, **row} for row in by_group)
        (split_dir / "policies.json").write_text(json.dumps(policies_json, indent=2), encoding="utf-8")

    metrics = pd.DataFrame(all_metrics)
    label_metrics = pd.DataFrame(all_labels)
    group_metrics = pd.DataFrame(all_groups)
    diagnosis = pd.DataFrame(all_diagnosis)
    metrics.to_csv(output / "metrics.csv", index=False)
    label_metrics.to_csv(output / "flight_label_metrics.csv", index=False)
    group_metrics.to_csv(output / "eval_group_metrics.csv", index=False)
    diagnosis.to_csv(output / "diagnosis_hypotheses.csv", index=False)

    summary = _summarize_metrics(metrics)
    if official_gate:
        gates = {
            "gate_r_a": {
                "status": "passed",
                "blind_holdout_read": False,
                "blind_holdout_flights": blind_holdout_count,
                "split_overlap": "asserted by split construction",
                "invalid_interval_truth_excluded_flights": len(invalid_rfly_interval_truth),
                "invalid_interval_truth_reason": INVALID_RFLY_INTERVAL_REASON,
            },
            "gate_r_b": _gate_rb(group_metrics),
            "gate_r_c": _gate_rc(metrics),
        }
        (output / "gates.json").write_text(
            json.dumps(_jsonable(gates), indent=2, allow_nan=True), encoding="utf-8",
        )
        summary = {
            **summary,
            "official_gate": True,
            "status": "official_gate_evaluated",
            "gate_status": {name: value["status"] for name, value in gates.items()},
        }
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=True), encoding="utf-8")

    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "artifact_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "RFLY-0 exploratory evaluation",
        "source_note": source_note,
        "official_gate": official_gate,
        "status": "official_gate_evaluated" if official_gate else "exploratory_not_official_gate",
        "rfly_anomaly_truth": "rfly_ctrl_lxl interval truth for Real-* anomalies; Real normal flights are all-false",
        "rfly_fault_interval_flights": len(rfly_intervals),
        "rfly_fault_interval_sources": sorted(set(source for _onset, _end, source in rfly_intervals.values())),
        "invalid_rfly_interval_truth_exclusions": invalid_rfly_interval_truth,
        "invalid_rfly_interval_truth_exclusion_count": len(invalid_rfly_interval_truth),
        "invalid_rfly_interval_truth_reason": INVALID_RFLY_INTERVAL_REASON,
        "blind_holdout_read": False,
        "blind_holdout_flights": blind_holdout_count,
        "decision_layers": "imported unchanged from src/ml/decision/decision_layers.py",
        "score_fusion": "imported unchanged from src/ml/evaluation/score_fusion.py",
        "evaluated_splits": sorted(folds),
        "development_source_ids_sha256": _ids_sha256(
            set().union(*(set(split[p]) for split in folds.values() for p in ("train", "val", "test")))
        ),
        "input_hashes": input_hashes,
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): _sha256(path)
            for path in files
        },
    }
    (output / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2, allow_nan=True), encoding="utf-8")
    return output


def _development_and_holdout(folds: dict[str, dict]) -> tuple[set[str], set[str]]:
    development = set().union(*(
        set(split[part]) for split in folds.values() for part in ("train", "val", "test")
    ))
    holdout = set().union(*(set(split.get("final_holdout", [])) for split in folds.values()))
    if development & holdout:
        raise AssertionError("Blind holdout entered RFLY development")
    return development, holdout


def run_rfly_only(run_name: str, splits: tuple[str, ...], *, official: bool = False) -> Path:
    if official:
        manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
        config = manifest["sources"]["rflymad"]
        folds = {name: config["splits"][name] for name in splits}
        development_all, holdout = _development_and_holdout(folds)
        invalid = _invalid_rfly_interval_truth(development_all)
        folds = _exclude_invalid_ids_from_folds(folds, set(invalid))
        development = set().union(*(
            set(split[part]) for split in folds.values() for part in ("train", "val", "test")
        ))
        rfly_silver = pd.read_parquet(RFLY_SILVER, filters=[("source_id", "in", sorted(development))])
        raw = pd.read_parquet(RFLY_GOLD, filters=[("source_id", "in", sorted(development))])
        if set(rfly_silver["source_id"].unique()) & holdout:
            raise AssertionError("RFLY blind holdout rows were read from Silver")
        if set(raw["source_id"].unique()) & holdout:
            raise AssertionError("RFLY blind holdout rows were read from Gold")
        if set(raw["source_id"].unique()) & set(invalid):
            raise AssertionError("Invalid RFLY interval-truth rows entered official Gold evaluation")
        cusum = {}
        distribution_scope = "official_selected_valid_development_only"
    else:
        rfly_silver = pd.read_parquet(RFLY_SILVER)
        provisional = build_px4_features(rfly_silver)
        folds = _build_rfly_splits(provisional)
        all_ids = set(rfly_silver["source_id"].unique())
        invalid = _invalid_rfly_interval_truth(all_ids)
        folds = _exclude_invalid_ids_from_folds(folds, set(invalid))
        raw, cusum = _build_features_with_split0_cusum(rfly_silver, set(folds["split_00"]["train"]))
        valid_ids = set().union(*(
            set(split[part]) for split in folds.values() for part in ("train", "val", "test")
        ))
        raw = raw[raw["source_id"].isin(valid_ids)].copy()
        holdout = set()
        distribution_scope = "exploratory_all_available_real_rfly_excluding_invalid_interval_truth"

    distribution = _distribution_report(rfly_silver)
    distribution["scope"] = distribution_scope
    distribution["invalid_interval_truth_excluded_flights"] = len(invalid)
    distribution["invalid_interval_truth_reason"] = INVALID_RFLY_INTERVAL_REASON
    OUT_RFLY.mkdir(parents=True, exist_ok=True)
    (OUT_RFLY / "distribution_report.json").write_text(
        json.dumps(_jsonable(distribution), indent=2, ensure_ascii=False), encoding="utf-8",
    )

    flight_labels = flight_label_table(raw).set_index("source_id")["flight_label"].to_dict()
    rfly_intervals = _load_rfly_fault_intervals(set(raw["source_id"].unique()), flight_labels)
    output = OUT_RFLY / run_name
    (output / "cusum_feature_baseline.json").parent.mkdir(parents=True, exist_ok=True)
    (output / "cusum_feature_baseline.json").write_text(json.dumps(cusum, indent=2), encoding="utf-8")
    return _run_matrix(
        raw=raw,
        folds=folds,
        selected_splits=splits,
        output=output,
        flight_labels=flight_labels,
        sead_t0={},
        sead_ranges={},
        rfly_intervals=rfly_intervals,
        invalid_rfly_interval_truth=invalid,
        source_note=(
            "rflymad-only official RFLY-0 amended-quota interval-truth run; invalid no-active-fault cases excluded"
            if official else "rflymad-only exploratory low-normal-quota interval-truth run; invalid no-active-fault cases excluded"
        ),
        input_hashes={
            "rfly_silver_sha256": _sha256(RFLY_SILVER),
            **({"rfly_gold_sha256": _sha256(RFLY_GOLD),
                "split_manifest_sha256": _sha256(SPLIT_MANIFEST)} if official else {}),
        },
        official_gate=official,
        blind_holdout_count=len(holdout),
    )


def _selected_sead_development(config: dict, split_names: tuple[str, ...]) -> tuple[dict[str, dict], set[str], set[str]]:
    folds = {name: config["splits"][name] for name in split_names}
    holdout = set(config["splits"]["split_00"]["final_holdout"])
    development = set().union(*(
        set(split[part]) for split in folds.values() for part in ("train", "val", "test")
    ))
    if development & holdout:
        raise AssertionError("SEAD blind holdout entered pooled exploratory development")
    return folds, development, holdout


def run_pooled(run_name: str, splits: tuple[str, ...], *, official: bool = False) -> Path:
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    sead_config = split_manifest["sources"]["uav_sead"]
    sead_folds, sead_development, sead_holdout = _selected_sead_development(sead_config, splits)
    if official:
        rfly_config = split_manifest["sources"]["rflymad"]
        rfly_folds = {name: rfly_config["splits"][name] for name in splits}
        rfly_development_all, rfly_holdout = _development_and_holdout(rfly_folds)
        invalid = _invalid_rfly_interval_truth(rfly_development_all)
        rfly_folds = _exclude_invalid_ids_from_folds(rfly_folds, set(invalid))
        rfly_development = set().union(*(
            set(split[part]) for split in rfly_folds.values() for part in ("train", "val", "test")
        ))
        sead_features = pd.read_parquet(
            SEAD_GOLD,
            filters=[("source_id", "in", sorted(sead_development))],
        )
        rfly_features = pd.read_parquet(
            RFLY_GOLD,
            filters=[("source_id", "in", sorted(rfly_development))],
        )
        if set(sead_features["source_id"].unique()) & sead_holdout:
            raise AssertionError("SEAD blind holdout rows were read in official pooled run")
        if set(rfly_features["source_id"].unique()) & rfly_holdout:
            raise AssertionError("RFLY blind holdout rows were read in official pooled run")
        if set(rfly_features["source_id"].unique()) & set(invalid):
            raise AssertionError("Invalid RFLY interval-truth rows entered official pooled Gold evaluation")
        raw = pd.concat([sead_features, rfly_features], ignore_index=True, sort=False)
        cusum = {}
        combined_holdout = sead_holdout | rfly_holdout
    else:
        rfly_silver = pd.read_parquet(RFLY_SILVER)
        rfly_provisional = build_px4_features(rfly_silver)
        rfly_folds = _build_rfly_splits(rfly_provisional)
        invalid = _invalid_rfly_interval_truth(set(rfly_silver["source_id"].unique()))
        rfly_folds = _exclude_invalid_ids_from_folds(rfly_folds, set(invalid))
        sead_silver = pd.read_parquet(
            SEAD_SILVER,
            filters=[("source_id", "in", sorted(sead_development))],
        )
        if set(sead_silver["source_id"].unique()) & sead_holdout:
            raise AssertionError("SEAD blind holdout rows were read in pooled exploratory run")
        valid_rfly_ids = set().union(*(
            set(split[part]) for split in rfly_folds.values() for part in ("train", "val", "test")
        ))
        rfly_silver = rfly_silver[rfly_silver["source_id"].isin(valid_rfly_ids)].copy()
        combined_silver = pd.concat([sead_silver, rfly_silver], ignore_index=True, sort=False)
        split0_train = set(sead_folds["split_00"]["train"]) | set(rfly_folds["split_00"]["train"])
        raw, cusum = _build_features_with_split0_cusum(combined_silver, split0_train)
        combined_holdout = sead_holdout

    folds = {}
    for name in splits:
        sead = sead_folds[name]
        rfly = rfly_folds[name]
        folds[name] = {
            "seed": int(sead["seed"]),
            "train": sorted(set(sead["train"]) | set(rfly["train"])),
            "val": sorted(set(sead["val"]) | set(rfly["val"])),
            "test": sorted(set(sead["test"]) | set(rfly["test"])),
        }

    rfly_labels = flight_label_table(raw[raw["source_id"].str.startswith("Real-")])
    flight_labels = {
        **sead_config["flight_labels"],
        **rfly_labels.set_index("source_id")["flight_label"].to_dict(),
    }
    rfly_intervals = _load_rfly_fault_intervals(set(raw["source_id"].unique()), flight_labels)
    if official:
        sead_time = pd.read_parquet(
            SEAD_SILVER,
            columns=["source_id", "timestamp"],
            filters=[("source_id", "in", sorted(sead_development))],
        )
        if set(sead_time["source_id"].unique()) & sead_holdout:
            raise AssertionError("SEAD blind holdout rows were read for official pooled timing")
        sead_t0 = sead_time.groupby("source_id")["timestamp"].min().to_dict()
    else:
        sead_t0 = sead_silver.groupby("source_id")["timestamp"].min().to_dict()
    sead_ranges = load_uav_sead_ranges(SEAD_LABELS)
    output = OUT_POOLED / run_name
    (output / "cusum_feature_baseline.json").parent.mkdir(parents=True, exist_ok=True)
    (output / "cusum_feature_baseline.json").write_text(json.dumps(cusum, indent=2), encoding="utf-8")
    return _run_matrix(
        raw=raw,
        folds=folds,
        selected_splits=splits,
        output=output,
        flight_labels=flight_labels,
        sead_t0=sead_t0,
        sead_ranges=sead_ranges,
        rfly_intervals=rfly_intervals,
        invalid_rfly_interval_truth=invalid,
        source_note=(
            "SEAD development + valid-interval RFLY official amended-quota pooled-normal run; ALFA excluded"
            if official else "SEAD development + valid-interval RFLY exploratory pooled-normal run; ALFA excluded"
        ),
        input_hashes={
            "rfly_silver_sha256": _sha256(RFLY_SILVER),
            "sead_silver_sha256": _sha256(SEAD_SILVER),
            "split_manifest_sha256": _sha256(SPLIT_MANIFEST),
            **({"rfly_gold_sha256": _sha256(RFLY_GOLD),
                "sead_gold_sha256": _sha256(SEAD_GOLD)} if official else {}),
        },
        official_gate=official,
        blind_holdout_count=len(combined_holdout),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rfly-only", "pooled", "both"), default="both")
    parser.add_argument("--run-name", default="exploratory_full")
    parser.add_argument("--splits", nargs="+", default=[f"split_{i:02d}" for i in range(5)])
    parser.add_argument("--official", action="store_true",
                        help="Run the amended-quota official RFLY-0 gate path from Gold + split_manifest")
    args = parser.parse_args()
    splits = tuple(args.splits)
    outputs = []
    if args.mode in {"rfly-only", "both"}:
        outputs.append(run_rfly_only(args.run_name, splits, official=args.official))
    if args.mode in {"pooled", "both"}:
        outputs.append(run_pooled(args.run_name, splits, official=args.official))
    for output in outputs:
        print(f"Exploratory artifact: {output}")
        print((output / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
