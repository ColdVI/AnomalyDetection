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
from src.ml.data.scaling import apply_scaler_params, fit_scaler_params  # noqa: E402
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
SEAD_SPLIT = ROOT / "data/gold/ml_features/split_manifest.json"
SEAD_LABELS = ROOT / "data/objectstore/bronze/uav_sead/labels.json"

OUT_RFLY = ROOT / "artifacts/rfly0/rflymad"
OUT_POOLED = ROOT / "artifacts/rfly0/pooled_sead_rfly"
DECISION_STRIDE_S = 1.0
RFLY_EXPLORATORY_QUOTA = (15, 15)
SCORE_SOURCES = ("existing_fusion", "ml9_fusion", "itki_komutu", "exploratory_fusion")

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


def _build_rfly_splits(rfly_features: pd.DataFrame) -> dict[str, dict]:
    flights = flight_label_table(rfly_features)
    return {
        f"split_{seed:02d}": make_group_split(
            flights,
            seed=seed,
            n_val=RFLY_EXPLORATORY_QUOTA[0],
            n_test_normal=RFLY_EXPLORATORY_QUOTA[1],
            by_session=True,
            final_holdout_fraction=0.0,
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
) -> np.ndarray:
    times = group["t_rel_s"].to_numpy(dtype=float)
    if source_id.startswith("Real-"):
        return np.ones(len(group), dtype=bool) if flight_label != "normal" else np.zeros(len(group), dtype=bool)
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


def _run_matrix(
    *,
    raw: pd.DataFrame,
    folds: dict[str, dict],
    selected_splits: tuple[str, ...],
    output: Path,
    flight_labels: dict[str, str],
    sead_t0: dict[str, float],
    sead_ranges: dict[str, list[tuple[float, float]]],
    source_note: str,
    input_hashes: dict[str, str],
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    unknown = set(selected_splits) - set(folds)
    if unknown:
        raise ValueError(f"Unknown split names: {sorted(unknown)}")
    folds = {name: folds[name] for name in selected_splits}

    all_metrics: list[dict] = []
    all_labels: list[dict] = []
    all_groups: list[dict] = []
    all_diagnosis: list[dict] = []
    module_definitions = {**PX4_ML9_CANDIDATE_MODULES, **PX4_ML12_THIN_MODULES}

    for split_name, split in folds.items():
        seed = int(split["seed"])
        parts = {name: set(split[name]) for name in ("train", "val", "test")}
        scaler = fit_scaler_params(
            raw[raw["source_id"].isin(parts["train"])],
            feature_columns(raw),
        )
        scaled = apply_scaler_params(raw, scaler)
        fitted = fit_modular_iforest(
            scaled,
            split,
            module_definitions,
            seed=seed,
            n_jobs=1,
        )
        scored = _score_modules(fitted, scaled, parts["val"])
        scored["exploratory_fusion"] = max_score_fusion(
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
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=True), encoding="utf-8")

    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "artifact_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "RFLY-0 exploratory evaluation",
        "source_note": source_note,
        "official_gate": False,
        "status": "exploratory_not_official_gate",
        "rfly_anomaly_truth": "whole-flight proxy; no TestInfo/rfly_ctrl_lxl intervals used",
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


def run_rfly_only(run_name: str, splits: tuple[str, ...]) -> Path:
    rfly_silver = pd.read_parquet(RFLY_SILVER)
    distribution = _distribution_report(rfly_silver)
    OUT_RFLY.mkdir(parents=True, exist_ok=True)
    (OUT_RFLY / "distribution_report.json").write_text(
        json.dumps(_jsonable(distribution), indent=2, ensure_ascii=False), encoding="utf-8",
    )

    provisional = build_px4_features(rfly_silver)
    folds = _build_rfly_splits(provisional)
    raw, cusum = _build_features_with_split0_cusum(rfly_silver, set(folds["split_00"]["train"]))
    flight_labels = flight_label_table(raw).set_index("source_id")["flight_label"].to_dict()
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
        source_note="rflymad-only exploratory low-normal-quota run",
        input_hashes={"rfly_silver_sha256": _sha256(RFLY_SILVER)},
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


def run_pooled(run_name: str, splits: tuple[str, ...]) -> Path:
    split_manifest = json.loads(SEAD_SPLIT.read_text(encoding="utf-8"))
    sead_config = split_manifest["sources"]["uav_sead"]
    sead_folds, sead_development, sead_holdout = _selected_sead_development(sead_config, splits)
    rfly_silver = pd.read_parquet(RFLY_SILVER)
    rfly_provisional = build_px4_features(rfly_silver)
    rfly_folds = _build_rfly_splits(rfly_provisional)

    sead_silver = pd.read_parquet(
        SEAD_SILVER,
        filters=[("source_id", "in", sorted(sead_development))],
    )
    if set(sead_silver["source_id"].unique()) & sead_holdout:
        raise AssertionError("SEAD blind holdout rows were read in pooled exploratory run")
    combined_silver = pd.concat([sead_silver, rfly_silver], ignore_index=True, sort=False)
    split0_train = set(sead_folds["split_00"]["train"]) | set(rfly_folds["split_00"]["train"])
    raw, cusum = _build_features_with_split0_cusum(combined_silver, split0_train)

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
        source_note="SEAD development + RFLY exploratory pooled-normal run; ALFA excluded",
        input_hashes={
            "rfly_silver_sha256": _sha256(RFLY_SILVER),
            "sead_silver_sha256": _sha256(SEAD_SILVER),
            "sead_split_manifest_sha256": _sha256(SEAD_SPLIT),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rfly-only", "pooled", "both"), default="both")
    parser.add_argument("--run-name", default="exploratory_full")
    parser.add_argument("--splits", nargs="+", default=[f"split_{i:02d}" for i in range(5)])
    args = parser.parse_args()
    splits = tuple(args.splits)
    outputs = []
    if args.mode in {"rfly-only", "both"}:
        outputs.append(run_rfly_only(args.run_name, splits))
    if args.mode in {"pooled", "both"}:
        outputs.append(run_pooled(args.run_name, splits))
    for output in outputs:
        print(f"Exploratory artifact: {output}")
        print((output / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
