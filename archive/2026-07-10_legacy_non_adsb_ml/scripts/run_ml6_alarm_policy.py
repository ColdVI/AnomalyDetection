"""ML-6: development verisinde iki kademeli SEAD alarm politikasi sec.

Bu script final holdout'u bilerek skorlamaz. Paketlenmis split_00 modelini
kullanir, alarm adaylarini development event etiketlerinde tarar ve mentor/
urun onayi icin tekrarlanabilir bir policy-candidate artifact'i yazar.
"""

from __future__ import annotations

import hashlib
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.artifacts import load_modular_iforest_bundle
from src.ml.data.scaling import apply_scaler_params
from src.ml.evaluation.events import event_metrics
from src.ml.models.modular_iforest import anomaly_scores

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts/models/uav_sead/ml6_modular_iforest"
FEATURE_PATH = ROOT / "data/gold/ml_features/uav_sead/uav_sead_ml_features.parquet"
SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
LABEL_PATH = ROOT / "data/objectstore/bronze/uav_sead/labels.json"
SILVER_PATH = ROOT / "data/silver/uav_sead_silver.parquet"

# Bunlar teknik gercek degil, ilk operasyonel kabul varsayimlaridir. Iki kademe
# tek bir esikte hem dusuk FA hem yuksek recall varmis gibi davranmayi engeller.
TIER_CONFIG = {
    "critical_signal": {
        "max_false_alarms_per_hour": 2.0,
        "min_event_recall": 0.30,
        "allowed_fusions": {"sinyal_kalitesi", "vote2", "vote3"},
        "meaning": "operator notification; GPS/signal-integrity odakli",
    },
    "advisory_warning": {
        "max_false_alarms_per_hour": 12.0,
        "min_event_recall": 0.50,
        "allowed_fusions": {"max", "nav_butunlugu", "vote2",
                            "irtifa_tutarliligi", "kontrol_cevabi", "ekf_redleri"},
        "meaning": "dashboard/inceleme uyarisi; otomatik aksiyon yok",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labels_and_ranges() -> tuple[dict, dict, dict]:
    raw = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
    labels, classes, ranges = {}, {}, {}
    for source_id, meta in raw.items():
        labels[source_id] = meta.get("label", "unknown")
        classes[source_id] = meta.get("class", labels[source_id])
        spans: list[tuple[float, float]] = []
        for annotation in meta.get("ranges", []):
            for _, intervals in annotation:
                spans.extend((float(start), float(end)) for start, end in intervals)
        ranges[source_id] = spans
    return labels, classes, ranges


def _score_development(model_dir: Path = MODEL_DIR) -> tuple[pd.DataFrame, dict, dict, list[str]]:
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split = manifest["sources"]["uav_sead"]["splits"]["split_00"]
    fitted, model_manifest = load_modular_iforest_bundle(model_dir)
    if model_manifest.get("blind_holdout_used") is not False:
        raise RuntimeError("Model artifact blind-holdout temizligini kanitlamiyor")
    if set(split["test"]) & set(split["final_holdout"]):
        raise RuntimeError("Development ve final holdout kesisiyor")

    scaler = json.loads((model_dir / "scaler.json").read_text(encoding="utf-8"))
    raw = pd.read_parquet(FEATURE_PATH)
    scaled = apply_scaler_params(raw, scaler)
    development = scaled[scaled["source_id"].isin(split["test"])].copy()
    module_names = []
    for name, item in fitted.items():
        score = anomaly_scores(item["model"], development[item["feature_columns"]])
        threshold = max(float(item["row_threshold_q99"]), np.finfo(float).eps)
        development[name] = score / threshold
        module_names.append(name)

    values = development[module_names].to_numpy(dtype=float)
    development["max"] = values.max(axis=1)
    development["vote2"] = np.partition(values, -2, axis=1)[:, -2]
    development["vote3"] = values.min(axis=1)
    return development, split, model_manifest, module_names


def _flight_inputs(development: pd.DataFrame,
                   module_names: list[str]) -> tuple[list[dict], list[str]]:
    labels, classes, ranges = _labels_and_ranges()
    t0 = pd.read_parquet(
        SILVER_PATH, columns=["source_id", "timestamp"]
    ).groupby("source_id")["timestamp"].min().to_dict()
    flights, excluded = [], []
    score_columns = module_names + ["max", "vote2", "vote3"]
    for source_id, group in development.groupby("source_id"):
        spans = ranges.get(source_id, [])
        is_normal = labels.get(source_id) == "normal"
        if not is_normal and not spans:
            excluded.append(source_id)
            continue
        group = group.sort_values("t_rel_s")
        t_s = group["t_rel_s"].to_numpy(dtype=float)
        absolute_us = float(t0[source_id]) + t_s * 1e6
        y_true = np.zeros(len(group), dtype=bool)
        for start, end in spans:
            y_true |= (absolute_us >= start) & (absolute_us <= end)
        flights.append({
            "source_id": source_id,
            "flight_label": labels.get(source_id, "unknown"),
            "anomaly_class": classes.get(source_id, "unknown"),
            "t_s": t_s,
            "y_true": y_true,
            "scores": {col: group[col].to_numpy(dtype=float) for col in score_columns},
        })
    return flights, excluded


def _evaluate(flights: list[dict], *, fusion: str, threshold: float,
              k: int, n: int, clear_s: float, cooldown_s: float) -> tuple[dict, pd.DataFrame]:
    rows = []
    for flight in flights:
        metrics = event_metrics(
            flight["t_s"], flight["y_true"], flight["scores"][fusion], threshold,
            k=k, n=n, clear_s=clear_s, cooldown_s=cooldown_s)
        rows.append({
            "source_id": flight["source_id"],
            "flight_label": flight["flight_label"],
            "anomaly_class": flight["anomaly_class"],
            "n_samples": len(flight["t_s"]),
            **metrics,
        })
    detail = pd.DataFrame(rows)
    n_events = int(detail["n_events"].sum())
    detected = int(detail["detected_events"].sum())
    overlap_detected = int(detail["overlap_detected_events"].sum())
    preexisting = int(detail["preexisting_alarm_events"].sum())
    false_events = int(detail["false_alarm_events"].sum())
    normal_hours = float(detail["normal_hours"].sum())
    delay_weight = detail["mean_detection_delay_s"] * detail["detected_events"]
    normal_rates = detail.loc[
        (detail["flight_label"] == "normal") & (detail["normal_hours"] > 0),
        "false_alarms_per_hour",
    ]
    summary = {
        "fusion": fusion,
        "threshold_ratio": threshold,
        "k": k,
        "n": n,
        "clear_s": clear_s,
        "cooldown_s": cooldown_s,
        "n_events": n_events,
        "detected_events": detected,
        "event_recall": detected / n_events if n_events else np.nan,
        "overlap_detected_events": overlap_detected,
        "event_overlap_recall": overlap_detected / n_events if n_events else np.nan,
        "preexisting_alarm_events": preexisting,
        "mean_detection_delay_s": (float(delay_weight.sum() / detected)
                                   if detected else np.nan),
        "false_alarm_events": false_events,
        "normal_hours": normal_hours,
        "false_alarms_per_hour": (false_events / normal_hours
                                  if normal_hours else np.nan),
        "normal_flight_fa_per_hour_p95": (float(normal_rates.quantile(0.95))
                                          if len(normal_rates) else np.nan),
        "alarm_fraction": float(np.average(
            detail["alarm_fraction"], weights=detail["n_samples"])),
    }
    return summary, detail


def _candidate_grid(flights: list[dict], module_names: list[str]) -> pd.DataFrame:
    rows = []
    for fusion in module_names + ["max", "vote2", "vote3"]:
        for threshold in [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]:
            for k, n in [(2, 3), (3, 5), (5, 9)]:
                for clear_s, cooldown_s in [(5.0, 10.0), (10.0, 30.0)]:
                    summary, _ = _evaluate(
                        flights, fusion=fusion, threshold=threshold,
                        k=k, n=n, clear_s=clear_s, cooldown_s=cooldown_s)
                    rows.append(summary)
    return pd.DataFrame(rows)


def _select_tier(candidates: pd.DataFrame, config: dict) -> dict:
    within_budget = candidates[
        candidates["fusion"].isin(config["allowed_fusions"])
        & (candidates["false_alarms_per_hour"]
           <= config["max_false_alarms_per_hour"])
    ].copy()
    if within_budget.empty:
        raise RuntimeError(f"Alarm butcesini karsilayan aday yok: {config}")
    ranked = within_budget.sort_values(
        ["event_recall", "false_alarms_per_hour",
         "normal_flight_fa_per_hour_p95", "mean_detection_delay_s"],
        ascending=[False, True, True, True],
    )
    best = ranked.iloc[0]
    best_result = {key: value.item() if hasattr(value, "item") else value
                   for key, value in best.to_dict().items()}
    requirements = {
        "max_false_alarms_per_hour": config["max_false_alarms_per_hour"],
        "min_event_recall": config["min_event_recall"],
    }
    eligible = ranked[ranked["event_recall"] >= config["min_event_recall"]]
    if eligible.empty:
        return {
            "status": "rejected; no candidate meets minimum utility",
            "requirements": requirements,
            "meaning": config["meaning"],
            "best_within_false_alarm_budget": best_result,
        }
    chosen = eligible.sort_values(
        ["event_recall", "false_alarms_per_hour",
         "normal_flight_fa_per_hour_p95", "mean_detection_delay_s"],
        ascending=[False, True, True, True],
    ).iloc[0]
    result = {key: value.item() if hasattr(value, "item") else value
              for key, value in chosen.to_dict().items()}
    return {
        "status": "development_candidate",
        "requirements": requirements,
        "meaning": config["meaning"],
        "policy": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--run-name", default="sead_alarm_policy")
    args = parser.parse_args()
    model_dir = args.model_dir if args.model_dir.is_absolute() else ROOT / args.model_dir

    development, split, model_manifest, module_names = _score_development(model_dir)
    flights, excluded = _flight_inputs(development, module_names)
    candidates = _candidate_grid(flights, module_names)
    decisions = {tier: _select_tier(candidates, config)
                 for tier, config in TIER_CONFIG.items()}

    result_dir = ROOT / "data/gold/ml6"
    result_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = result_dir / f"{args.run_name}_sweep.csv"
    candidates.to_csv(sweep_path, index=False)

    class_rows = []
    for tier, decision in decisions.items():
        policy = (decision.get("policy")
                  or decision["best_within_false_alarm_budget"])
        _, detail = _evaluate(
            flights, fusion=policy["fusion"], threshold=policy["threshold_ratio"],
            k=policy["k"], n=policy["n"], clear_s=policy["clear_s"],
            cooldown_s=policy["cooldown_s"])
        anomalous = detail[detail["flight_label"] != "normal"]
        for anomaly_class, group in anomalous.groupby("anomaly_class"):
            n_events = int(group["n_events"].sum())
            detected = int(group["detected_events"].sum())
            class_rows.append({
                "tier": tier,
                "tier_status": decision["status"],
                "anomaly_class": anomaly_class,
                "n_flights": int(group["source_id"].nunique()),
                "n_events": n_events,
                "detected_events": detected,
                "event_recall": detected / n_events if n_events else np.nan,
            })
    by_class_path = result_dir / f"{args.run_name}_by_class.csv"
    pd.DataFrame(class_rows).to_csv(by_class_path, index=False)

    artifact = {
        "policy_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "uav_sead",
        "status": ("development_rejected; model/feature revision required"
                   if any(d["status"].startswith("rejected") for d in decisions.values())
                   else "development_candidate; requires operational-budget approval"),
        "blind_holdout_used": False,
        "blind_holdout_locked": True,
        "selection_data": "split_00 development test only",
        "development_flights": len(split["test"]),
        "final_holdout_flights_unopened": len(split["final_holdout"]),
        "excluded_anomalous_flights_without_ranges": excluded,
        "score_definition": (
            "module IsolationForest row score / module normal-val Q99; "
            "max=OR, vote2=second-highest module ratio, vote3=minimum ratio"
        ),
        "budgets_are_product_assumptions": True,
        "tier_decisions": decisions,
        "inputs": {
            "model_version": model_manifest["model_version"],
            "model_manifest_sha256": _sha256(model_dir / "manifest.json"),
            "split_manifest_sha256": _sha256(SPLIT_PATH),
        },
        "development_outputs": {
            "sweep": str(sweep_path.relative_to(ROOT)).replace("\\", "/"),
            "by_class": str(by_class_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    policy_dir = ROOT / "artifacts/policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = args.run_name.removeprefix("sead_")
    policy_path = policy_dir / f"uav_sead_{artifact_name}_candidate.json"
    policy_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    for tier, decision in decisions.items():
        policy = (decision.get("policy")
                  or decision["best_within_false_alarm_budget"])
        print(tier, decision["status"], {
            key: policy[key] for key in [
                "fusion", "threshold_ratio", "k", "n", "clear_s", "cooldown_s",
                "event_recall", "event_overlap_recall", "preexisting_alarm_events",
                "false_alarms_per_hour",
                "normal_flight_fa_per_hour_p95", "mean_detection_delay_s",
            ]
        })
    print(f"\nPolicy adayi: {policy_path}")
    print("Final holdout acilmadi.")


if __name__ == "__main__":
    main()
