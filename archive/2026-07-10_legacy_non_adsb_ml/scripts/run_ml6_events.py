"""ML-6: SEAD development setinde event metrikleri ve K-of-N persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.ml.data.scaling import apply_scaler_params
from src.ml.evaluation.events import (
    event_metrics,
    load_uav_sead_ranges,
    range_mask,
    uav_sead_absolute_us,
)
from src.ml.models.modular_iforest import PX4_BASE_MODULES, anomaly_scores

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    feat_root = ROOT / "data/gold/ml_features"
    manifest = json.loads((feat_root / "split_manifest.json").read_text(encoding="utf-8"))
    split = manifest["sources"]["uav_sead"]["splits"]["split_00"]
    labels = manifest["sources"]["uav_sead"]["flight_labels"]
    scaler = json.loads((ROOT / "artifacts/scalers/uav_sead_robust_scaler.json").read_text())
    raw = pd.read_parquet(feat_root / "uav_sead/uav_sead_ml_features.parquet")
    scaled = apply_scaler_params(raw, scaler)
    train = scaled[scaled["source_id"].isin(split["train"])]
    val = scaled[scaled["source_id"].isin(split["val"])]
    test_mask = scaled["source_id"].isin(split["test"])
    test = scaled[test_mask].copy()
    fused = np.zeros(len(test), dtype=float)
    for cols in PX4_BASE_MODULES.values():
        cols = [c for c in cols if c in scaled.columns]
        model = IsolationForest(n_estimators=300, max_samples=256, random_state=0, n_jobs=1).fit(train[cols])
        tau = float(np.quantile(anomaly_scores(model, val[cols]), 0.99))
        fused = np.maximum(fused, anomaly_scores(model, test[cols]) / max(tau, np.finfo(float).eps))
    test["score"] = fused

    ranges = load_uav_sead_ranges(ROOT / "data/objectstore/bronze/uav_sead/labels.json")
    t0 = pd.read_parquet(
        ROOT / "data/silver/uav_sead_silver.parquet",
        columns=["source_id", "timestamp"]).groupby("source_id")["timestamp"].min().to_dict()
    rows = []
    for sid, g in test.groupby("source_id"):
        spans = ranges.get(sid, [])
        is_normal = labels[sid] == "normal"
        if not is_normal and not spans:
            continue
        g = g.sort_values("t_rel_s")
        absolute_us = uav_sead_absolute_us(g["t_rel_s"].to_numpy(), t0[sid])
        y = range_mask(absolute_us, spans)
        for policy, k, n in [("1-of-1", 1, 1), ("2-of-3", 2, 3), ("3-of-5", 3, 5)]:
            metrics = event_metrics(
                g["t_rel_s"].to_numpy(), y, g["score"].to_numpy(), 1.0, k=k, n=n)
            rows.append({"source_id": sid, "flight_label": labels[sid],
                         "policy": policy, **metrics})
    result = pd.DataFrame(rows)
    out = ROOT / "data/gold/ml6/sead_event_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    for policy, group in result.groupby("policy"):
        events = int(group["n_events"].sum())
        detected = int(group["detected_events"].sum())
        overlap_detected = int(group["overlap_detected_events"].sum())
        preexisting = int(group["preexisting_alarm_events"].sum())
        false_alarms = int(group["false_alarm_events"].sum())
        hours = float(group["normal_hours"].sum())
        print(policy, {
            "event_recall": round(detected / events, 3) if events else None,
            "event_overlap_recall": round(overlap_detected / events, 3) if events else None,
            "preexisting_alarm_events": preexisting,
            "false_alarms_per_hour": round(false_alarms / hours, 3) if hours else None,
            "events": events, "false_alarm_events": false_alarms,
        })
    print(f"\nYazildi: {out}")


if __name__ == "__main__":
    main()
