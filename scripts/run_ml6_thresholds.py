"""ML-6: varsayilan EKF'siz SEAD modulleri icin val-max ve POT karsilastirmasi."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.ml.data.scaling import apply_scaler_params
from src.ml.evaluation.thresholds import pot_threshold
from src.ml.models.modular_iforest import PX4_BASE_MODULES, anomaly_scores

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    feat = ROOT / "data/gold/ml_features"
    manifest = json.loads((feat / "split_manifest.json").read_text(encoding="utf-8"))
    scaler = json.loads((ROOT / "artifacts/scalers/uav_sead_robust_scaler.json").read_text())
    df = apply_scaler_params(pd.read_parquet(
        feat / "uav_sead/uav_sead_ml_features.parquet"), scaler)
    labels = manifest["sources"]["uav_sead"]["flight_labels"]
    rows = []
    for split_name, split in manifest["sources"]["uav_sead"]["splits"].items():
        train = df[df["source_id"].isin(split["train"])]
        val = df[df["source_id"].isin(split["val"])]
        test = df[df["source_id"].isin(split["test"])]  # blind holdout haric
        fused_val = fused_test = None
        for cols in PX4_BASE_MODULES.values():
            cols = [c for c in cols if c in df.columns]
            model = IsolationForest(
                n_estimators=300, max_samples=256,
                random_state=split["seed"], n_jobs=1).fit(train[cols])
            val_f = val.assign(_s=anomaly_scores(model, val[cols])).groupby("source_id")["_s"].max()
            test_f = test.assign(_s=anomaly_scores(model, test[cols])).groupby("source_id")["_s"].max()
            scale = max(float(val_f.median()), np.finfo(float).eps)
            fused_val = val_f / scale if fused_val is None else np.maximum(fused_val, val_f / scale)
            fused_test = test_f / scale if fused_test is None else np.maximum(fused_test, test_f / scale)
        y = np.array([0 if labels[f] == "normal" else 1 for f in fused_test.index])
        for method, threshold in [
            ("val-max", float(fused_val.max())),
            ("POT", pot_threshold(fused_val.values)),
        ]:
            rows.append({
                "split": split_name, "threshold_method": method, "threshold": threshold,
                "detection": float((fused_test[y == 1] > threshold).mean()),
                "false_alarm": float((fused_test[y == 0] > threshold).mean()),
            })
    result = pd.DataFrame(rows)
    out = ROOT / "data/gold/ml6/sead_base_thresholds.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result.groupby("threshold_method")[["detection", "false_alarm"]]
          .agg(["mean", "std"]).round(3))
    print(f"\nYazildi: {out}")


if __name__ == "__main__":
    main()
