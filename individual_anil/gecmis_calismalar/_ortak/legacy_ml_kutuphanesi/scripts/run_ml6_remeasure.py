"""Causal CUSUM sonrasinda temel ML-1 bulgularini yeniden olc."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.ml.data.scaling import apply_scaler_params
from src.ml.models.modular_iforest import (
    ALFA_MODULES, PX4_BASE_MODULES, fit_modular_iforest, score_flights,
)

ROOT = Path(__file__).resolve().parents[1]


def run_source(source: str, modules: dict) -> pd.DataFrame:
    feat_root = ROOT / "data/gold/ml_features"
    manifest = json.loads((feat_root / "split_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["sources"][source]
    scaler = json.loads((ROOT / "artifacts/scalers" / f"{source}_robust_scaler.json").read_text())
    raw = pd.read_parquet(feat_root / source / f"{source}_ml_features.parquet")
    if source == "alfa":
        raw = raw[raw["label"] != "unknown"]
    scaled = apply_scaler_params(raw, scaler)
    rows = []
    for split_name, split in entry["splits"].items():
        fitted = fit_modular_iforest(scaled, split, modules, seed=split["seed"])
        test = scaled[scaled["source_id"].isin(split["test"])]
        scored = score_flights(fitted, test)
        y = np.array([0 if entry["flight_labels"][f] == "normal" else 1 for f in scored.index])
        rows.append({
            "source": source, "split": split_name,
            "flight_roc": roc_auc_score(y, scored["fusion"]),
            "detection_at_1": float((scored.loc[y == 1, "fusion"] > 1).mean()),
            "false_alarm_at_1": float((scored.loc[y == 0, "fusion"] > 1).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    feat_root = ROOT / "data/gold/ml_features"
    manifest = json.loads((feat_root / "split_manifest.json").read_text(encoding="utf-8"))
    alfa = pd.read_parquet(feat_root / "alfa/alfa_ml_features.parquet")
    known = alfa[alfa["label"] != "unknown"]
    labels = manifest["sources"]["alfa"]["flight_labels"]
    grouped = known.groupby("source_id")
    single_rows = []
    for col in ["alt_error_cusum_pos", "alt_error", "climb_residual", "xtrack_error",
                "roll_error_cusum_pos", "roll_spec_energy_5s", "energy_rate"]:
        if col in known:
            score = grouped[col].apply(lambda x: x.abs().max())
            valid = score.dropna()
            y = np.array([0 if labels[f] == "normal" else 1 for f in valid.index])
            single_rows.append({"feature": col, "flight_roc": roc_auc_score(y, valid.values),
                                "n_flights": len(valid)})
    single = pd.DataFrame(single_rows).sort_values("flight_roc", ascending=False)
    results = pd.concat([
        run_source("alfa", ALFA_MODULES),
        run_source("uav_attack", PX4_BASE_MODULES),
    ], ignore_index=True)
    out_dir = ROOT / "data/gold/ml6"
    out_dir.mkdir(parents=True, exist_ok=True)
    single.to_csv(out_dir / "causal_cusum_single_features.csv", index=False)
    results.to_csv(out_dir / "causal_modular_remeasure.csv", index=False)
    print("ALFA causal tek feature:\n", single.round(3).to_string(index=False))
    print("\nCausal modular yeniden olcum:\n",
          results.groupby("source")[["flight_roc", "detection_at_1", "false_alarm_at_1"]]
          .agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
