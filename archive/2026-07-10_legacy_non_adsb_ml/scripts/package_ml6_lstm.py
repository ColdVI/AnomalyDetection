"""ALFA ve UAV Attack split_00 LSTM-AE checkpoint bundle'larini uret."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.ml.artifacts import save_lstm_bundle
from src.ml.data.scaling import apply_scaler_params
from src.ml.data.windowing import build_windows
from src.ml.models.lstm_autoencoder import (
    AE_FEATURES, STRIDE, WINDOW, LSTMAutoencoder,
    reconstruction_scores, train_lstm_autoencoder,
)
from src.ml.training_log import write_training_log

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.set_num_threads(4)
    feat_root = ROOT / "data/gold/ml_features"
    manifest = json.loads((feat_root / "split_manifest.json").read_text(encoding="utf-8"))
    for source in ["alfa", "uav_attack"]:
        raw = pd.read_parquet(feat_root / source / f"{source}_ml_features.parquet")
        if source == "alfa":
            raw = raw[raw["label"] != "unknown"]
        scaler = json.loads((ROOT / "artifacts/scalers" / f"{source}_robust_scaler.json").read_text())
        cols = [c for c in AE_FEATURES[source] if c in raw.columns]
        scaled = apply_scaler_params(raw, scaler)
        sequence = scaled[["source_id", "t_rel_s", "label"]].copy()
        for col in cols:
            sequence[col] = scaled[col].where(raw[col].notna())
        x, mask, meta = build_windows(
            sequence, cols, window=WINDOW[source], stride=STRIDE[source], max_gap_s=2.0)
        split = manifest["sources"][source]["splits"]["split_00"]
        train_idx = meta["source_id"].isin(split["train"]).to_numpy()
        val_idx = meta["source_id"].isin(split["val"]).to_numpy()
        model, training = train_lstm_autoencoder(
            LSTMAutoencoder(len(cols)), x[train_idx], mask[train_idx],
            x[val_idx], mask[val_idx], seed=split["seed"])
        # Kalici egitim izi kurali (ML-11 Bolum 5): her egitim loss izi birakir.
        write_training_log(training.pop("history"), source, "ml6_lstm_ae",
                           datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        val_scores = reconstruction_scores(model, x[val_idx], mask[val_idx])
        threshold = float(np.quantile(val_scores, 0.99))
        path = save_lstm_bundle(
            model, ROOT / "artifacts/models" / source / "ml6_lstm_ae",
            scaler_params=scaler,
            calibration={"window_threshold_q99": threshold},
            metadata={
                "source": source, "model_version": "ml6_lstm_ae",
                "feature_version": "causal_features_v1", "split_id": "split_00",
                "feature_columns": cols, "window": WINDOW[source], "stride": STRIDE[source],
                "max_gap_s": 2.0, "training": training,
                "train_flights": split["train"], "validation_flights": split["val"],
                "blind_holdout_used": False,
            })
        print(f"{source}: {path}")


if __name__ == "__main__":
    main()
