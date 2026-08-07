"""Create a non-overwriting, explicitly retrained UAV-SEAD LSTM-AE bundle."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.artifacts import save_lstm_bundle
from src.ml.data.scaling import apply_scaler_params
from src.ml.data.windowing import build_windows
from src.ml.models.lstm_autoencoder import (
    AE_FEATURES,
    STRIDE,
    WINDOW,
    LSTMAutoencoder,
    reconstruction_scores,
    train_lstm_autoencoder,
)
from src.ml.training_log import write_training_log


def main() -> None:
    source = "uav_sead"
    output = ROOT / "artifacts/models/uav_sead/ml8a_retrained_lstm_ae"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")

    torch.set_num_threads(4)
    feature_root = ROOT / "data/gold/ml_features"
    manifest = json.loads((feature_root / "split_manifest.json").read_text(encoding="utf-8"))
    split = manifest["sources"][source]["splits"]["split_00"]
    allowed = set(split["train"]) | set(split["val"])
    holdout = set(split["final_holdout"])
    assert not allowed & holdout

    path = feature_root / source / f"{source}_ml_features.parquet"
    cols = AE_FEATURES[source]
    raw = pd.read_parquet(
        path,
        columns=["source_id", "t_rel_s", "label", *cols],
        filters=[("source_id", "in", sorted(allowed))],
    )
    assert not set(raw["source_id"].unique()) & holdout
    scaler = json.loads(
        (ROOT / "artifacts/scalers/uav_sead_robust_scaler.json").read_text(encoding="utf-8")
    )
    scaled = apply_scaler_params(raw, scaler)
    sequence = scaled[["source_id", "t_rel_s", "label"]].copy()
    for col in cols:
        sequence[col] = scaled[col].where(raw[col].notna())
    x, mask, meta = build_windows(
        sequence,
        cols,
        window=WINDOW[source],
        stride=STRIDE[source],
        max_gap_s=2.0,
    )
    train_idx = meta["source_id"].isin(split["train"]).to_numpy()
    val_idx = meta["source_id"].isin(split["val"]).to_numpy()
    if not train_idx.any() or not val_idx.any():
        raise RuntimeError("SEAD LSTM train/validation windows are empty")

    model, training = train_lstm_autoencoder(
        LSTMAutoencoder(len(cols)),
        x[train_idx],
        mask[train_idx],
        x[val_idx],
        mask[val_idx],
        seed=int(split["seed"]),
    )
    # Kalici egitim izi kurali (ML-11 Bolum 5): her egitim loss izi birakir.
    write_training_log(training.pop("history"), source, "ml8a_retrained_lstm_ae",
                       datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    val_scores = reconstruction_scores(model, x[val_idx], mask[val_idx])
    threshold = float(np.quantile(val_scores, 0.99))
    bundle = save_lstm_bundle(
        model,
        output,
        scaler_params=scaler,
        calibration={"window_threshold_q99": threshold},
        metadata={
            "source": source,
            "model_version": "ml8a_retrained_lstm_ae",
            "comparison_status": "retrained_during_ml8a_not_original_ml6_ml7_baseline",
            "feature_version": "causal_features_v1_px4_shared",
            "split_id": "split_00_novelty_normal_only",
            "feature_columns": cols,
            "window": WINDOW[source],
            "stride": STRIDE[source],
            "max_gap_s": 2.0,
            "training": training,
            "train_flights": split["train"],
            "validation_flights": split["val"],
            "blind_holdout_used": False,
        },
    )
    print(bundle)


if __name__ == "__main__":
    main()
