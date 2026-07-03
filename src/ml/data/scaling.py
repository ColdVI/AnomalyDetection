"""Train-only RobustScaler (ML-0 fazi).

Scaler YALNIZCA train (normal ucuslar) uzerinde fit edilir -- val/test
istatistikleri fit'e sizarsa threshold sahte iyimser olur. RobustScaler
(medyan/IQR) secildi cunku anomali iceren kuyruklara mean/std'den dayanikli.

Parametreler JSON olarak artifacts/scalers/ altina yazilir: model kodu
sklearn objesine degil, kalici/incelenebilir parametrelere bagimli olsun.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


def fit_scaler_params(train_features: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train setinde RobustScaler fit edip {kolon: {center, scale}} dondurur.

    Tamamen-NaN veya sabit kolonlar scale=1 ile gecirilir (bolme hatasi yok);
    NaN'lar fit oncesi kolon medyaniyla doldurulur (impute istatistigi de
    train-only'dir ve parametrelere kaydedilir).
    """
    params: dict = {"feature_columns": feature_cols, "columns": {}}
    X = train_features[feature_cols].astype(float)
    medians = X.median()
    X = X.fillna(medians)
    scaler = RobustScaler().fit(X)
    for i, col in enumerate(feature_cols):
        center = float(scaler.center_[i]) if np.isfinite(scaler.center_[i]) else 0.0
        scale = float(scaler.scale_[i])
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        impute = float(medians[col]) if np.isfinite(medians[col]) else 0.0
        params["columns"][col] = {"center": center, "scale": scale, "impute": impute}
    return params


def apply_scaler_params(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Kaydedilmis parametrelerle olcekler: (x - center) / scale, NaN -> impute."""
    out = df.copy()
    for col, p in params["columns"].items():
        if col not in out.columns:
            continue
        out[col] = (out[col].astype(float).fillna(p["impute"]) - p["center"]) / p["scale"]
    return out


def write_scaler_params(params: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    logger.info("Scaler parametreleri yazildi: %s (%d kolon)", path, len(params["columns"]))
