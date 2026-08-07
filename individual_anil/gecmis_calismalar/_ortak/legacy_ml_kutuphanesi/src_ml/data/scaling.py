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


def infer_source_schema_groups(df: pd.DataFrame) -> pd.Series:
    """Return coarse source-schema groups for structural-missing handling.

    RflyMAD case ids carry their source in the path prefix. Other current PX4
    development sources are never pooled with a second schema except SEAD+RFLY,
    so they can share a conservative default group.
    """
    if "source_type" in df.columns:
        return df["source_type"].astype(str)
    source_ids = df["source_id"].astype(str)
    return source_ids.map(
        lambda value: "rflymad"
        if value.startswith(("Real-", "SampleData/"))
        else "default"
    )


def infer_source_column_presence(
    frame: pd.DataFrame,
    feature_cols: list[str],
    source_groups: pd.Series | None = None,
) -> dict[str, list[str]]:
    """Infer which feature columns are structurally present per source group.

    A column is considered present for a source group only if at least one row in
    that group has a finite/non-null value. Columns absent from a source schema
    after a pooled concat therefore remain NaN during scaling instead of being
    filled with another source's train median.
    """
    groups = source_groups if source_groups is not None else infer_source_schema_groups(frame)
    if len(groups) != len(frame):
        raise ValueError("source_groups length must match frame length")
    groups = pd.Series(groups.to_numpy(), index=frame.index)
    presence: dict[str, list[str]] = {}
    for group_name, index in groups.groupby(groups).groups.items():
        subset = frame.loc[index]
        presence[str(group_name)] = [
            col for col in feature_cols
            if col in subset.columns and subset[col].notna().any()
        ]
    return presence


def fit_scaler_params(
    train_features: pd.DataFrame,
    feature_cols: list[str],
    *,
    source_groups: pd.Series | None = None,
    source_presence: dict[str, list[str]] | None = None,
) -> dict:
    """Train setinde RobustScaler fit edip {kolon: {center, scale}} dondurur.

    Tamamen-NaN veya sabit kolonlar scale=1 ile gecirilir (bolme hatasi yok);
    NaN'lar fit oncesi kolon medyaniyla doldurulur (impute istatistigi de
    train-only'dir ve parametrelere kaydedilir).
    """
    params: dict = {"feature_columns": feature_cols, "columns": {}}
    if source_presence is None and source_groups is not None:
        source_presence = infer_source_column_presence(
            train_features, feature_cols, source_groups,
        )
    if source_presence is not None:
        params["source_column_presence"] = {
            str(group): sorted(set(cols)) for group, cols in source_presence.items()
        }
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


def apply_scaler_params(
    df: pd.DataFrame,
    params: dict,
    *,
    source_groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Kaydedilmis parametrelerle olcekler: (x - center) / scale, NaN -> impute."""
    out = df.copy()
    for col, p in params["columns"].items():
        if col not in out.columns:
            continue
        out[col] = (out[col].astype(float).fillna(p["impute"]) - p["center"]) / p["scale"]
    presence = params.get("source_column_presence")
    if presence:
        groups = source_groups if source_groups is not None else infer_source_schema_groups(df)
        if len(groups) != len(out):
            raise ValueError("source_groups length must match frame length")
        groups = pd.Series(groups.to_numpy(), index=out.index)
        all_columns = set(params["columns"])
        for group_name, index in groups.groupby(groups).groups.items():
            present = set(presence.get(str(group_name), all_columns))
            absent = sorted(all_columns - present)
            if absent:
                out.loc[index, absent] = np.nan
    return out


def write_scaler_params(params: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    logger.info("Scaler parametreleri yazildi: %s (%d kolon)", path, len(params["columns"]))
