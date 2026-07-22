"""Frozen train-normal median/MAD scaling for RESIDUAL-V1 residual channels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustScalerParams:
    channel: str
    median: float
    mad: float
    clip: float
    fit_rows: int

    def to_dict(self) -> dict:
        return asdict(self)


class ZeroMADChannel(ValueError):
    """Raised so a constant train-normal channel is explicitly excluded."""


def fit_robust_scaler(
    residual: pd.Series,
    train_normal: pd.Series,
    *,
    channel: str,
    clip: float = 8.0,
) -> RobustScalerParams:
    values = pd.to_numeric(residual, errors="coerce")
    mask = train_normal.astype(bool) & values.notna()
    fit = values.loc[mask]
    if fit.empty:
        raise ValueError(f"{channel}: no finite train-normal residuals")
    median = float(fit.median())
    mad = float((fit - median).abs().median())
    if not np.isfinite(mad) or mad == 0.0:
        raise ZeroMADChannel(f"{channel}: train-normal MAD is zero")
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("clip must be finite and positive")
    return RobustScalerParams(
        channel=channel,
        median=median,
        mad=mad,
        clip=float(clip),
        fit_rows=int(len(fit)),
    )


def robust_z(residual: pd.Series, params: RobustScalerParams) -> pd.Series:
    values = pd.to_numeric(residual, errors="coerce")
    scaled = ((values - params.median) / params.mad).clip(-params.clip, params.clip)
    return scaled.rename("z")
