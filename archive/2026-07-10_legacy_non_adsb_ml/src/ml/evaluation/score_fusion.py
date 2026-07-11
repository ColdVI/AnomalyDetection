"""Shared score calibration and causal fusion helpers for ML-9/ML-10."""

from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_probability(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map scores to their empirical normal-validation CDF probability."""
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.sort(reference[np.isfinite(reference)])
    if not len(finite):
        raise ValueError("Normal validation score reference is empty")
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    result[valid] = (
        np.searchsorted(finite, values[valid], side="right") + 0.5
    ) / (len(finite) + 1.0)
    return result


def max_score_fusion(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.Series:
    """Causal row-wise maximum of already calibrated score sources."""
    available = [column for column in columns if column in frame.columns]
    if not available:
        raise ValueError("At least one score source is required for fusion")
    return frame[available].max(axis=1)


def last_causal_per_bucket(
    frame: pd.DataFrame,
    *,
    stride_seconds: float,
    columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Select the last observed row in each elapsed-time bucket per flight."""
    frames = []
    keep = list(columns)
    for _, group in frame.sort_values("t_rel_s").groupby("source_id", sort=False):
        group = group.copy()
        group["_bucket"] = np.floor(
            group["t_rel_s"].astype(float) / stride_seconds
        ).astype(np.int64)
        frames.append(group.groupby("_bucket", sort=True).tail(1)[keep])
    return pd.concat(frames, ignore_index=True)
