"""Causal K-of-N decisions and validation-normal threshold calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def alarm_states(
    frame: pd.DataFrame,
    *,
    score_col: str,
    threshold: float,
    k: int = 2,
    n: int = 3,
) -> pd.Series:
    """Return the row-level binary anomaly state after the K-of-N rule."""

    result = np.zeros(len(frame), dtype=bool)
    work = frame.copy()
    work["__row_position"] = np.arange(len(work))
    for _, group in work.groupby("flight_id", sort=False):
        ordered = group.sort_values("timestamp_utc")
        exceed = ordered[score_col].gt(threshold) & ordered[score_col].notna()
        active = exceed.rolling(n, min_periods=n).sum().ge(k)
        result[ordered["__row_position"].to_numpy(dtype=int)] = active.to_numpy()
    return pd.Series(result, index=frame.index)


def alarm_onsets(
    frame: pd.DataFrame,
    *,
    score_col: str,
    threshold: float,
    k: int = 2,
    n: int = 3,
) -> pd.Series:
    """Return only false-to-true transitions of the binary anomaly state."""

    states = alarm_states(
        frame, score_col=score_col, threshold=threshold, k=k, n=n
    )
    result = np.zeros(len(frame), dtype=bool)
    work = frame.copy()
    work["__row_position"] = np.arange(len(work))
    work["__alarm_state"] = states.to_numpy()
    for _, group in work.groupby("flight_id", sort=False):
        ordered = group.sort_values("timestamp_utc")
        active = ordered["__alarm_state"].astype(bool)
        onsets = active & ~active.shift(1, fill_value=False)
        result[ordered["__row_position"].to_numpy(dtype=int)] = onsets.to_numpy()
    return pd.Series(result, index=frame.index)


def exposure_hours(frame: pd.DataFrame) -> float:
    durations = frame.groupby("flight_id")["timestamp_utc"].agg(lambda values: values.max() - values.min())
    return float(durations.clip(lower=0.0).sum() / 3600.0)


def false_events_per_hour(
    frame: pd.DataFrame,
    *,
    score_col: str,
    threshold: float,
    k: int = 2,
    n: int = 3,
) -> float:
    hours = exposure_hours(frame)
    if hours <= 0.0:
        return float("inf")
    return float(alarm_onsets(frame, score_col=score_col, threshold=threshold, k=k, n=n).sum() / hours)


def calibrate_threshold(
    validation_normal: pd.DataFrame,
    *,
    score_col: str,
    target_false_events_per_hour: float = 0.1,
    k: int = 2,
    n: int = 3,
) -> dict:
    finite = validation_normal.loc[
        validation_normal[score_col].notna() & validation_normal["quality_good"], score_col
    ].to_numpy(dtype=float)
    if not len(finite):
        raise ValueError(f"No finite validation scores for {score_col}")
    candidates = np.unique(np.quantile(finite, np.linspace(0.80, 1.0, 401)))
    candidates = np.r_[candidates, np.inf]
    for threshold in candidates:
        fa = false_events_per_hour(
            validation_normal,
            score_col=score_col,
            threshold=float(threshold),
            k=k,
            n=n,
        )
        if fa <= target_false_events_per_hour:
            return {
                "threshold": float(threshold),
                "validation_false_events_per_hour": fa,
                "target_false_events_per_hour": float(target_false_events_per_hour),
                "k": int(k),
                "n": int(n),
            }
    raise RuntimeError("No threshold satisfies the false-event budget")
