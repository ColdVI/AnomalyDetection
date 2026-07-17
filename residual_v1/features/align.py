"""Causal natural-rate telemetry alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from residual_v1.ingest.alfa_channels import CHANNELS as ALFA_CHANNELS
from residual_v1.ingest.rfly_channels import CHANNELS as RFLY_CHANNELS


def default_tolerances(dataset: str, *, periods: float = 2.0) -> dict[str, float]:
    """Return topic tolerances in seconds from registered nominal rates."""

    channels = ALFA_CHANNELS if dataset == "alfa" else RFLY_CHANNELS
    tolerance: dict[str, float] = {}
    for channel in channels:
        tolerance[channel.topic] = max(tolerance.get(channel.topic, 0.0), periods / channel.nominal_hz)
    return tolerance


def observed_tolerances(
    flight: Mapping[str, pd.DataFrame],
    registered: Mapping[str, float],
    *,
    median_periods: float = 1.5,
) -> dict[str, float]:
    """Adapt tolerances to a processed topic's observed natural rate."""

    if median_periods <= 0:
        raise ValueError("median_periods must be positive")
    result = dict(registered)
    for topic, frame in flight.items():
        if "t" not in frame or len(frame) < 2:
            continue
        dt = pd.to_numeric(frame["t"], errors="coerce").diff()
        dt = dt[(dt > 0) & np.isfinite(dt)]
        if dt.empty:
            continue
        observed = median_periods * float(dt.median())
        result[topic] = max(float(result.get(topic, 0.0)), observed)
    return result


def _tolerance_for(
    topic: str,
    columns: Sequence[str],
    tolerances: Mapping[str, float],
) -> float:
    candidates = [float(tolerances[key]) for key in (topic, *columns) if key in tolerances]
    if not candidates:
        raise ValueError(f"no alignment tolerance registered for topic {topic}")
    tolerance = max(candidates)
    if tolerance <= 0:
        raise ValueError(f"alignment tolerance must be positive for {topic}")
    return tolerance


def _validate_clock(frame: pd.DataFrame, topic: str) -> pd.DataFrame:
    if "t" not in frame:
        raise ValueError(f"{topic}: missing t")
    result = frame.copy().sort_values("t", kind="stable").reset_index(drop=True)
    time = pd.to_numeric(result["t"], errors="coerce")
    if time.isna().any() or not bool((time.diff().dropna() > 0).all()):
        raise ValueError(f"{topic}: t must be finite and strictly increasing")
    result["t"] = time
    return result


def _stale_mask(
    time: pd.Series,
    topic: str,
    column: str,
    stale: Mapping[str, Sequence[Mapping[str, float]]] | None,
) -> np.ndarray:
    mask = np.zeros(len(time), dtype=bool)
    if not stale:
        return mask
    segments = [*stale.get(topic, ()), *stale.get(column, ())]
    values = time.to_numpy(float)
    for segment in segments:
        mask |= (values >= float(segment["start_s"])) & (values <= float(segment["end_s"]))
    return mask


def align_to_clock(
    flight: Mapping[str, pd.DataFrame],
    clock_topic: str,
    tolerances: Mapping[str, float],
    *,
    stale: Mapping[str, Sequence[Mapping[str, float]]] | None = None,
) -> pd.DataFrame:
    """Causally attach natural-rate topics to one reference clock.

    Values are carried only from the most recent past sample inside the
    registered tolerance. Each attached channel receives a staleness column.
    """

    if clock_topic not in flight:
        raise ValueError(f"missing clock topic: {clock_topic}")
    output = _validate_clock(flight[clock_topic], clock_topic)
    for column in output.columns.difference(["t"]):
        output[f"{column}_staleness_ms"] = 0.0

    for topic, raw in flight.items():
        if topic == clock_topic or raw.empty:
            continue
        source = _validate_clock(raw, topic)
        source_columns = list(source.columns.difference(["t"]))
        if not source_columns:
            continue
        rename: dict[str, str] = {}
        for column in source_columns:
            rename[column] = column if column not in output else f"{topic}__{column}"
        source_time = f"__source_t_{topic}"
        source = source.rename(columns=rename).rename(columns={"t": source_time})
        tolerance = _tolerance_for(topic, source_columns, tolerances)
        output = pd.merge_asof(
            output.sort_values("t", kind="stable"),
            source.sort_values(source_time, kind="stable"),
            left_on="t",
            right_on=source_time,
            direction="backward",
            tolerance=tolerance,
        )
        age_ms = (output["t"] - output[source_time]) * 1000.0
        for original, attached in rename.items():
            staleness_column = f"{attached}_staleness_ms"
            output[staleness_column] = age_ms.where(output[source_time].notna(), np.inf)
            mask = _stale_mask(output["t"], topic, original, stale)
            if mask.any():
                output.loc[mask, attached] = np.nan
                output.loc[mask, staleness_column] = np.inf
        output = output.drop(columns=[source_time])
    return output.reset_index(drop=True)
