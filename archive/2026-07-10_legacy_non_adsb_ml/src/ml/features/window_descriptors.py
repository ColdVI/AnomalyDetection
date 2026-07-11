"""Causal, flight-local temporal descriptors for ML-8A.

The v1 descriptor vocabulary is frozen by ``docs/ML8_PLAN.md``.  Windows are
right-aligned and use only rows in ``[t - window_seconds, t]``.  Every flight
is processed independently, so neither rolling state nor forward filling can
cross a ``source_id`` boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


DESCRIPTORS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q10",
    "q25",
    "q75",
    "q90",
    "range",
    "first",
    "last",
    "last_minus_first",
    "linear_slope",
    "diff_mean",
    "diff_std",
    "diff_abs_max",
    "lag1_autocorrelation",
    "missing_fraction",
    "stale_fraction",
)

SOURCE_CHANNEL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "alfa": (
        "xtrack_error",
        "alt_error",
        "gps_speed_residual",
        "roll_error",
        "pitch_error",
        "airspeed_error",
        "path_dev_mag",
    ),
    "uav_sead": (
        "gps_speed_residual",
        "alt_baro_residual",
        "alt_local_residual",
        "attitude_error_mag",
        "control_strain",
        "innovation_check_flags_bit_count",
        "gps_check_fail_flags_bit_count",
        "filter_fault_flags_bit_count",
        "timeout_flags_bit_count",
        "attitude_missing",
        "battery_missing",
        "gps_health_missing",
    ),
}

SCHEMA_VERSION = "descriptor_schema_v1"
WINDOW_SECONDS = 10.0
STRIDE_SECONDS = 1.0
FFILL_LIMIT_SECONDS = 2.0


def available_channels(df: pd.DataFrame, source: str) -> list[str]:
    """Return v1 channels that are actually present for ``source``.

    The source-specific allow-list prevents an absent ALFA/SEAD channel from
    being invented or silently replaced with another feature.
    """

    if source not in SOURCE_CHANNEL_CANDIDATES:
        raise ValueError(f"Unsupported descriptor source: {source!r}")
    return [name for name in SOURCE_CHANNEL_CANDIDATES[source] if name in df.columns]


def build_descriptor_schema(
    source_frames: Mapping[str, pd.DataFrame],
    *,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = STRIDE_SECONDS,
    ffill_limit_seconds: float = FFILL_LIMIT_SECONDS,
) -> dict:
    """Build the deterministic v1 schema from observed DataFrame columns."""

    sources = {
        source: {"channels": available_channels(frame, source)}
        for source, frame in sorted(source_frames.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "causal": True,
        "window_alignment": "right",
        "window_seconds": float(window_seconds),
        "stride_seconds": float(stride_seconds),
        "ffill_limit_seconds": float(ffill_limit_seconds),
        "sources": sources,
        "descriptors": list(DESCRIPTORS),
    }


def schema_bytes(schema: Mapping) -> bytes:
    """Canonical serialized representation used both on disk and for SHA-256."""

    return (json.dumps(schema, indent=2, sort_keys=True) + "\n").encode("utf-8")


def descriptor_schema_sha256(schema_or_path: Mapping | str | Path) -> str:
    """Return the SHA-256 of a schema mapping or its exact on-disk bytes."""

    if isinstance(schema_or_path, Mapping):
        payload = schema_bytes(schema_or_path)
    else:
        payload = Path(schema_or_path).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def write_descriptor_schema(schema: Mapping, path: str | Path) -> str:
    """Write a canonical schema JSON and return its SHA-256."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = schema_bytes(schema)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _limited_forward_fill(
    values: np.ndarray, times: np.ndarray, limit_seconds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causally forward-fill observations and expose missing/stale masks.

    ``missing`` describes the original row. ``stale`` becomes true when no
    observation has ever occurred or the last one is older than the allowed
    carry-forward horizon.  No backward fill or interpolation is performed.
    """

    observed = np.isfinite(values)
    missing = ~observed
    last_observed_at = np.where(observed, times, np.nan)
    last_observed_at = pd.Series(last_observed_at).ffill().to_numpy(dtype=float)
    age = times - last_observed_at
    stale = ~np.isfinite(age) | (age > limit_seconds)
    filled = pd.Series(values).ffill().to_numpy(dtype=float, copy=True)
    filled[stale] = np.nan
    return filled, missing, stale


def _numeric_descriptors(values: np.ndarray, times: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(values)
    x = values[finite]
    tx = times[finite]
    if not len(x):
        return {name: np.nan for name in DESCRIPTORS[:-2]}

    result = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "range": float(np.max(x) - np.min(x)),
        "first": float(x[0]),
        "last": float(x[-1]),
        "last_minus_first": float(x[-1] - x[0]),
    }
    if len(x) >= 2 and np.ptp(tx) > 0.0:
        result["linear_slope"] = float(np.polyfit(tx, x, 1)[0])
    else:
        result["linear_slope"] = np.nan

    diffs = np.diff(x)
    if len(diffs):
        result.update(
            diff_mean=float(np.mean(diffs)),
            diff_std=float(np.std(diffs, ddof=0)),
            diff_abs_max=float(np.max(np.abs(diffs))),
        )
    else:
        result.update(diff_mean=np.nan, diff_std=np.nan, diff_abs_max=np.nan)

    if len(x) >= 2 and np.std(x[:-1]) > 0.0 and np.std(x[1:]) > 0.0:
        result["lag1_autocorrelation"] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    else:
        result["lag1_autocorrelation"] = np.nan
    return result


def _window_endpoints(times: np.ndarray, stride_seconds: float) -> np.ndarray:
    # Endpointleri gözlenen zamanlara bağla. Yoğun bir 1 s grid, uçuş içindeki
    # uzun kayıt boşluklarında veri olmayan hayali pencereler/saatler üretir.
    selected = [float(times[0])]
    for value in times[1:]:
        value = float(value)
        if value - selected[-1] >= stride_seconds - 1e-12:
            selected.append(value)
    return np.asarray(selected, dtype=float)


def describe_flight(
    flight: pd.DataFrame,
    channels: Iterable[str],
    *,
    time_col: str = "t_rel_s",
    source_id_col: str = "source_id",
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = STRIDE_SECONDS,
    ffill_limit_seconds: float = FFILL_LIMIT_SECONDS,
) -> pd.DataFrame:
    """Compute right-aligned descriptors for exactly one flight."""

    if flight.empty:
        return pd.DataFrame()
    if flight[source_id_col].nunique(dropna=False) != 1:
        raise ValueError("describe_flight expects exactly one source_id")
    channels = list(channels)
    missing_channels = [name for name in channels if name not in flight.columns]
    if missing_channels:
        raise KeyError(f"Descriptor channels absent from flight: {missing_channels}")
    if window_seconds <= 0 or stride_seconds <= 0 or ffill_limit_seconds < 0:
        raise ValueError("window/stride must be positive and ffill limit non-negative")

    ordered = flight.sort_values(time_col, kind="mergesort").copy()
    ordered[time_col] = pd.to_numeric(ordered[time_col], errors="coerce")
    ordered = ordered[np.isfinite(ordered[time_col])]
    if ordered.empty:
        return pd.DataFrame()
    times = ordered[time_col].to_numpy(dtype=float)
    source_id = ordered[source_id_col].iloc[0]

    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for channel in channels:
        raw = pd.to_numeric(ordered[channel], errors="coerce").to_numpy(dtype=float, copy=True)
        raw[~np.isfinite(raw)] = np.nan
        prepared[channel] = _limited_forward_fill(raw, times, ffill_limit_seconds)

    rows: list[dict[str, object]] = []
    for endpoint in _window_endpoints(times, stride_seconds):
        in_window = (times >= endpoint - window_seconds) & (times <= endpoint + 1e-12)
        row: dict[str, object] = {
            source_id_col: source_id,
            time_col: float(endpoint),
            "window_start_s": float(max(times[0], endpoint - window_seconds)),
            "window_end_s": float(endpoint),
            "window_row_count": int(np.count_nonzero(in_window)),
        }
        window_times = times[in_window]
        for channel, (filled, missing, stale) in prepared.items():
            stats = _numeric_descriptors(filled[in_window], window_times)
            stats["missing_fraction"] = float(np.mean(missing[in_window]))
            stats["stale_fraction"] = float(np.mean(stale[in_window]))
            for descriptor in DESCRIPTORS:
                row[f"{channel}__{descriptor}"] = stats[descriptor]
        rows.append(row)
    return pd.DataFrame(rows)


def build_window_descriptors(
    df: pd.DataFrame,
    channels: Iterable[str],
    **kwargs,
) -> pd.DataFrame:
    """Compute descriptors independently per flight and concatenate them."""

    source_id_col = kwargs.get("source_id_col", "source_id")
    if source_id_col not in df.columns:
        raise KeyError(source_id_col)
    outputs = [
        describe_flight(group, channels, **kwargs)
        for _, group in df.groupby(source_id_col, sort=False, dropna=False)
    ]
    outputs = [frame for frame in outputs if not frame.empty]
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def interval_overlap_fraction(
    window_start: float,
    window_end: float,
    intervals: Iterable[tuple[float, float]],
) -> float:
    """Fraction of a window covered by the union of anomaly intervals."""

    start = float(window_start)
    end = float(window_end)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return 0.0
    clipped = sorted(
        (max(start, float(a)), min(end, float(b)))
        for a, b in intervals
        if float(b) > start and float(a) < end
    )
    if not clipped:
        return 0.0
    merged: list[list[float]] = []
    for left, right in clipped:
        if right <= left:
            continue
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    overlap = sum(right - left for left, right in merged)
    return float(np.clip(overlap / (end - start), 0.0, 1.0))


def guard_band_label(overlap_fraction: float) -> str:
    """Apply ML-8A's frozen 0% / guard-band / >=50% label policy."""

    fraction = float(overlap_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("overlap_fraction must be in [0, 1]")
    if fraction >= 0.5:
        return "positive"
    if fraction == 0.0:
        return "negative"
    return "guard_band"


def label_windows_from_intervals(
    windows: pd.DataFrame,
    intervals_by_source: Mapping[str, Iterable[tuple[float, float]]],
    *,
    source_id_col: str = "source_id",
    start_col: str = "window_start_s",
    end_col: str = "window_end_s",
    t0_by_source: Mapping[str, float] | None = None,
    interval_unit: str = "seconds",
) -> pd.DataFrame:
    """Attach overlap fractions and guard-band labels to descriptor windows.

    For UAV-SEAD, pass absolute-microsecond ranges, ``t0_by_source`` and
    ``interval_unit='absolute_us'``.  This uses the same
    ``absolute_us = t0 + t_rel_s * 1e6`` reconstruction as ML-6.
    """

    if interval_unit not in {"seconds", "absolute_us"}:
        raise ValueError("interval_unit must be 'seconds' or 'absolute_us'")
    if interval_unit == "absolute_us" and t0_by_source is None:
        raise ValueError("t0_by_source is required for absolute_us ranges")

    out = windows.copy()
    fractions: list[float] = []
    for row in out[[source_id_col, start_col, end_col]].itertuples(index=False, name=None):
        source_id, start, end = row
        intervals = intervals_by_source.get(source_id, ())
        if interval_unit == "absolute_us":
            if source_id not in t0_by_source:  # type: ignore[operator]
                raise KeyError(f"Missing UAV-SEAD t0 for {source_id!r}")
            t0 = float(t0_by_source[source_id])  # type: ignore[index]
            start = t0 + float(start) * 1e6
            end = t0 + float(end) * 1e6
        fractions.append(interval_overlap_fraction(start, end, intervals))
    out["anomaly_overlap_fraction"] = fractions
    out["train_label"] = [guard_band_label(value) for value in fractions]
    return out
