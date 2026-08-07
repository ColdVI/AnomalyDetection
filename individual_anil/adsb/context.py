"""Causal normal-context features for ADS-B anomaly calibration.

The context is deliberately *not* an anomaly label.  It describes only the
normal operating regime that is knowable at the current row: a lagged flight
phase, message cadence, ground state, circular track representation, and the
elapsed time since the previous message.

Flight phase uses only rows strictly before the scored row.  This prevents a
vertical-rate anomaly at row ``t`` from routing itself into a more permissive
calibration group at the same row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalContextConfig:
    """Pre-registered numeric choices for causal context construction."""

    phase_history_rows: int
    level_rate_threshold_mps: float
    cadence_edges_s: tuple[float, ...]
    max_gap_s: float
    flight_id_col: str = "flight_id"
    time_col: str = "timestamp_utc"
    on_ground_col: str = "on_ground"
    vertical_rate_col: str = "vertical_rate_ms"
    track_col: str = "track_deg"

    def __post_init__(self) -> None:
        if self.phase_history_rows < 1:
            raise ValueError("phase_history_rows must be >= 1")
        if not np.isfinite(self.level_rate_threshold_mps) or self.level_rate_threshold_mps <= 0:
            raise ValueError("level_rate_threshold_mps must be finite and > 0")
        edges = np.asarray(self.cadence_edges_s, dtype=float)
        if len(edges) == 0 or not np.isfinite(edges).all() or np.any(edges <= 0):
            raise ValueError("cadence_edges_s must contain finite positive values")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("cadence_edges_s must be strictly increasing")
        if not np.isfinite(self.max_gap_s) or self.max_gap_s <= edges[-1]:
            raise ValueError("max_gap_s must be finite and exceed the last cadence edge")


def _assert_required_and_ordered(frame: pd.DataFrame, config: CausalContextConfig) -> None:
    required = {
        config.flight_id_col,
        config.time_col,
        config.on_ground_col,
        config.vertical_rate_col,
        config.track_col,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing causal-context columns: {sorted(missing)}")
    if frame[config.flight_id_col].isna().any():
        raise ValueError("flight_id cannot be missing in causal context")

    for flight_id, group in frame.groupby(config.flight_id_col, sort=False):
        times = pd.to_numeric(group[config.time_col], errors="coerce").to_numpy(float)
        if not np.isfinite(times).all():
            raise ValueError(f"{flight_id}: timestamps must be finite")
        if len(times) > 1 and np.any(np.diff(times) < 0):
            raise ValueError(f"{flight_id}: rows must be sorted by timestamp_utc")


def _lagged_phase(frame: pd.DataFrame, config: CausalContextConfig) -> pd.Series:
    flight = frame[config.flight_id_col]
    vertical_rate = pd.to_numeric(frame[config.vertical_rate_col], errors="coerce")
    lagged = vertical_rate.groupby(flight, sort=False).shift(1)
    history = (
        lagged.groupby(flight, sort=False)
        .rolling(config.phase_history_rows, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
        .reindex(frame.index)
    )

    ground = frame[config.on_ground_col].astype("boolean")
    phase = pd.Series("unknown", index=frame.index, dtype="string")
    phase.loc[ground.eq(True).fillna(False)] = "ground"
    airborne = ground.eq(False).fillna(False)
    known = airborne & history.notna()
    phase.loc[known & history.gt(config.level_rate_threshold_mps)] = "climb"
    phase.loc[known & history.lt(-config.level_rate_threshold_mps)] = "descent"
    phase.loc[known & history.abs().le(config.level_rate_threshold_mps)] = "level"
    return phase


def _cadence_bucket(dt_s: pd.Series, config: CausalContextConfig) -> pd.Series:
    result = pd.Series("initial_or_invalid", index=dt_s.index, dtype="string")
    positive = dt_s.gt(0) & dt_s.le(config.max_gap_s)
    edges = np.asarray(config.cadence_edges_s, dtype=float)
    bucket_number = np.searchsorted(edges, dt_s.to_numpy(float), side="left")
    for number in range(len(edges) + 1):
        result.loc[positive & (bucket_number == number)] = f"cadence_{number}"
    result.loc[dt_s.gt(config.max_gap_s).fillna(False)] = "gap"
    return result


def build_causal_context(frame: pd.DataFrame, config: CausalContextConfig) -> pd.DataFrame:
    """Return causal context columns aligned one-for-one with ``frame``.

    No sorting is performed silently.  A caller must supply flight-contiguous,
    timestamp-ordered data so accidental future leakage fails closed.
    """

    _assert_required_and_ordered(frame, config)
    flight = frame[config.flight_id_col]
    time = pd.to_numeric(frame[config.time_col], errors="coerce")
    dt_s = time.groupby(flight, sort=False).diff()

    track = pd.to_numeric(frame[config.track_col], errors="coerce")
    valid_track = track.ge(0) & track.lt(360)
    radians = np.deg2rad(track.where(valid_track))

    context = pd.DataFrame(index=frame.index)
    context["context_dt_s"] = dt_s.astype(float)
    context["context_log1p_dt_s"] = np.log1p(dt_s.where(dt_s.ge(0)))
    context["context_phase"] = _lagged_phase(frame, config)
    context["context_cadence"] = _cadence_bucket(dt_s, config)
    context["track_sin"] = np.sin(radians)
    context["track_cos"] = np.cos(radians)
    context["context_evaluable"] = (
        context["context_phase"].ne("unknown")
        & context["context_cadence"].str.startswith("cadence_")
    )
    return context
