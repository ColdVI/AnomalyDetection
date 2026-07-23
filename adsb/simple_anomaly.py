"""Small, preregistered ADS-B phase and anomaly rules.

The module intentionally contains no learned model and no threshold search.  The
constants mirror ``docs/ADSB_BASIT_ANOMALI_ONKAYIT_20260722.md`` and must not be
changed in response to discovery trigger counts.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


TAKEOFF_VERTICAL_RATE_MPS = 2.5
CRUISE_ABS_VERTICAL_RATE_MPS = 1.0
LANDING_VERTICAL_RATE_MPS = -2.5
PHASE_SUSTAINED_SAMPLES = 4
MAX_CONTIGUOUS_GAP_S = 30.0
MIN_PHASE_ALTITUDE_RANGE_M = 300.0
CRUISE_RELATIVE_ALTITUDE_FRACTION = 0.60

ALTITUDE_DEVIATION_M = 150.0
ALTITUDE_MIN_DURATION_S = 120.0
ALTITUDE_SOURCE_RESIDUAL_LARGE_MPS = 5.0

ROUTE_DEVIATION_DEG = 20.0
ROUTE_CONSECUTIVE_SAMPLES = 4
ROUTE_LOW_SPEED_MPS = 30.0

PHASE_LABELS = ("takeoff", "cruise", "landing", "uncertain")

ALTITUDE_EVENT_COLUMNS = [
    "event_id", "flight_id", "start_time", "end_time", "duration_s",
    "n_samples", "direction", "cruise_median_alt_m",
    "peak_abs_deviation_m", "median_abs_deviation_m",
    "data_quality_suspect", "altitude_source_residual_abs_max_mps",
]

ROUTE_EVENT_COLUMNS = [
    "event_id", "flight_id", "start_time", "end_time", "duration_s",
    "n_samples", "phase", "peak_abs_heading_residual_deg",
    "median_abs_heading_residual_deg", "low_speed_context",
    "low_speed_fraction", "vector_residual_magnitude_max_mps",
]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _runs(
    mask: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_gap_s: float,
    keys: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Return inclusive positional runs, breaking on gaps or optional key changes."""
    result: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for position, active in enumerate(mask.astype(bool)):
        continuous = False
        if active and previous is not None:
            delta = timestamps[position] - timestamps[previous]
            continuous = bool(np.isfinite(delta) and 0.0 < delta <= max_gap_s)
            if keys is not None:
                continuous = continuous and bool(keys[position] == keys[previous])
        if active and start is None:
            start = position
        elif active and not continuous:
            if start is not None and previous is not None:
                result.append((start, previous))
            start = position
        elif not active and start is not None:
            if previous is not None:
                result.append((start, previous))
            start = None
        previous = position if active else None
    if start is not None and previous is not None:
        result.append((start, previous))
    return result


def _sustained_runs(
    mask: np.ndarray,
    timestamps: np.ndarray,
    *,
    minimum_samples: int,
) -> list[tuple[int, int]]:
    return [
        run for run in _runs(mask, timestamps, max_gap_s=MAX_CONTIGUOUS_GAP_S)
        if run[1] - run[0] + 1 >= minimum_samples
    ]


def flight_phase(
    frame: pd.DataFrame,
    *,
    flight_id_col: str = "flight_id",
    time_col: str = "timestamp_utc",
    altitude_col: str = "alt",
    vertical_rate_col: str = "vertical_rate_ms",
) -> pd.Series:
    """Assign strict three-phase labels, leaving incomplete traces uncertain.

    A flight is resolved only when sustained climb, high-level cruise, and
    sustained descent evidence occur in that order.  Output follows the input
    index even when rows arrive unsorted.
    """
    _require_columns(
        frame, (flight_id_col, time_col, altitude_col, vertical_rate_col),
    )
    result = pd.Series("uncertain", index=frame.index, dtype="object", name="flight_phase")
    for _, group in frame.groupby(flight_id_col, sort=False, dropna=False):
        ordered = group.sort_values(time_col, kind="mergesort")
        times = pd.to_numeric(ordered[time_col], errors="coerce").to_numpy(dtype=float)
        altitude = pd.to_numeric(ordered[altitude_col], errors="coerce").to_numpy(dtype=float)
        vertical_rate = pd.to_numeric(
            ordered[vertical_rate_col], errors="coerce"
        ).to_numpy(dtype=float)
        finite_altitude = altitude[np.isfinite(altitude)]
        if len(finite_altitude) < PHASE_SUSTAINED_SAMPLES:
            continue
        low, high = np.quantile(finite_altitude, [0.05, 0.95])
        altitude_range = float(high - low)
        if not np.isfinite(altitude_range) or altitude_range < MIN_PHASE_ALTITUDE_RANGE_M:
            continue

        climb_runs = _sustained_runs(
            np.isfinite(vertical_rate) & (vertical_rate > TAKEOFF_VERTICAL_RATE_MPS),
            times,
            minimum_samples=PHASE_SUSTAINED_SAMPLES,
        )
        level_runs = _sustained_runs(
            np.isfinite(vertical_rate)
            & (np.abs(vertical_rate) < CRUISE_ABS_VERTICAL_RATE_MPS),
            times,
            minimum_samples=PHASE_SUSTAINED_SAMPLES,
        )
        descent_runs = _sustained_runs(
            np.isfinite(vertical_rate) & (vertical_rate < LANDING_VERTICAL_RATE_MPS),
            times,
            minimum_samples=PHASE_SUSTAINED_SAMPLES,
        )
        if not climb_runs or not level_runs or not descent_runs:
            continue

        candidates: list[tuple[float, int, int, int]] = []
        for level_start, level_end in level_runs:
            level_altitude = altitude[level_start:level_end + 1]
            finite_level_altitude = level_altitude[np.isfinite(level_altitude)]
            if len(finite_level_altitude) == 0:
                continue
            level_median = float(np.median(finite_level_altitude))
            relative_altitude = (level_median - low) / altitude_range
            if not np.isfinite(relative_altitude):
                continue
            prior_climbs = [run for run in climb_runs if run[1] < level_start]
            later_descents = [run for run in descent_runs if run[0] > level_end]
            if (
                relative_altitude >= CRUISE_RELATIVE_ALTITUDE_FRACTION
                and prior_climbs
                and later_descents
            ):
                descent_start = min(run[0] for run in later_descents)
                candidates.append((relative_altitude, level_start, level_end, descent_start))
        if not candidates:
            continue

        _, cruise_start, _, landing_start = max(
            candidates,
            key=lambda item: (item[0], item[2] - item[1] + 1, -item[1]),
        )
        ordered_labels = np.full(len(ordered), "uncertain", dtype=object)
        ordered_labels[:cruise_start] = "takeoff"
        ordered_labels[cruise_start:landing_start] = "cruise"
        ordered_labels[landing_start:] = "landing"
        result.loc[ordered.index] = ordered_labels
    return result


def detect_altitude_deviation_events(
    frame: pd.DataFrame,
    *,
    flight_id_col: str = "flight_id",
    time_col: str = "timestamp_utc",
    altitude_col: str = "alt",
    phase_col: str = "flight_phase",
    source_residual_col: str = "altitude_source_residual",
) -> pd.DataFrame:
    """Find preregistered cruise-altitude deviation events."""
    _require_columns(frame, (flight_id_col, time_col, altitude_col, phase_col))
    rows: list[dict] = []
    for flight_id, group in frame.groupby(flight_id_col, sort=False, dropna=False):
        cruise = group.loc[group[phase_col].eq("cruise")].sort_values(
            time_col, kind="mergesort"
        )
        altitude = pd.to_numeric(cruise[altitude_col], errors="coerce")
        if altitude.notna().sum() == 0:
            continue
        reference = float(altitude.median())
        deviation = altitude.to_numpy(dtype=float) - reference
        times = pd.to_numeric(cruise[time_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(deviation) & (np.abs(deviation) >= ALTITUDE_DEVIATION_M)
        direction = np.where(deviation >= 0.0, "above", "below")
        event_number = 0
        for start, end in _runs(
            mask, times, max_gap_s=MAX_CONTIGUOUS_GAP_S, keys=direction,
        ):
            duration = float(times[end] - times[start])
            if not np.isfinite(duration) or duration <= ALTITUDE_MIN_DURATION_S:
                continue
            event_number += 1
            event_slice = cruise.iloc[start:end + 1]
            event_deviation = np.abs(deviation[start:end + 1])
            if source_residual_col in event_slice:
                source_residual = np.abs(
                    pd.to_numeric(event_slice[source_residual_col], errors="coerce")
                    .to_numpy(dtype=float)
                )
                finite_source = source_residual[np.isfinite(source_residual)]
            else:
                finite_source = np.array([], dtype=float)
            source_max = float(finite_source.max()) if len(finite_source) else None
            rows.append({
                "event_id": f"{flight_id}_alt_{event_number:03d}",
                "flight_id": flight_id,
                "start_time": float(times[start]),
                "end_time": float(times[end]),
                "duration_s": duration,
                "n_samples": int(end - start + 1),
                "direction": str(direction[start]),
                "cruise_median_alt_m": reference,
                "peak_abs_deviation_m": float(np.nanmax(event_deviation)),
                "median_abs_deviation_m": float(np.nanmedian(event_deviation)),
                "data_quality_suspect": bool(
                    source_max is not None
                    and source_max >= ALTITUDE_SOURCE_RESIDUAL_LARGE_MPS
                ),
                "altitude_source_residual_abs_max_mps": source_max,
            })
    return pd.DataFrame(rows, columns=ALTITUDE_EVENT_COLUMNS)


def detect_route_deviation_events(
    frame: pd.DataFrame,
    *,
    flight_id_col: str = "flight_id",
    time_col: str = "timestamp_utc",
    phase_col: str = "flight_phase",
    residual_col: str = "heading_residual",
    speed_col: str = "ground_speed_ms",
) -> pd.DataFrame:
    """Find preregistered four-sample heading-residual events."""
    _require_columns(frame, (flight_id_col, time_col, phase_col, residual_col, speed_col))
    rows: list[dict] = []
    for flight_id, group in frame.groupby(flight_id_col, sort=False, dropna=False):
        ordered = group.sort_values(time_col, kind="mergesort")
        times = pd.to_numeric(ordered[time_col], errors="coerce").to_numpy(dtype=float)
        residual = pd.to_numeric(ordered[residual_col], errors="coerce").to_numpy(dtype=float)
        speed = pd.to_numeric(ordered[speed_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(residual) & (np.abs(residual) >= ROUTE_DEVIATION_DEG)
        event_number = 0
        for start, end in _runs(mask, times, max_gap_s=MAX_CONTIGUOUS_GAP_S):
            n_samples = end - start + 1
            if n_samples < ROUTE_CONSECUTIVE_SAMPLES:
                continue
            event_number += 1
            event_slice = ordered.iloc[start:end + 1]
            event_residual = np.abs(residual[start:end + 1])
            event_speed = speed[start:end + 1]
            finite_speed = event_speed[np.isfinite(event_speed)]
            low_speed = finite_speed < ROUTE_LOW_SPEED_MPS
            phases = event_slice[phase_col].dropna().astype(str)
            phase = phases.mode().iloc[0] if len(phases) else "uncertain"
            if phases.nunique() > 1:
                phase = "mixed"
            vector_columns = {
                "east_velocity_residual", "north_velocity_residual",
            }
            if vector_columns.issubset(event_slice.columns):
                east = pd.to_numeric(
                    event_slice["east_velocity_residual"], errors="coerce"
                ).to_numpy(dtype=float)
                north = pd.to_numeric(
                    event_slice["north_velocity_residual"], errors="coerce"
                ).to_numpy(dtype=float)
                magnitude = np.hypot(east, north)
                finite_magnitude = magnitude[np.isfinite(magnitude)]
            else:
                finite_magnitude = np.array([], dtype=float)
            rows.append({
                "event_id": f"{flight_id}_route_{event_number:03d}",
                "flight_id": flight_id,
                "start_time": float(times[start]),
                "end_time": float(times[end]),
                "duration_s": float(times[end] - times[start]),
                "n_samples": int(n_samples),
                "phase": phase,
                "peak_abs_heading_residual_deg": float(np.nanmax(event_residual)),
                "median_abs_heading_residual_deg": float(np.nanmedian(event_residual)),
                "low_speed_context": bool(low_speed.any()) if len(low_speed) else False,
                "low_speed_fraction": float(low_speed.mean()) if len(low_speed) else None,
                "vector_residual_magnitude_max_mps": (
                    float(finite_magnitude.max()) if len(finite_magnitude) else None
                ),
            })
    return pd.DataFrame(rows, columns=ROUTE_EVENT_COLUMNS)
