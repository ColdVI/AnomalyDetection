"""Residual feature-matrix construction on an observed reference clock."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from residual_v1.features.phases import PHASES
from residual_v1.features.physics import (
    coordinated_turn_yaw_rate,
    finite_difference,
    motor_pwm_summary,
    vector_norm,
)
from residual_v1.features.spec import ResidualChannelSpec
from residual_v1.features.waypoints import label_waypoint_boundaries

logger = logging.getLogger(__name__)
_LAG_WINDOWS = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0))


def _triangular_average(age: np.ndarray, values: np.ndarray, low: float, high: float) -> float:
    selected = (age >= low) & (age < high) & np.isfinite(values)
    if low == 0.0:
        selected |= (age == high) & np.isfinite(values)
    if not selected.any():
        return float("nan")
    selected_age = age[selected]
    width = max(high - low, np.finfo(float).eps)
    weights = 1.0 + (high - selected_age) / width
    return float(np.average(values[selected], weights=weights))


def lag_summary(time: pd.Series, values: pd.Series, *, name: str) -> pd.DataFrame:
    """Last value, three causal triangular windows, and a one-second delta."""

    t = pd.to_numeric(time, errors="coerce").to_numpy(float)
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("time must be finite and strictly increasing")
    output = np.full((len(t), 5), np.nan, dtype=float)
    for index, current in enumerate(t):
        start = int(np.searchsorted(t, current - 1.0, side="left"))
        history_t = t[start : index + 1]
        history_x = x[start : index + 1]
        age = current - history_t
        output[index, 0] = x[index]
        for offset, (low, high) in enumerate(_LAG_WINDOWS, start=1):
            output[index, offset] = _triangular_average(age, history_x, low, high)
        output[index, 4] = output[index, 0] - output[index, 3]
    return pd.DataFrame(
        output,
        index=values.index,
        columns=(
            f"{name}__last",
            f"{name}__tri_0_025",
            f"{name}__tri_025_05",
            f"{name}__tri_05_10",
            f"{name}__delta_1s",
        ),
    )


def future_mean(
    time: pd.Series,
    values: pd.Series,
    *,
    horizon_s: float,
) -> pd.Series:
    t = pd.to_numeric(time, errors="coerce").to_numpy(float)
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    result = np.full(len(t), np.nan, dtype=float)
    for index, current in enumerate(t):
        end = int(np.searchsorted(t, current + horizon_s, side="right"))
        future = x[index + 1 : end]
        finite = future[np.isfinite(future)]
        if len(finite):
            result[index] = float(finite.mean())
    return pd.Series(result, index=values.index, name=values.name)


def augment_physics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    aliases = {
        "airspeed": "airspeed_context",
        "roll": "roll_context",
        "pitch": "pitch_context",
    }
    for source, destination in aliases.items():
        if source in result and destination not in result:
            result[destination] = result[source]
    if "airspeed" in result and "airspeed_derivative" not in result:
        result["airspeed_derivative"] = finite_difference(
            result["t"], result["airspeed"], window_s=0.5
        )
    if {"roll_context", "airspeed_context"}.issubset(result):
        result["coordinated_turn_term"] = coordinated_turn_yaw_rate(
            result["roll_context"], result["airspeed_context"]
        )
    rate_columns = ("roll_rate", "pitch_rate", "yaw_rate")
    if set(rate_columns).issubset(result):
        result["attitude_rate_vector_norm"] = vector_norm(
            result, rate_columns, name="attitude_rate_vector_norm"
        )
    motor_columns = {f"motor_pwm_{index}" for index in range(4)}
    if motor_columns.issubset(result):
        summaries = motor_pwm_summary(result)
        result[summaries.columns] = summaries
    if "vertical_acceleration" not in result:
        if "local_az" in result:
            result["vertical_acceleration"] = -pd.to_numeric(result["local_az"], errors="coerce")
        elif "accel_z" in result:
            result["vertical_acceleration"] = -pd.to_numeric(result["accel_z"], errors="coerce")
    velocity_columns = ("local_vx", "local_vy", "local_vz")
    if set(velocity_columns).issubset(result):
        result["velocity_response_norm"] = vector_norm(
            result, velocity_columns, name="velocity_response_norm"
        )
    return result


def _context_speed(frame: pd.DataFrame) -> pd.Series:
    if "airspeed_context" in frame:
        return pd.to_numeric(frame["airspeed_context"], errors="coerce")
    if {"local_vx", "local_vy"}.issubset(frame):
        return np.sqrt(
            pd.to_numeric(frame["local_vx"], errors="coerce") ** 2
            + pd.to_numeric(frame["local_vy"], errors="coerce") ** 2
        )
    return pd.Series(1.0, index=frame.index, dtype=float)


def build_xy(
    flight_aligned: pd.DataFrame,
    spec: ResidualChannelSpec,
    phases: pd.DataFrame,
    *,
    events: Sequence[Mapping[str, object]] = (),
    flight_id: str = "unknown",
    boundary_configs: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Build one channel's X/y while retaining an explicit training mask."""

    if len(flight_aligned) != len(phases):
        raise ValueError("aligned flight and phases must have equal length")
    if "t" not in flight_aligned or "phase" not in phases or "phase_boundary" not in phases:
        raise ValueError("missing time or phase columns")
    frame = augment_physics(flight_aligned.reset_index(drop=True))
    phase_frame = phases.reset_index(drop=True)
    if not np.allclose(frame["t"], phase_frame["t"], equal_nan=False):
        raise ValueError("phase clock does not match aligned flight")
    required = {*spec.all_inputs, spec.response}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"{spec.name}: missing columns {missing}")

    feature_parts = [
        lag_summary(frame["t"], frame[column], name=column)
        for column in spec.command_inputs
    ]
    context_parts = [
        pd.to_numeric(frame[column], errors="coerce").rename(f"{column}__last").to_frame()
        for column in spec.context_inputs
    ]
    speed = _context_speed(frame).rename("context_speed")
    common = pd.DataFrame(index=frame.index)
    if spec.command_inputs:
        common["context_speed"] = speed
        common["context_speed_sq"] = speed**2
    for phase in PHASES:
        common[f"phase_{phase}"] = (phase_frame["phase"] == phase).astype(float)
    for column in spec.command_inputs:
        if column.endswith("_context") or column in {"battery_voltage", "coordinated_turn_term"}:
            continue
        if feature_parts:
            common[f"{column}__speed_interaction"] = speed * feature_parts[
                list(spec.command_inputs).index(column)
            ][f"{column}__last"]
    X_all = pd.concat([*feature_parts, *context_parts, common], axis=1)
    y_all = future_mean(frame["t"], frame[spec.response], horizon_s=spec.horizon_s)
    y_all.name = spec.response

    candidate = (phase_frame["phase"] != "ground") & ~phase_frame["phase_boundary"].astype(bool)
    boundary_reports: dict[str, dict[str, object]] = {}
    for boundary_name in spec.boundary_masks:
        if boundary_name != "waypoint":
            raise RuntimeError(f"unsupported boundary mask reached build_xy: {boundary_name}")
        config = boundary_configs.get(boundary_name) if boundary_configs else None
        waypoint = label_waypoint_boundaries(frame, config=config)
        boundary = waypoint["waypoint_boundary"].astype(bool)
        candidate &= ~boundary
        boundary_reports[boundary_name] = {
            "event_count": int(waypoint["waypoint_event"].sum()),
            "masked_row_count": int(boundary.sum()),
            "event_times_s": frame.loc[waypoint["waypoint_event"], "t"].astype(float).tolist(),
        }
    finite = X_all.notna().all(axis=1) & y_all.notna()
    keep = candidate & finite
    candidate_count = int(candidate.sum())
    nan_drop_count = int((candidate & ~finite).sum())
    nan_drop_ratio = nan_drop_count / candidate_count if candidate_count else 0.0
    if nan_drop_ratio > 0.20:
        logger.warning(
            "%s/%s drops %.1f%% candidate rows for missing features",
            flight_id,
            spec.name,
            100.0 * nan_drop_ratio,
        )

    onset = min((float(event["onset_s"]) for event in events), default=float("inf"))
    retained_time = pd.to_numeric(frame.loc[keep, "t"], errors="coerce")
    row_meta = pd.DataFrame(
        {
            "flight_id": flight_id,
            "t": retained_time.to_numpy(float),
            "phase": phase_frame.loc[keep, "phase"].to_numpy(),
            "train_eligible": (retained_time < onset - 10.0).to_numpy(bool),
        },
        index=X_all.index[keep],
    )
    X = X_all.loc[keep].copy()
    y = y_all.loc[keep].copy()
    forbidden = [column for column in X if column == spec.response or column.startswith(f"{spec.response}__")]
    if forbidden:
        raise RuntimeError(f"response leakage reached feature matrix: {forbidden}")
    meta = {
        "flight_id": flight_id,
        "channel": spec.name,
        "input_rows": int(len(frame)),
        "candidate_rows": candidate_count,
        "output_rows": int(len(X)),
        "nan_drop_count": nan_drop_count,
        "nan_drop_ratio": nan_drop_ratio,
        "boundary_masks": boundary_reports,
        "row_meta": row_meta,
    }
    return X, y, meta
