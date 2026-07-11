"""Causal physics-consistency residuals for ADS-B trajectory rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_008.8
MAX_PHYSICS_DT_S = 120.0

MODEL_FEATURES = [
    "abs_speed_residual_mps",
    "abs_track_residual_deg",
    "abs_vrate_residual_mps",
    "abs_horizontal_accel_mps2",
    "abs_horizontal_jerk_mps3",
    "abs_track_rate_dps",
    "abs_turn_residual_dps",
    "abs_baro_geom_delta_rate_mps",
    "duplicate_position_moving",
]


def circular_difference_deg(a, b) -> np.ndarray:
    """Smallest signed angular difference a-b in [-180, 180)."""

    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


def _distance_and_bearing(lat1, lon1, lat2, lon2) -> tuple[np.ndarray, np.ndarray]:
    lat1r = np.radians(np.asarray(lat1, dtype=float))
    lon1r = np.radians(np.asarray(lon1, dtype=float))
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    lon2r = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    distance = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return distance, bearing


def _phase(frame: pd.DataFrame) -> pd.Series:
    on_ground = frame["on_ground"].fillna(False).astype(bool)
    speed = frame["ground_speed_ms"].fillna(0.0)
    vrate = frame["vertical_rate_ms"].fillna(0.0)
    phase = np.full(len(frame), "cruise", dtype=object)
    phase[vrate.to_numpy() > 1.5] = "climb"
    phase[vrate.to_numpy() < -1.5] = "descent"
    phase[(on_ground | (speed < 15.0)).to_numpy()] = "ground"
    return pd.Series(phase, index=frame.index)


def _one_flight(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("timestamp_utc").copy()
    numeric_columns = [
        "timestamp_utc", "lat", "lon", "alt", "alt_geom_m", "ground_speed_ms",
        "track_deg", "vertical_rate_ms", "roll_deg",
    ]
    for column in numeric_columns:
        group[column] = pd.to_numeric(group[column], errors="coerce")
    previous = group.shift(1)
    dt = group["timestamp_utc"] - previous["timestamp_utc"]
    valid_dt = dt.gt(0.0) & dt.le(MAX_PHYSICS_DT_S)
    distance, bearing = _distance_and_bearing(
        previous["lat"], previous["lon"], group["lat"], group["lon"]
    )
    distance = pd.Series(distance, index=group.index).where(valid_dt)
    bearing = pd.Series(bearing, index=group.index).where(valid_dt)

    position_speed = (distance / dt).where(valid_dt)
    speed_residual = position_speed - group["ground_speed_ms"]
    track_residual = pd.Series(
        circular_difference_deg(bearing, group["track_deg"]), index=group.index
    ).where(valid_dt & group["track_deg"].notna())

    derived_vrate = ((group["alt"] - previous["alt"]) / dt).where(valid_dt)
    vrate_residual = derived_vrate - group["vertical_rate_ms"]
    acceleration = (position_speed.diff() / dt).where(valid_dt)
    jerk = (acceleration.diff() / dt).where(valid_dt)
    track_rate = pd.Series(
        circular_difference_deg(group["track_deg"], previous["track_deg"]), index=group.index
    ).div(dt).where(valid_dt)

    speed_for_turn = group["ground_speed_ms"].where(group["ground_speed_ms"].abs() >= 10.0)
    expected_turn_rate = np.degrees(
        9.80665 * np.tan(np.radians(group["roll_deg"])) / speed_for_turn
    )
    turn_residual = (track_rate - expected_turn_rate).where(group["roll_deg"].notna())

    baro_geom_delta = group["alt"] - group["alt_geom_m"]
    baro_geom_delta_rate = (baro_geom_delta.diff() / dt).where(valid_dt)
    duplicate_position = (
        group["lat"].eq(previous["lat"])
        & group["lon"].eq(previous["lon"])
        & group["ground_speed_ms"].fillna(0.0).gt(15.0)
        & valid_dt
    )

    group["dt_s"] = dt
    group["distance_m"] = distance
    group["position_speed_mps"] = position_speed
    group["bearing_deg"] = bearing
    group["speed_residual_mps"] = speed_residual
    group["track_residual_deg"] = track_residual
    group["derived_vrate_mps"] = derived_vrate
    group["vrate_residual_mps"] = vrate_residual
    group["horizontal_accel_mps2"] = acceleration
    group["horizontal_jerk_mps3"] = jerk
    group["track_rate_dps"] = track_rate
    group["expected_turn_rate_dps"] = expected_turn_rate
    group["turn_residual_dps"] = turn_residual
    group["baro_geom_delta_m"] = baro_geom_delta
    group["baro_geom_delta_rate_mps"] = baro_geom_delta_rate
    group["duplicate_position_moving"] = duplicate_position.astype(float)
    group["phase"] = _phase(group)
    group["is_turn"] = track_rate.abs().gt(1.5)
    group["quality_good"] = (
        valid_dt
        & ~group["flags_stale"].fillna(False).astype(bool)
        & group["ads_source_type"].eq("adsb_icao")
        & group[["lat", "lon", "ground_speed_ms", "track_deg"]].notna().all(axis=1)
    )

    signed_to_absolute = {
        "speed_residual_mps": "abs_speed_residual_mps",
        "track_residual_deg": "abs_track_residual_deg",
        "vrate_residual_mps": "abs_vrate_residual_mps",
        "horizontal_accel_mps2": "abs_horizontal_accel_mps2",
        "horizontal_jerk_mps3": "abs_horizontal_jerk_mps3",
        "track_rate_dps": "abs_track_rate_dps",
        "turn_residual_dps": "abs_turn_residual_dps",
        "baro_geom_delta_rate_mps": "abs_baro_geom_delta_rate_mps",
    }
    for source, target in signed_to_absolute.items():
        group[target] = group[source].abs()
    return group


def add_physics_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    """Add causal residuals independently within each flight."""

    if frame.empty:
        return frame.copy()
    required = {
        "flight_id", "timestamp_utc", "lat", "lon", "alt", "alt_geom_m",
        "ground_speed_ms", "track_deg", "vertical_rate_ms", "roll_deg",
        "on_ground", "flags_stale", "ads_source_type",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing ADS-B columns: {sorted(missing)}")
    parts = [_one_flight(group) for _, group in frame.groupby("flight_id", sort=False)]
    return pd.concat(parts, ignore_index=True)
