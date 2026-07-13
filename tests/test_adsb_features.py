"""adsb/features.py testleri -- elle hesaplanmis beklenen degerlerle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adsb.features import (
    EARTH_RADIUS_M,
    altitude_source_residual,
    build_feature_table,
    heading_residual,
    speed_residual,
    turn_bank_residual,
    velocity_component_residuals,
    vertical_rate_residual,
)

_M_PER_DEG_LAT = EARTH_RADIUS_M * np.pi / 180.0


def _straight_flight(n=5, dt=10.0, speed_mps=100.0, climb_mps=5.0) -> pd.DataFrame:
    t = np.arange(n) * dt
    dlat_per_step = (speed_mps * dt) / _M_PER_DEG_LAT
    lat = 40.0 + np.arange(n) * dlat_per_step
    lon = np.full(n, 29.0)
    alt = 1000.0 + np.arange(n) * climb_mps * dt
    return pd.DataFrame({
        "flight_id": "F1", "timestamp_utc": t, "lat": lat, "lon": lon, "alt": alt,
        "ground_speed_ms": speed_mps, "track_deg": 0.0, "vertical_rate_ms": climb_mps,
    })


def test_vertical_rate_residual_zero_for_consistent_climb():
    df = _straight_flight()
    res = vertical_rate_residual(df)
    assert np.isnan(res.iloc[0])
    assert res.iloc[1:].abs().max() < 1e-6


def test_speed_residual_zero_for_consistent_flight():
    df = _straight_flight()
    res = speed_residual(df)
    assert res.iloc[1:].abs().max() < 1e-6


def test_velocity_component_residuals_zero_for_consistent_northbound_flight():
    df = _straight_flight()
    residuals = velocity_component_residuals(df)
    assert residuals.iloc[0].isna().all()
    assert residuals.iloc[1:].abs().to_numpy().max() < 1e-6


def test_velocity_component_residual_preserves_signed_north_shift():
    df = _straight_flight(speed_mps=100.0)
    dt = float(df["timestamp_utc"].iloc[1] - df["timestamp_utc"].iloc[0])
    df["lat"] = 40.0 + np.arange(len(df)) * (102.0 * dt / _M_PER_DEG_LAT)
    residuals = velocity_component_residuals(df)
    assert residuals["east_velocity_residual"].iloc[1:].abs().max() < 1e-6
    assert residuals["north_velocity_residual"].iloc[1:].to_numpy() == pytest.approx(-2.0)


def test_velocity_component_residual_skips_nonpositive_dt():
    df = _straight_flight(n=3)
    df["timestamp_utc"] = [0.0, 0.0, -1.0]
    residuals = velocity_component_residuals(df)
    assert residuals.isna().all().all()


def test_velocity_component_residual_is_prefix_invariant():
    df = _straight_flight(n=6)
    full = velocity_component_residuals(df)
    prefix = velocity_component_residuals(df.iloc[:4])
    pd.testing.assert_frame_equal(full.iloc[:4], prefix)


def test_heading_residual_zero_for_northbound_flight():
    df = _straight_flight()
    res = heading_residual(df)
    assert res.iloc[1:].abs().max() < 0.1


def test_turn_bank_residual_zero_for_coordinated_turn():
    n, dt, speed, roll = 5, 10.0, 100.0, 10.0
    g = 9.80665
    rate_deg_s = np.degrees(g * np.tan(np.radians(roll)) / speed)
    track = np.arange(n) * rate_deg_s * dt
    df = pd.DataFrame({
        "flight_id": "F1", "timestamp_utc": np.arange(n) * dt,
        "track_deg": track, "roll_deg": roll, "ground_speed_ms": speed,
    })
    res = turn_bank_residual(df)
    assert res.iloc[1:].abs().max() < 1e-6


def test_build_feature_table_missing_roll_gives_nan_column():
    df = _straight_flight()
    out = build_feature_table(df)
    assert out["turn_bank_residual"].isna().all()


def test_build_feature_table_missing_alt_geom_gives_nan_column():
    df = _straight_flight()
    out = build_feature_table(df)
    assert out["altitude_source_residual"].isna().all()


def test_altitude_source_residual_zero_for_constant_gap():
    df = _straight_flight()
    df["alt_geom_m"] = df["alt"] + 50.0  # sabit ofset -- jeoit sapmasi gibi, ZAMANLA degismiyor
    res = altitude_source_residual(df)
    assert res.iloc[1:].abs().max() < 1e-6


def test_altitude_source_residual_nonzero_when_gap_drifts():
    df = _straight_flight()
    df["alt_geom_m"] = df["alt"] + 50.0
    df.loc[3:, "alt_geom_m"] += 200.0  # bir kaynak aniden sapiyor (sahte/hatali)
    res = altitude_source_residual(df)
    assert abs(res.iloc[3]) > 1.0


def test_build_feature_table_does_not_leak_across_flight_boundary():
    a = _straight_flight(n=3)
    a["alt_geom_m"] = a["alt"] + 50.0
    b = _straight_flight(n=3)
    b["flight_id"] = "F2"
    b["timestamp_utc"] = b["timestamp_utc"] + 10_000
    b["alt_geom_m"] = b["alt"] + 80.0  # farkli ofset -- sinir-otesi diff olsaydi yakalardik
    df = pd.concat([a, b], ignore_index=True)
    out = build_feature_table(df)
    first_of_b = out[out["flight_id"] == "F2"].iloc[0]
    assert np.isnan(first_of_b["vertical_rate_residual"])
    assert np.isnan(first_of_b["speed_residual"])
    assert np.isnan(first_of_b["heading_residual"])
    assert np.isnan(first_of_b["east_velocity_residual"])
    assert np.isnan(first_of_b["north_velocity_residual"])
    assert np.isnan(first_of_b["altitude_source_residual"])
