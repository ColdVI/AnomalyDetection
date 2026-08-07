from __future__ import annotations

import numpy as np
import pandas as pd

from src.adsb_behavioral.physics_residuals import add_physics_residuals, circular_difference_deg


def _straight_flight(rows: int = 20) -> pd.DataFrame:
    dt = 10.0
    speed = 100.0
    lon_step = speed * dt / 111_320.0
    return pd.DataFrame({
        "flight_id": ["f1"] * rows,
        "timestamp_utc": np.arange(rows) * dt,
        "lat": np.zeros(rows),
        "lon": np.arange(rows) * lon_step,
        "alt": 1_000.0 + np.arange(rows) * 5.0 * dt,
        "alt_geom_m": 1_020.0 + np.arange(rows) * 5.0 * dt,
        "on_ground": False,
        "ground_speed_ms": speed,
        "track_deg": 90.0,
        "vertical_rate_ms": 5.0,
        "roll_deg": np.nan,
        "flags_stale": False,
        "ads_source_type": "adsb_icao",
    })


def test_straight_constant_flight_has_small_consistency_residuals():
    result = add_physics_residuals(_straight_flight())
    valid = result[result["quality_good"]]
    assert valid["speed_residual_mps"].abs().median() < 0.2
    assert valid["track_residual_deg"].abs().median() < 0.1
    assert valid["vrate_residual_mps"].abs().median() < 1e-9


def test_residuals_are_prefix_invariant():
    full = add_physics_residuals(_straight_flight(30))
    prefix = add_physics_residuals(_straight_flight(18))
    columns = ["position_speed_mps", "speed_residual_mps", "derived_vrate_mps", "track_rate_dps"]
    for column in columns:
        np.testing.assert_allclose(full[column].iloc[:18], prefix[column], equal_nan=True)


def test_circular_difference_wraps_at_360():
    result = circular_difference_deg([1.0, 359.0], [359.0, 1.0])
    np.testing.assert_allclose(result, [2.0, -2.0])
