import numpy as np
import pandas as pd

from residual_v1.decision.calibrate import (
    CalibrationConfig,
    InsufficientCalibrationExposure,
    calibrate_channel_threshold,
)
import pytest
from residual_v1.decision.cusum import score_cusum_channel, threshold_crossing_alarms


def test_wrapper_reuses_two_sided_core_for_positive_and_negative_shifts():
    frame = pd.DataFrame(
        {
            "flight_id": ["positive"] * 4 + ["negative"] * 4,
            "t": list(range(4)) * 2,
            "z": [0.0, 3.0, 3.0, 3.0, 0.0, -3.0, -3.0, -3.0],
        }
    )
    scored = score_cusum_channel(frame, channel="c", k=1.0, max_gap_s=2.0)
    assert np.allclose(scored.loc[scored["flight_id"] == "positive", "cusum_score"], [0, 2, 4, 6])
    assert np.allclose(scored.loc[scored["flight_id"] == "negative", "cusum_score"], [0, 2, 4, 6])


def test_alarm_is_a_rising_crossing_not_a_repeated_above_threshold_row():
    frame = pd.DataFrame({"flight_id": "f", "t": np.arange(5.0), "z": [0, 3, 3, 3, 3]})
    scored = score_cusum_channel(frame, channel="c", max_gap_s=2.0)
    alarms = threshold_crossing_alarms(scored, threshold=3.0, refractory_s=60.0)
    assert len(alarms) == 1
    assert alarms.iloc[0]["t_alarm"] == 2.0


def test_bootstrap_calibration_returns_rate_at_or_below_target():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "flight_id": np.repeat(["f1", "f2", "f3"], 121),
            "t": np.tile(np.arange(121.0), 3),
            "z": rng.normal(size=363),
        }
    )
    report = calibrate_channel_threshold(
        frame,
        channel="c",
        target_alarms_per_hour=12.0,
        config=CalibrationConfig(repetitions=20, seed=3),
    )
    assert report["threshold_h"] > 0.0
    assert report["bootstrap_rate_mean"] <= 12.0 + 1e-9
    assert len(report["bootstrap_rates"]) == 20


def test_calibration_refuses_target_below_one_alarm_resolution():
    frame = pd.DataFrame(
        {"flight_id": "f", "t": np.arange(61.0), "z": np.zeros(61)}
    )
    with pytest.raises(InsufficientCalibrationExposure, match="minimum one-alarm exposure"):
        calibrate_channel_threshold(
            frame,
            channel="c",
            target_alarms_per_hour=0.5,
            config=CalibrationConfig(repetitions=5),
        )
