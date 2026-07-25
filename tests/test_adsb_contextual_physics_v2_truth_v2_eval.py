"""Focused contracts for the contextual_physics_v2 truth-v2 evaluator."""

from __future__ import annotations

import pandas as pd

from scripts.adsb_contextual_physics_v2_truth_v2_eval import (
    _persistence_detector,
    _with_exposure_bounds,
)


def test_exposure_bounds_reset_at_channel_boundary() -> None:
    frame = pd.DataFrame(
        {
            "channel": ["a", "a", "b", "b"],
            "flight_id": ["f", "f", "f", "f"],
            "timestamp_utc": [10.0, 12.0, 10.0, 12.0],
        }
    )

    bounded = _with_exposure_bounds(frame, "timestamp_utc")

    assert bounded["t_start"].tolist() == [10.0, 10.0, 10.0, 10.0]
    assert bounded["t_end"].tolist() == [10.0, 12.0, 10.0, 12.0]


def test_persistence_detector_uses_frozen_calibration_multiplier() -> None:
    detector = _persistence_detector(
        {"persistence_v2": {"reference_shift_multiplier": 1.37}}
    )

    assert detector.config.reference_shift_multiplier == 1.37
    assert detector.config.max_gap_s == 30.0
    assert detector.config.missing_reset_s == 60.0
    assert detector.config.surprise_clip == 6.0
