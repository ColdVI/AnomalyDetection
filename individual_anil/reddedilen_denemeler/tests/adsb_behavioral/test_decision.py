from __future__ import annotations

import numpy as np
import pandas as pd

from src.adsb_behavioral.decision import alarm_onsets, alarm_states, calibrate_threshold


def _scores(values, flight_id="f1") -> pd.DataFrame:
    return pd.DataFrame({
        "flight_id": flight_id,
        "timestamp_utc": np.arange(len(values), dtype=float),
        "score": values,
        "quality_good": True,
    })


def test_k_of_n_requires_persistence_and_returns_only_onset():
    frame = _scores([0, 9, 0, 9, 9, 9, 0, 0])
    onsets = alarm_onsets(frame, score_col="score", threshold=5.0, k=2, n=3)
    assert onsets.sum() == 1
    assert bool(onsets.iloc[3]) is True


def test_validation_calibration_can_choose_zero_false_alarm_threshold():
    frame = pd.concat([_scores(np.linspace(0, 1, 100), "a"), _scores(np.linspace(0, 1, 100), "b")])
    calibrated = calibrate_threshold(
        frame,
        score_col="score",
        target_false_events_per_hour=0.0,
    )
    assert calibrated["validation_false_events_per_hour"] == 0.0


def test_alarm_states_keep_binary_anomaly_active_after_onset():
    frame = _scores([0.0, 2.0, 2.0, 2.0])
    states = alarm_states(frame, score_col="score", threshold=1.0, k=1, n=1)
    assert states.tolist() == [False, True, True, True]