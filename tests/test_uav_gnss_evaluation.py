import numpy as np
import pandas as pd
import pytest

from uav_gnss.evaluation import deadline_event_metrics, natural_burden, wilson_interval


def test_episode_burden_uses_scoreable_flight_time():
    frame = pd.DataFrame(
        {
            "flight_id": ["A"] * 5,
            "timestamp_s": [0.0, 1.0, 2.0, 20.0, 21.0],
            "dt_s": [0.0, 1.0, 1.0, 18.0, 1.0],
            "landed": [0, 0, 0, 0, 0],
            "evaluable": [False, True, True, False, True],
        }
    )
    burden = natural_burden(
        frame,
        np.array([False, True, True, False, True]),
        merge_gap_s=10.0,
    )
    assert burden["n_alert_episodes"] == 2
    assert burden["scoreable_flight_hours"] == pytest.approx(3 / 3600)
    assert burden["episodes_per_scoreable_flight_hour"] == pytest.approx(2400)


def test_deadline_recall_and_fault_mode_are_separate():
    frame = pd.DataFrame(
        {
            "flight_id": ["A"] * 4 + ["B"] * 4,
            "timestamp_s": [9, 10, 12, 20, 9, 10, 12, 20],
            "fault_onset_s": [10] * 8,
            "fault_end_s": [20] * 8,
            "flight_mode": ["hover"] * 8,
            "fault_mode": [3] * 4 + [4] * 4,
            "fault_mode_name": ["noise"] * 4 + ["scale_factor"] * 4,
            "evaluable": [True] * 8,
        }
    )
    alarms = np.array([False, False, True, False, False, False, False, True])
    result = deadline_event_metrics(frame, alarms, deadline_s=5)
    assert result["recall"] == 0.5
    assert result["by_fault_mode"]["3"]["recall"] == 1.0
    assert result["by_fault_mode"]["4"]["recall"] == 0.0


def test_wilson_interval_is_not_naive_recall():
    lower, upper = wilson_interval(7, 10)
    assert lower < 0.7 < upper

