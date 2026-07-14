from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adsb.evaluation import (
    EpisodeContract,
    alarm_episodes,
    diagnostic_window_metrics,
    event_detection_metrics,
    natural_alert_burden,
    scoreable_exposure,
)


def test_window_metrics_keep_mixed_q_out_of_steady_state():
    result = diagnostic_window_metrics(
        np.array([0.0, 0.25, 1.0, np.nan]), np.array([0.0, 0.5, 1.0, 9.0])
    )
    assert result["n_unscoreable"] == 1
    assert result["q_strata"] == {"q_eq_0": 1, "q_mixed": 1, "q_eq_1": 1}
    assert result["primary_y_any"]["n_positive"] == 2
    assert result["secondary_steady_state"]["n"] == 2


def test_exposure_uses_interval_union_not_window_sum():
    meta = pd.DataFrame(
        {"flight_id": ["a", "a", "b"], "t_start": [0.0, 5.0, 0.0], "t_end": [10.0, 15.0, 5.0]}
    )
    result = scoreable_exposure(meta)
    assert result["scoreable_seconds"] == pytest.approx(20.0)
    assert result["n_scoreable_flights"] == 2


def test_alarm_episode_merge_is_flight_local_and_uses_t_end():
    meta = pd.DataFrame(
        {"flight_id": ["a", "a", "a", "b"], "t_start": [0, 0, 0, 0], "t_end": [10, 65, 130, 50]}
    )
    episodes = alarm_episodes(meta, np.ones(4, dtype=bool), contract=EpisodeContract(merge_gap_s=60))
    assert len(episodes) == 3
    assert episodes.loc[episodes.flight_id == "a", "n_emissions"].tolist() == [2, 1]


def test_natural_burden_reports_episode_rate_and_flight_fraction_separately():
    meta = pd.DataFrame(
        {"flight_id": ["a", "a", "b"], "t_start": [0.0, 1800.0, 0.0], "t_end": [1800.0, 3600.0, 3600.0]}
    )
    result = natural_alert_burden(meta, np.array([True, True, False]))
    assert result["n_alert_episodes"] == 2
    assert result["scoreable_flight_hours"] == pytest.approx(2.0)
    assert result["alert_episodes_per_scoreable_flight_hour"] == pytest.approx(1.0)
    assert result["alerted_flight_fraction"] == pytest.approx(0.5)


def test_event_metric_does_not_point_adjust_whole_event():
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "flight_id": ["a", "b"],
            "observable_onset": [100.0, 200.0],
            "event_end": [160.0, 260.0],
        }
    )
    meta = pd.DataFrame({"flight_id": ["a", "a", "b"], "t_end": [90.0, 130.0, 270.0]})
    result = event_detection_metrics(events, meta, np.array([True, True, True]))
    assert result["event_recall"] == pytest.approx(0.5)
    assert result["first_alarm_delay_s"]["median"] == pytest.approx(30.0)
