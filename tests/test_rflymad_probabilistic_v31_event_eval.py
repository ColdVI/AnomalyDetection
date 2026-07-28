from __future__ import annotations

import numpy as np

from scripts.evaluate_rflymad_probabilistic_v31_events import (
    _cusum_events,
    _intervals,
    _kofn_events,
    _persistence_events,
    _refractory,
)


COMMON = {"max_gap_seconds": 5.0, "merge_gap_seconds": 0.0}


def test_persistence_is_time_based_not_row_count() -> None:
    times = np.array([0.0, 0.1, 0.2, 1.1])
    scores = np.full(4, 3.0)
    events = _persistence_events(times, scores, 2.0, 1.0, COMMON, 10.0)
    assert len(events) == 1
    assert events[0].alarm_s == 1.1


def test_large_gap_splits_events() -> None:
    events = _intervals(
        np.array([0.0, 0.5, 8.0, 8.5]),
        np.ones(4, dtype=bool),
        max_gap_s=5.0,
    )
    assert [(event.start_s, event.end_s) for event in events] == [(0.0, 0.5), (8.0, 8.5)]


def test_refractory_suppresses_repeated_alarm() -> None:
    raw = [
        type("E", (), {"start_s": 0.0, "end_s": 0.0, "alarm_s": 0.0})(),
        type("E", (), {"start_s": 3.0, "end_s": 3.0, "alarm_s": 3.0})(),
        type("E", (), {"start_s": 11.0, "end_s": 11.0, "alarm_s": 11.0})(),
    ]
    assert [event.alarm_s for event in _refractory(raw, 10.0)] == [0.0, 11.0]


def test_kofn_is_causal_and_uses_time_horizon() -> None:
    times = np.array([0.0, 0.1, 0.2, 0.8])
    scores = np.array([3.0, 0.0, 3.0, 3.0])
    events = _kofn_events(times, scores, 2.0, 0.3, 2, COMMON, 0.0)
    assert len(events) == 1
    assert events[0].alarm_s == 0.2


def test_cusum_resets_after_alarm_and_honors_refractory() -> None:
    times = np.arange(6, dtype=float)
    scores = np.full(6, 2.0)
    events = _cusum_events(times, scores, allowance=1.0, h=2.0, common=COMMON, refractory=3.0)
    assert [event.alarm_s for event in events] == [1.0, 5.0]


def test_truth_intervals_do_not_expand_to_whole_flight() -> None:
    times = np.arange(6, dtype=float)
    truth = np.array([False, False, True, True, False, False])
    events = _intervals(times, truth, max_gap_s=5.0)
    assert [(event.start_s, event.end_s) for event in events] == [(2.0, 3.0)]
