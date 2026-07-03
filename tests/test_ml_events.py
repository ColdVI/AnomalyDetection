import numpy as np

from src.ml.evaluation.events import event_metrics, k_of_n_alarm, persistent_alarm


def test_k_of_n_is_causal_and_rejects_single_spike():
    prefix = np.array([0.0, 2.0, 0.0, 2.0, 2.0])
    a = k_of_n_alarm(prefix, 1.0, k=2, n=3)
    b = k_of_n_alarm(np.r_[prefix, 99.0], 1.0, k=2, n=3)[:len(prefix)]
    assert np.array_equal(a, b)
    assert not a[1]  # tek spike alarm degil
    assert a[3]      # son uc ornegin ikisi esik ustu


def test_event_metrics_reports_detection_delay_and_false_alarm_episode():
    t = np.arange(10, dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    scores = np.array([2, 0, 0, 0, 2, 2, 0, 0, 0, 0], dtype=float)
    m = event_metrics(t, y, scores, 1.0, k=1, n=1)
    assert m["n_events"] == 1
    assert m["detected_events"] == 1
    assert m["mean_detection_delay_s"] == 1.0
    assert m["false_alarm_events"] == 1
    assert m["anomaly_coverage"] == 2 / 3


def test_persistent_alarm_latches_and_applies_cooldown_causally():
    t = np.arange(10, dtype=float)
    scores = np.array([0, 2, 0, 0, 0, 2, 0, 0, 2, 0], dtype=float)
    alarm, onsets = persistent_alarm(
        t, scores, 1.0, clear_s=2.0, cooldown_s=2.0)
    assert np.flatnonzero(onsets).tolist() == [1, 8]
    assert alarm[:4].tolist() == [False, True, True, True]

    extended_alarm, extended_onsets = persistent_alarm(
        np.r_[t, 10.0], np.r_[scores, 99.0], 1.0,
        clear_s=2.0, cooldown_s=2.0)
    assert np.array_equal(alarm, extended_alarm[:len(t)])
    assert np.array_equal(onsets, extended_onsets[:len(t)])


def test_alarm_started_before_event_is_not_counted_as_detection():
    t = np.arange(6, dtype=float)
    y = np.array([0, 0, 0, 1, 1, 0], dtype=bool)
    scores = np.array([0, 2, 2, 2, 2, 0], dtype=float)
    m = event_metrics(t, y, scores, 1.0, clear_s=5.0)
    assert m["detected_events"] == 0
    assert m["overlap_detected_events"] == 1
    assert m["preexisting_alarm_events"] == 1
    assert m["false_alarm_events"] == 1
