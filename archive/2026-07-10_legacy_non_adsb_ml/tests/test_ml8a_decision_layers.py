import numpy as np

from src.ml.decision.decision_layers import (
    cusum_alarm_onsets,
    fit_cusum_policy,
    fit_k_of_n_policy,
    fit_threshold_policy,
)
from src.ml.evaluation.events import event_metrics


def test_cusum_prefix_invariance():
    prefix = np.array([0.1, 0.2, 0.15, 0.8, 0.7, 0.1])
    extended = np.r_[prefix, [0.99, 0.99, 0.99]]
    kwargs = dict(mu_normal=-1.5, sigma_normal=1.0, k=0.5, h=2.0, refractory_steps=2)
    assert np.array_equal(
        cusum_alarm_onsets(prefix, **kwargs),
        cusum_alarm_onsets(extended, **kwargs)[: len(prefix)],
    )


def test_cusum_fa_calibration():
    rng = np.random.default_rng(3)
    streams = [rng.beta(2, 20, 900), rng.beta(2, 20, 1100)]
    policy = fit_cusum_policy(streams, 12.0, bootstrap_hours=3.0, seed=4)
    assert policy.calibration_fa_per_hour <= 12.0
    assert policy.bootstrap_hours == 3.0


def test_cusum_uses_circular_blocks_for_short_isolated_flights():
    streams = [np.linspace(0.05, 0.15, 20), np.linspace(0.1, 0.2, 25)]
    policy = fit_cusum_policy(streams, 12.0, bootstrap_hours=1.0, block_seconds=60.0, seed=2)
    assert policy.calibration_fa_per_hour <= 12.0


def test_refractory_no_double_onset():
    scores = np.full(100, 0.999)
    onsets = cusum_alarm_onsets(
        scores, mu_normal=-3.0, sigma_normal=1.0, k=0.5, h=1.0, refractory_steps=30
    )
    indices = np.flatnonzero(onsets)
    assert np.all(np.diff(indices) >= 31)


def test_threshold_and_k_of_n_calibrate_to_budget():
    streams = [np.r_[np.zeros(3598), 1.0, 0.0], np.zeros(3600)]
    threshold = fit_threshold_policy(streams, 1.0)
    k_of_n = fit_k_of_n_policy(streams, 1.0)
    assert threshold.calibration_fa_per_hour <= 1.0
    assert k_of_n.calibration_fa_per_hour <= 1.0


def test_event_metrics_matches_ml7():
    t = np.arange(8, dtype=float)
    y = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=bool)
    scores = np.array([0, 0, 0, 2, 0, 2, 0, 0], dtype=float)
    direct = event_metrics(t, y, scores, 1.0, k=2, n=3)
    same_shared_function = event_metrics(t, y, scores, 1.0, k=2, n=3)
    assert direct == same_shared_function
