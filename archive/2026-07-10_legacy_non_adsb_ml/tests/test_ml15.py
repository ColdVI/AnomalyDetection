"""ML-15 drift-aware calibration unit tests."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from src.ml.decision import decision_layers
from src.ml.decision.drift_calibration import fit_drift_corrected_policy
from src.ml.decision.decision_layers import CusumPolicy, KOfNPolicy, ThresholdPolicy


def _spike(length: int, value: float = 0.9) -> np.ndarray:
    stream = np.zeros(length, dtype=float)
    stream[-1] = value
    return stream


def _sessions_for_correction() -> dict[str, list[np.ndarray]]:
    return {
        "short_a": [_spike(10)],
        "short_b": [_spike(10)],
        "long_a": [_spike(120)],
        "long_b": [_spike(120)],
        "long_c": [_spike(120)],
    }


def test_drift_corrected_threshold_is_deterministic_and_stricter():
    sessions = _sessions_for_correction()
    kwargs = dict(
        val_streams_by_session=sessions,
        budget=50.0,
        decision_fit_fn=decision_layers.fit_threshold_policy,
        seed=7,
    )
    first_policy, first_report = fit_drift_corrected_policy(**kwargs)
    second_policy, second_report = fit_drift_corrected_policy(**kwargs)
    base_policy = decision_layers.fit_threshold_policy(
        [stream for streams in sessions.values() for stream in streams],
        50.0,
        stride_seconds=1.0,
    )

    assert isinstance(first_policy, ThresholdPolicy)
    assert first_policy == second_policy
    assert first_report == second_report
    assert first_report["drift_multiplier"] > 1.0
    assert first_policy.fa_budget_per_hour == first_report["effective_budget_per_hour"]
    assert first_policy.threshold >= base_policy.threshold


def test_drift_corrected_policy_reduces_false_alarm_direction_on_stress_stream():
    sessions = _sessions_for_correction()
    corrected, report = fit_drift_corrected_policy(
        sessions,
        50.0,
        decision_layers.fit_threshold_policy,
        seed=0,
    )
    uncorrected = decision_layers.fit_threshold_policy(
        [stream for streams in sessions.values() for stream in streams],
        50.0,
        stride_seconds=1.0,
    )
    stress = np.r_[np.zeros(15), 0.9, np.zeros(5), 0.9]

    assert report["drift_multiplier"] > 1.0
    assert corrected.apply(stress).sum() <= uncorrected.apply(stress).sum()


def test_drift_multiplier_floor_keeps_budget_when_jackknife_is_optimistic():
    sessions = {f"s{i}": [np.zeros(40, dtype=float)] for i in range(4)}
    policy, report = fit_drift_corrected_policy(
        sessions,
        12.0,
        decision_layers.fit_threshold_policy,
        seed=0,
    )

    assert isinstance(policy, ThresholdPolicy)
    assert report["drift_multiplier"] == 1.0
    assert report["effective_budget_per_hour"] == 12.0


def test_drift_multiplier_cap_limits_extreme_jackknife_ratio():
    sessions = {
        "tiny_a": [_spike(1)],
        "tiny_b": [_spike(1)],
        "tiny_c": [_spike(1)],
        "tiny_d": [_spike(1)],
        "long": [_spike(10_000)],
    }
    _, report = fit_drift_corrected_policy(
        sessions,
        2.0,
        decision_layers.fit_threshold_policy,
        seed=0,
    )

    assert report["drift_multiplier"] == 5.0
    assert report["effective_budget_per_hour"] == 0.4


def test_drift_calibration_fallback_requires_and_uses_external_multiplier():
    sessions = {"a": [_spike(20)], "b": [_spike(20)], "c": [_spike(20)]}
    with pytest.raises(ValueError, match="fallback_drift_multiplier"):
        fit_drift_corrected_policy(
            sessions,
            12.0,
            decision_layers.fit_threshold_policy,
            seed=0,
        )

    policy, report = fit_drift_corrected_policy(
        sessions,
        12.0,
        decision_layers.fit_threshold_policy,
        seed=0,
        fallback_drift_multiplier=2.5,
    )
    assert isinstance(policy, ThresholdPolicy)
    assert report["fallback_used"] is True
    assert report["fallback_source"] == "provided_ml14_median_shift"
    assert report["drift_multiplier"] == 2.5
    assert report["effective_budget_per_hour"] == 4.8


def test_drift_calibration_returns_existing_policy_classes():
    sessions = {f"s{i}": [np.linspace(0.0, 0.9, 80)] for i in range(4)}
    threshold, _ = fit_drift_corrected_policy(
        sessions,
        100.0,
        decision_layers.fit_threshold_policy,
        seed=1,
    )
    k_of_n, _ = fit_drift_corrected_policy(
        sessions,
        100.0,
        decision_layers.fit_k_of_n_policy,
        seed=1,
    )
    cusum_fit = functools.partial(
        decision_layers.fit_cusum_policy,
        bootstrap_hours=0.02,
        block_seconds=2.0,
    )
    cusum, _ = fit_drift_corrected_policy(
        sessions,
        100.0,
        cusum_fit,
        seed=1,
    )

    assert isinstance(threshold, ThresholdPolicy)
    assert isinstance(k_of_n, KOfNPolicy)
    assert isinstance(cusum, CusumPolicy)
    assert decision_layers.fit_threshold_policy is decision_layers.fit_threshold_policy
    assert decision_layers.fit_k_of_n_policy is decision_layers.fit_k_of_n_policy
    assert decision_layers.fit_cusum_policy is decision_layers.fit_cusum_policy
@pytest.mark.parametrize(
    "fit_fn",
    [
        decision_layers.fit_threshold_policy,
        decision_layers.fit_k_of_n_policy,
        functools.partial(
            decision_layers.fit_cusum_policy,
            bootstrap_hours=0.02,
            block_seconds=2.0,
        ),
    ],
)
def test_parallel_jackknife_is_bitwise_equivalent_to_sequential(fit_fn):
    sessions = {
        f"session_{index}": [
            np.random.default_rng(index).beta(2.0, 8.0, size=80),
        ]
        for index in range(5)
    }
    kwargs = dict(
        val_streams_by_session=sessions,
        budget=100.0,
        decision_fit_fn=fit_fn,
        seed=17,
    )

    sequential_policy, sequential_report = fit_drift_corrected_policy(**kwargs, n_jobs=1)
    parallel_policy, parallel_report = fit_drift_corrected_policy(**kwargs, n_jobs=2)

    assert parallel_policy == sequential_policy
    assert parallel_report["session_ratios"] == sequential_report["session_ratios"]
    assert parallel_report["drift_multiplier"] == sequential_report["drift_multiplier"]
    assert (
        parallel_report["effective_budget_per_hour"]
        == sequential_report["effective_budget_per_hour"]
    )
    assert parallel_report["jackknife_n_jobs"] == 2
    assert sequential_report["jackknife_n_jobs"] == 1


def test_parallel_jackknife_rejects_zero_workers():
    with pytest.raises(ValueError, match="n_jobs must be non-zero"):
        fit_drift_corrected_policy(
            _sessions_for_correction(),
            50.0,
            decision_layers.fit_threshold_policy,
            seed=7,
            n_jobs=0,
        )
