import numpy as np
import pandas as pd
import pytest

from residual_v1.eval.sanity_gates import (
    GateError,
    require_s3_pass,
    s1_magnitude_gate,
    s3_event_separation_gate,
)


def _flight_frame(scores: list[float], magnitudes: list[float]) -> pd.DataFrame:
    parts = []
    for index, (score, magnitude) in enumerate(zip(scores, magnitudes, strict=True)):
        parts.append(
            pd.DataFrame(
                {
                    "flight_id": f"f{index}",
                    "z": [score, -score],
                    "input_magnitude": [magnitude, magnitude],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_s1_flags_magnitude_ordering_at_threshold():
    result = s1_magnitude_gate(
        _flight_frame([1, 2, 3, 4], [10, 20, 30, 40]),
        dataset="d",
        channel="c",
    )
    assert result.status == "flagged"
    assert np.isclose(result.metrics["spearman_rho"], 1.0)


def test_s1_passes_independent_flight_ordering():
    result = s1_magnitude_gate(
        _flight_frame([1, 2, 3, 4, 5], [30, 10, 50, 20, 40]),
        dataset="d",
        channel="c",
    )
    assert result.status == "passed"
    assert result.metrics["spearman_rho"] < 0.5


def test_s1_returns_not_evaluable_for_constant_proxy():
    result = s1_magnitude_gate(
        _flight_frame([1, 2, 3], [1, 1, 1]),
        dataset="d",
        channel="c",
    )
    assert result.status == "not_evaluable"


def test_s3_passes_separated_pre_and_post_distributions():
    time = np.arange(0.0, 120.0, 0.5)
    z = np.zeros(len(time))
    z[(time >= 100.0) & (time <= 115.0)] = 5.0
    frame = pd.DataFrame({"flight_id": "f", "t": time, "z": z})
    result = s3_event_separation_gate(
        frame,
        dataset="d",
        channel="c",
        fault_class="fault",
        events=[{"flight_id": "f", "onset_s": 100.0}],
    )
    assert result.status == "passed"
    assert result.metrics["ks_pvalue"] < 0.01


def test_s3_fails_when_frozen_windows_do_not_separate():
    time = np.arange(0.0, 120.0, 0.5)
    frame = pd.DataFrame({"flight_id": "f", "t": time, "z": np.sin(time)})
    result = s3_event_separation_gate(
        frame,
        dataset="d",
        channel="c",
        fault_class="fault",
        events=[{"flight_id": "f", "onset_s": 100.0}],
    )
    assert result.status == "failed"


def test_calibration_lock_requires_explicit_pass_for_every_class():
    with pytest.raises(GateError, match="S-3 PASS required"):
        require_s3_pass({"engine": {"status": "failed"}}, ["engine"])
    require_s3_pass({"motor": {"status": "passed"}}, ["motor"])
    require_s3_pass({"sensor": "passed"}, ["sensor"])
