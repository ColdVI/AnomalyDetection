import numpy as np
import pandas as pd
import pytest

from residual_v1.models.g1_ridge import (
    InsufficientSessionCoverage,
    fit_g1_channel,
    grouped_session_splits,
)


def test_g1_recovers_known_linear_gain_within_ten_percent():
    rng = np.random.default_rng(11)
    rows_per_session = 200
    sessions = [f"s{index}" for index in range(6)]
    flights = [f"f{index}" for index in range(6)]
    command = rng.normal(size=rows_per_session * len(sessions))
    response = 1.25 + 3.0 * command + rng.normal(
        scale=0.02,
        size=len(command),
    )
    matrix = pd.DataFrame(
        {
            "flight_id": np.repeat(flights, rows_per_session),
            "t": np.tile(np.arange(rows_per_session) / 10.0, len(sessions)),
            "phase": "cruise",
            "train_eligible": True,
            "command__last": command,
            "response": response,
        }
    )
    result = fit_g1_channel(
        matrix,
        channel="synthetic",
        response="response",
        session_by_flight=dict(zip(flights, sessions, strict=True)),
        expected_positive_signs={"synthetic": ("command__last",)},
    )
    assert result.coefficients["command__last"] == pytest.approx(3.0, rel=0.10)
    assert result.report["coverage"]["cv_folds"] == 5
    assert result.report["cv_r2"] > 0.99
    assert result.coefficient_sanity["status"] == "passed"
    assert np.allclose(
        result.residuals["r"],
        result.residuals["y"] - result.residuals["y_hat"],
    )


def test_session_folds_are_disjoint_and_single_session_fails_closed():
    groups = np.repeat(["s1", "s2", "s3"], 5)
    for fit_index, validation_index in grouped_session_splits(groups):
        assert not (
            set(groups[fit_index])
            & set(groups[validation_index])
        )
    with pytest.raises(InsufficientSessionCoverage, match="at least 2 sessions"):
        grouped_session_splits(["only"] * 10)
