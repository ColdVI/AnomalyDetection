import json

import numpy as np
import pandas as pd
import pytest

from residual_v1.features.physics import coordinated_turn_yaw_rate, finite_difference
from residual_v1.features.spec import (
    ALFA_SPECS,
    RFLY_SPECS,
    ResidualChannelSpec,
    descriptor_schema_payload,
    descriptor_schema_sha256,
)


@pytest.mark.parametrize("feature", ["roll_rate", "roll_rate_lag1", "lag_2_roll_rate", "roll_rate_history"])
def test_ar_leakage_guard_rejects_response_and_history(feature):
    with pytest.raises(ValueError, match="response"):
        ResidualChannelSpec("bad", ("aileron_cmd", feature), "roll_rate")


def test_response_history_is_also_rejected_as_context():
    with pytest.raises(ValueError, match="response or response history"):
        ResidualChannelSpec("bad", ("aileron_cmd",), "roll_rate", ("roll_rate_lag1",))


def test_registered_specs_and_descriptor_hash_are_stable():
    assert [spec.name.split("_")[0] for spec in ALFA_SPECS] == [f"R{i}" for i in range(1, 7)]
    assert [spec.name.split("_")[0] for spec in RFLY_SPECS] == [f"Q{i}" for i in range(1, 5)]
    payload = descriptor_schema_payload()
    assert payload["schema"] == "descriptor_schema_residual_v1"
    assert len(descriptor_schema_sha256()) == 64
    json.dumps(payload)
    assert ALFA_SPECS[-1].boundary_masks == ("waypoint",)
    assert all(not spec.boundary_masks for spec in (*ALFA_SPECS[:-1], *RFLY_SPECS))


def test_unknown_boundary_mask_is_rejected():
    with pytest.raises(ValueError, match="unsupported boundary masks"):
        ResidualChannelSpec("bad", (), "response", boundary_masks=("unknown",))


def test_physics_helpers_use_observed_samples():
    time = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    values = pd.Series([0.0, 0.5, 1.0, 1.5, 2.0], name="speed")
    derivative = finite_difference(time, values)
    assert np.allclose(derivative, 2.0)
    yaw_rate = coordinated_turn_yaw_rate(pd.Series([0.0, np.pi / 4]), pd.Series([10.0, 10.0]))
    assert np.isclose(yaw_rate.iloc[0], 0.0)
    assert np.isclose(yaw_rate.iloc[1], 9.80665 / 10.0)
