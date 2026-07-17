import numpy as np
import pandas as pd

from residual_v1.features.align import align_to_clock, observed_tolerances


def test_backward_tolerance_staleness_and_stale_mask():
    flight = {
        "clock": pd.DataFrame({"t": [1.0, 1.2, 2.0], "roll": [0.0, 0.1, 0.2]}),
        "slow": pd.DataFrame({"t": [0.9, 1.1, 1.7, 2.1], "ground_speed": [9, 11, 17, 21]}),
    }
    result = align_to_clock(
        flight,
        "clock",
        {"slow": 0.25},
        stale={"ground_speed": [{"start_s": 1.15, "end_s": 1.25}]},
    )
    # t=1.0 takes 0.9, never the future 1.1 sample.
    assert result.loc[0, "ground_speed"] == 9
    assert np.isclose(result.loc[0, "ground_speed_staleness_ms"], 100.0)
    # The causally matched 1.1 value is then masked by the stale segment.
    assert np.isnan(result.loc[1, "ground_speed"])
    assert np.isinf(result.loc[1, "ground_speed_staleness_ms"])
    # 1.7 is 300 ms old at t=2.0 and therefore outside tolerance.
    assert np.isnan(result.loc[2, "ground_speed"])
    assert np.isinf(result.loc[2, "ground_speed_staleness_ms"])


def test_observed_topic_rate_can_expand_a_decimated_tolerance():
    flight = {"imu": pd.DataFrame({"t": [0.0, 0.4, 0.8], "x": [1, 2, 3]})}
    tolerance = observed_tolerances(flight, {"imu": 0.04})
    assert np.isclose(tolerance["imu"], 0.6)
