"""Equivalence tests: apply_detector_profile_fast must match the frozen
apply_detector_profile exactly (not approximately) on every scenario, since
it is a pure performance rewrite of the same recurrence -- see ADR-040/041.
"""

import numpy as np
import pandas as pd
import pytest

from adsb.contextual_decision import ChannelAlertBudget, DetectorProfile, apply_detector_profile
from adsb.contextual_decision_fast import apply_detector_profile_fast


def _assert_identical(slow: pd.DataFrame, fast: pd.DataFrame) -> None:
    assert slow["alarm"].tolist() == fast["alarm"].tolist()
    assert slow["reset_reason"].tolist() == fast["reset_reason"].tolist()
    np.testing.assert_array_equal(
        slow["temporal_evidence"].to_numpy(), fast["temporal_evidence"].to_numpy()
    )
    assert slow["anomaly_type"].tolist() == fast["anomaly_type"].tolist()
    assert slow["channel"].tolist() == fast["channel"].tolist()
    assert slow["alert_alpha"].tolist() == fast["alert_alpha"].tolist()


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("instant", {}),
        ("persistence", {"persistence_s": 2.0}),
        ("accumulation", {"reference_surprisal": 1.0, "accumulation_threshold": 5.0}),
    ],
)
def test_matches_frozen_implementation_single_flight(mode, extra):
    budget = ChannelAlertBudget(total_alpha=0.05, channel_alpha={"c": 0.02})
    profile = DetectorProfile(anomaly_type="a", channel="c", mode=mode, max_gap_s=10.0, **extra)
    rng = np.random.default_rng(0)
    n = 40
    times = np.sort(rng.uniform(0.0, 3.0, size=n)).cumsum()
    p_values = rng.uniform(1e-4, 1.0, size=n)
    frame = pd.DataFrame(
        {
            "flight_id": ["A"] * n,
            "timestamp_utc": times,
            "channel": ["c"] * n,
            "conformal_p_value": p_values,
        }
    )
    slow = apply_detector_profile(frame, profile=profile, budget=budget)
    fast = apply_detector_profile_fast(frame, profile=profile, budget=budget)
    _assert_identical(slow, fast)


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("instant", {}),
        ("persistence", {"persistence_s": 3.0}),
        ("accumulation", {"reference_surprisal": 0.8, "accumulation_threshold": 4.0}),
    ],
)
def test_matches_frozen_implementation_multi_flight_with_gaps(mode, extra):
    budget = ChannelAlertBudget(total_alpha=0.05, channel_alpha={"c": 0.02})
    profile = DetectorProfile(anomaly_type="a", channel="c", mode=mode, max_gap_s=5.0, **extra)
    rng = np.random.default_rng(1)
    frames = []
    for flight in ("A", "B", "C"):
        n = int(rng.integers(5, 25))
        # Occasionally insert a gap large enough to force a reset (> max_gap_s).
        steps = rng.choice([0.5, 1.0, 2.0, 8.0], size=n, p=[0.4, 0.3, 0.2, 0.1])
        times = np.cumsum(steps)
        p_values = rng.uniform(1e-4, 1.0, size=n)
        frames.append(
            pd.DataFrame(
                {
                    "flight_id": [flight] * n,
                    "timestamp_utc": times,
                    "channel": ["c"] * n,
                    "conformal_p_value": p_values,
                }
            )
        )
    # Interleave flights out of contiguous-block order to exercise groupby(sort=False).
    frame = pd.concat(frames, ignore_index=False).sample(frac=1.0, random_state=2).reset_index(drop=True)
    # Restore per-flight time ordering (a decision frame is always flight-locally sorted upstream).
    frame = frame.sort_values(["flight_id", "timestamp_utc"], kind="stable").reset_index(drop=True)
    slow = apply_detector_profile(frame, profile=profile, budget=budget)
    fast = apply_detector_profile_fast(frame, profile=profile, budget=budget)
    _assert_identical(slow, fast)


def test_single_row_flight_edge_case():
    budget = ChannelAlertBudget(total_alpha=0.05, channel_alpha={"c": 0.02})
    profile = DetectorProfile(anomaly_type="a", channel="c", mode="instant", max_gap_s=5.0)
    frame = pd.DataFrame(
        {
            "flight_id": ["A"],
            "timestamp_utc": [0.0],
            "channel": ["c"],
            "conformal_p_value": [0.01],
        }
    )
    slow = apply_detector_profile(frame, profile=profile, budget=budget)
    fast = apply_detector_profile_fast(frame, profile=profile, budget=budget)
    _assert_identical(slow, fast)


def test_rejects_mixed_channel_same_as_frozen_implementation():
    frame = pd.DataFrame(
        {
            "flight_id": ["A", "A"],
            "timestamp_utc": [0.0, 1.0],
            "channel": ["speed", "track"],
            "conformal_p_value": [0.1, 0.1],
        }
    )
    budget = ChannelAlertBudget(total_alpha=0.05, channel_alpha={"speed": 0.02})
    profile = DetectorProfile(anomaly_type="speed_spike", channel="speed", mode="instant", max_gap_s=10.0)
    with pytest.raises(ValueError, match="exactly one"):
        apply_detector_profile_fast(frame, profile=profile, budget=budget)
