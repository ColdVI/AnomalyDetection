import pandas as pd
import pytest

from adsb.contextual_decision import (
    ChannelAlertBudget,
    DetectorProfile,
    apply_detector_profile,
)


def test_budget_allocations_cannot_exceed_total():
    with pytest.raises(ValueError, match="exceed"):
        ChannelAlertBudget(total_alpha=0.05, channel_alpha={"speed": 0.04, "track": 0.02})


def test_separate_channel_persistence_threshold():
    frame = pd.DataFrame(
        {
            "flight_id": ["A"] * 4,
            "timestamp_utc": [0.0, 1.0, 2.0, 3.0],
            "channel": ["track"] * 4,
            "conformal_p_value": [0.01] * 4,
        }
    )
    budget = ChannelAlertBudget(total_alpha=0.05, channel_alpha={"track": 0.02})
    profile = DetectorProfile(
        anomaly_type="track_frozen",
        channel="track",
        mode="persistence",
        max_gap_s=10.0,
        persistence_s=2.0,
    )
    result = apply_detector_profile(frame, profile=profile, budget=budget)
    assert result["alarm"].tolist() == [False, False, True, True]


def test_time_normalized_accumulation_is_cadence_comparable():
    budget = ChannelAlertBudget(total_alpha=0.05, channel_alpha={"velocity": 0.02})
    profile = DetectorProfile(
        anomaly_type="position_ramp",
        channel="velocity",
        mode="accumulation",
        max_gap_s=10.0,
        reference_surprisal=1.0,
        accumulation_threshold=100.0,
    )
    fast = pd.DataFrame(
        {
            "flight_id": ["fast"] * 5,
            "timestamp_utc": [0.0, 1.0, 2.0, 3.0, 4.0],
            "channel": ["velocity"] * 5,
            "conformal_p_value": [0.01] * 5,
        }
    )
    slow = pd.DataFrame(
        {
            "flight_id": ["slow"] * 3,
            "timestamp_utc": [0.0, 2.0, 4.0],
            "channel": ["velocity"] * 3,
            "conformal_p_value": [0.01] * 3,
        }
    )
    fast_evidence = apply_detector_profile(fast, profile=profile, budget=budget)[
        "temporal_evidence"
    ].iloc[-1]
    slow_evidence = apply_detector_profile(slow, profile=profile, budget=budget)[
        "temporal_evidence"
    ].iloc[-1]
    assert fast_evidence == pytest.approx(slow_evidence)


def test_profile_rejects_implicit_or_mixed_channel_fusion():
    frame = pd.DataFrame(
        {
            "flight_id": ["A", "A"],
            "timestamp_utc": [0.0, 1.0],
            "channel": ["speed", "track"],
            "conformal_p_value": [0.1, 0.1],
        }
    )
    budget = ChannelAlertBudget(total_alpha=0.05, channel_alpha={"speed": 0.02})
    profile = DetectorProfile(
        anomaly_type="speed_spike", channel="speed", mode="instant", max_gap_s=10.0
    )
    with pytest.raises(ValueError, match="exactly one"):
        apply_detector_profile(frame, profile=profile, budget=budget)
