"""ADS-B contextual katman testleri

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

import pandas as pd

import pytest

from adsb.conditional_calibration import (
    NATURAL_CALIBRATION_ROLE,
    ConditionalCalibrationConfig,
    HierarchicalConformalCalibrator,
)

import numpy as np

from adsb.context import CausalContextConfig, build_causal_context

from adsb.contextual_decision import (
    ChannelAlertBudget,
    DetectorProfile,
    apply_detector_profile,
)

from adsb.contextual_decision import ChannelAlertBudget, DetectorProfile, apply_detector_profile

from adsb.contextual_decision_fast import apply_detector_profile_fast

from adsb.contextual_scaling import (
    NATURAL_FIT_ROLE,
    StrictNaturalRobustScaler,
    StrictScalingConfig,
)

from adsb.context import CausalContextConfig

from adsb.contextual_windowing import build_contextual_forecast_windows



# ===== kaynak: test_adsb_conditional_calibration =====

def _calibration() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel": ["speed"] * 6 + ["track"] * 3,
            "context_phase": ["level"] * 3 + ["climb"] * 3 + ["level"] * 3,
            "context_cadence": ["cadence_0"] * 3 + ["cadence_1"] * 3 + ["cadence_0"] * 3,
            "score": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.5, 2.0],
        }
    )


def _fit() -> HierarchicalConformalCalibrator:
    return HierarchicalConformalCalibrator(ConditionalCalibrationConfig(min_group_size=3)).fit(
        _calibration(), data_role=NATURAL_CALIBRATION_ROLE, contains_synthetic=False
    )


def test_exact_empirical_tail_and_hierarchical_fallback():
    scored = pd.DataFrame(
        {
            "channel": ["speed", "speed", "speed"],
            "context_phase": ["level", "level", "unknown"],
            "context_cadence": ["cadence_0", "unseen", "unseen"],
            "score": [2.0, 2.0, 2.0],
        }
    )
    result = _fit().transform(scored)
    assert result.loc[0, "conformal_p_value"] == pytest.approx(0.75)
    assert result.loc[0, "calibration_level"] == "channel+context_phase+context_cadence"
    assert result.loc[1, "calibration_level"] == "channel+context_phase"
    assert result.loc[2, "calibration_level"] == "channel"


def test_fit_rejects_synthetic_or_wrong_role():
    calibrator = HierarchicalConformalCalibrator(ConditionalCalibrationConfig(min_group_size=3))
    with pytest.raises(ValueError, match="Synthetic"):
        calibrator.fit(
            _calibration(), data_role=NATURAL_CALIBRATION_ROLE, contains_synthetic=True
        )
    with pytest.raises(ValueError, match="Only"):
        calibrator.fit(_calibration(), data_role="truth_v2", contains_synthetic=False)


def test_alarm_threshold_is_mandatory_and_explicit():
    scored = _calibration().iloc[[0]].copy()
    calibrator = _fit()
    with pytest.raises(ValueError, match="mandatory"):
        calibrator.alarms(scored, alpha=None)
    result = calibrator.alarms(scored, alpha=0.2)
    assert result.loc[0, "alert_alpha"] == 0.2
    assert bool(result.loc[0, "alarm"]) is False



# ===== kaynak: test_adsb_context =====

def _config() -> CausalContextConfig:
    return CausalContextConfig(
        phase_history_rows=2,
        level_rate_threshold_mps=1.0,
        cadence_edges_s=(1.0, 5.0),
        max_gap_s=60.0,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_id": ["A"] * 5,
            "timestamp_utc": [0.0, 1.0, 3.0, 6.0, 70.0],
            "on_ground": [False] * 5,
            "vertical_rate_ms": [0.0, 0.0, 20.0, 0.0, 0.0],
            "track_deg": [359.0, 1.0, 90.0, np.nan, 180.0],
        }
    )


def test_context_is_lagged_causal_and_track_is_circular():
    context = build_causal_context(_frame(), _config())

    assert context.loc[0, "context_phase"] == "unknown"
    assert context.loc[2, "context_phase"] == "level"  # current 20 m/s cannot route itself
    assert context.loc[3, "context_phase"] == "climb"
    assert context.loc[1, "context_cadence"] == "cadence_0"
    assert context.loc[2, "context_cadence"] == "cadence_1"
    assert context.loc[4, "context_cadence"] == "gap"
    assert context.loc[0, "track_cos"] == pytest.approx(context.loc[1, "track_cos"], abs=1e-3)
    assert context.loc[0, "track_sin"] == pytest.approx(-context.loc[1, "track_sin"], abs=1e-3)


def test_future_changes_do_not_change_context_prefix():
    original = _frame()
    changed = original.copy()
    changed.loc[4, ["vertical_rate_ms", "track_deg"]] = [999.0, 270.0]
    left = build_causal_context(original, _config()).iloc[:4]
    right = build_causal_context(changed, _config()).iloc[:4]
    pd.testing.assert_frame_equal(left, right)


def test_context_rejects_unsorted_flight_rows():
    frame = _frame()
    frame.loc[2, "timestamp_utc"] = 0.5
    with pytest.raises(ValueError, match="sorted"):
        build_causal_context(frame, _config())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase_history_rows": 0},
        {"level_rate_threshold_mps": 0.0},
        {"cadence_edges_s": (5.0, 1.0)},
        {"max_gap_s": 5.0},
    ],
)
def test_context_config_fails_closed(kwargs):
    values = {
        "phase_history_rows": 2,
        "level_rate_threshold_mps": 1.0,
        "cadence_edges_s": (1.0, 5.0),
        "max_gap_s": 60.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        CausalContextConfig(**values)



# ===== kaynak: test_adsb_contextual_decision =====

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



# ===== kaynak: test_adsb_contextual_decision_fast =====

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



# ===== kaynak: test_adsb_contextual_scaling =====

def test_zero_mad_is_excluded_without_floor_and_transform_is_clipped():
    frame = pd.DataFrame({"active": [0.0, 1.0, 2.0, 100.0], "constant": [4.0] * 4})
    scaler = StrictNaturalRobustScaler(StrictScalingConfig(clip=3.0)).fit(
        frame,
        ("active", "constant"),
        data_role=NATURAL_FIT_ROLE,
        contains_synthetic=False,
    )
    assert scaler.active_channels == ("active",)
    assert scaler.excluded_channels_ == ("constant",)
    assert scaler.to_dict()["mad_zero_policy"] == "exclude_without_floor"
    assert scaler.transform(frame)["active"].max() == 3.0


def test_scaler_rejects_synthetic_fit():
    with pytest.raises(ValueError, match="Synthetic"):
        StrictNaturalRobustScaler(StrictScalingConfig(clip=3.0)).fit(
            pd.DataFrame({"x": [0.0, 1.0]}),
            ("x",),
            data_role=NATURAL_FIT_ROLE,
            contains_synthetic=True,
        )


def test_all_zero_mad_channels_fail_closed():
    with pytest.raises(ValueError, match="MAD=0"):
        StrictNaturalRobustScaler(StrictScalingConfig(clip=3.0)).fit(
            pd.DataFrame({"x": [1.0, 1.0]}),
            ("x",),
            data_role=NATURAL_FIT_ROLE,
            contains_synthetic=False,
        )



# ===== kaynak: test_adsb_contextual_windowing =====

def test_windows_are_next_row_causal_and_never_cross_flights():
    frame = pd.DataFrame(
        {
            "flight_id": ["A"] * 4 + ["B"] * 4,
            "timestamp_utc": [0.0, 1.0, 2.0, 3.0, 10.0, 12.0, 14.0, 16.0],
            "on_ground": [False] * 8,
            "vertical_rate_ms": [0.0] * 8,
            "track_deg": [359.0, 1.0, 2.0, 3.0, 90.0, 90.0, 90.0, 90.0],
            "signal": np.arange(8, dtype=float),
            "target": np.arange(10, 18, dtype=float),
        }
    )
    config = CausalContextConfig(
        phase_history_rows=2,
        level_rate_threshold_mps=1.0,
        cadence_edges_s=(1.0, 5.0),
        max_gap_s=60.0,
    )
    batch = build_contextual_forecast_windows(
        frame,
        signal_columns=("signal",),
        target_channels=("target",),
        history_rows=2,
        context_config=config,
    )
    assert len(batch.X) == 4
    assert batch.meta["flight_id"].tolist() == ["A", "A", "B", "B"]
    assert batch.y[:, 0].tolist() == [12.0, 13.0, 16.0, 17.0]
    assert batch.X[0, :, 0].tolist() == [0.0, 1.0]
    assert batch.X[2, :, 0].tolist() == [4.0, 5.0]
    assert "track_sin" in batch.input_features
    assert "phase=level" in batch.input_features

