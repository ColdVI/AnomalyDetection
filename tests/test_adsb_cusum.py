"""Causal vector Page CUSUM tests with hand-checkable state transitions."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
import pytest

from adsb.cusum import ROBUST_MAD_SCALE, CusumConfig, VectorPageCUSUM
from adsb.features import VECTOR_RESIDUAL_FEATURES

EAST, NORTH = VECTOR_RESIDUAL_FEATURES


def _config(**overrides) -> CusumConfig:
    values = {
        "target_vector_shift_mps": 2.0,
        "threshold_h": 1.0,
        "max_gap_s": 60.0,
        "missing_reset_s": 3.0,
        "z_clip": 3.0,
    }
    values.update(overrides)
    return CusumConfig(**values)


def _features(
    times,
    east,
    north,
    *,
    flights=None,
    on_ground=None,
) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {
            "flight_id": flights if flights is not None else ["F1"] * n,
            "timestamp_utc": times,
            "on_ground": on_ground if on_ground is not None else [False] * n,
            EAST: east,
            NORTH: north,
        }
    )


def _normal_train() -> pd.DataFrame:
    # The first value is a deliberately extreme flight-start row and is not an
    # eligible transition.  The remaining ten values have median 0 and raw
    # MAD 2, so robust MAD is exactly 2 * 1.4826.
    values = [99.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    return _features(np.arange(len(values), dtype=float), values, values)


def _fit_detector(**config_overrides) -> VectorPageCUSUM:
    return VectorPageCUSUM(_config(**config_overrides)).fit(_normal_train())


def test_fit_uses_train_only_robust_median_mad_and_physical_k():
    detector = _fit_detector()
    expected_mad = 2.0 * ROBUST_MAD_SCALE
    for channel in (EAST, NORTH):
        calibration = detector.calibration_[channel]
        assert calibration["median"] == 0.0
        assert calibration["mad"] == pytest.approx(expected_mad)
        assert calibration["k"] == pytest.approx((2.0 / sqrt(2.0)) / (2.0 * expected_mad))


def test_zero_mad_channel_is_excluded_without_floor():
    train = _normal_train()
    train[NORTH] = 0.0
    detector = VectorPageCUSUM(_config()).fit(train)
    assert detector.excluded_channels_ == {NORTH: "mad_zero"}
    assert NORTH not in detector.calibration_

    scored = detector.score_rows(_features([0.0, 1.0], [0.0, 4.0], [0.0, 1_000.0]))
    assert scored[f"{NORTH}_cusum_positive"].isna().all()
    assert scored[f"{NORTH}_cusum_negative"].isna().all()
    assert scored["cusum_observed_channels"].tolist() == [0, 1]

    serialized = detector.to_dict()
    assert serialized["state_count"] == 2
    assert serialized["configured_state_count"] == 4
    assert serialized["axis_coverage_status"] == "degraded_axis_coverage"


def test_signed_page_updates_match_formula_and_joint_alarm_uses_four_states():
    detector = _fit_detector(z_clip=10.0, threshold_h=0.5)
    calibration = detector.calibration_[EAST]
    mad = calibration["mad"]
    k = calibration["k"]
    test = _features(
        [0.0, 1.0, 2.0],
        [0.0, 2.0 * mad, -2.0 * mad],
        [0.0, 0.0, 0.0],
    )
    scored = detector.score_rows(test)

    east_positive = f"{EAST}_cusum_positive"
    east_negative = f"{EAST}_cusum_negative"
    state_columns = {
        f"{EAST}_cusum_positive",
        f"{EAST}_cusum_negative",
        f"{NORTH}_cusum_positive",
        f"{NORTH}_cusum_negative",
    }
    assert state_columns.issubset(scored.columns)
    assert scored.loc[1, east_positive] == pytest.approx(2.0 - k)
    assert scored.loc[1, east_negative] == 0.0
    assert scored.loc[2, east_positive] == 0.0
    assert scored.loc[2, east_negative] == pytest.approx(2.0 - k)
    assert scored.loc[1, "cusum_joint_score"] == scored.loc[1, east_positive]
    assert scored.loc[1, "cusum_joint_alarm"]
    assert scored.loc[2, "cusum_joint_alarm"]


def test_joint_alarm_uses_strictly_greater_than_frozen_h():
    provisional = _fit_detector(z_clip=10.0)
    mad = provisional.calibration_[EAST]["mad"]
    exact_state = 2.0 - provisional.calibration_[EAST]["k"]
    detector = _fit_detector(z_clip=10.0, threshold_h=exact_state)
    scored = detector.score_rows(
        _features([0.0, 1.0], [0.0, 2.0 * mad], [0.0, 0.0])
    )
    assert scored.loc[1, "cusum_joint_score"] == pytest.approx(exact_state)
    assert not scored.loc[1, "cusum_joint_alarm"]


def test_global_resets_cover_ground_time_and_flight_boundaries():
    detector = _fit_detector(z_clip=10.0)
    high = 4.0 * detector.calibration_[EAST]["mad"]
    test = _features(
        [0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 99.0, 99.0, 200.0],
        [0.0, high, high, high, high, high, high, high, high],
        [0.0] * 9,
        flights=["F1"] * 8 + ["F2"],
        on_ground=[False, False, True, False, False, False, False, False, False],
    )
    scored = detector.score_rows(test)
    east_positive = f"{EAST}_cusum_positive"

    assert scored["cusum_reset_reason"].tolist() == [
        "flight_start",
        "none",
        "on_ground",
        "ground_transition",
        "none",
        "long_gap",
        "negative_dt",
        "zero_dt",
        "flight_start",
    ]
    assert scored.loc[1, east_positive] > 0.0
    assert (scored.loc[[2, 3, 5, 6, 8], east_positive] == 0.0).all()
    assert not scored.loc[[0, 2, 3, 5, 6, 7, 8], "cusum_evaluable"].any()


def test_zero_dt_skips_update_without_resetting_accumulated_state():
    detector = _fit_detector(z_clip=10.0)
    high = 3.0 * detector.calibration_[EAST]["mad"]
    scored = detector.score_rows(
        _features([0.0, 1.0, 1.0, 2.0], [0.0, high, 100.0 * high, high], [0.0] * 4)
    )
    state = f"{EAST}_cusum_positive"
    assert scored.loc[2, "cusum_reset_reason"] == "zero_dt"
    assert not scored.loc[2, "cusum_evaluable"]
    assert scored.loc[2, state] == scored.loc[1, state]
    assert scored.loc[3, state] > scored.loc[2, state]


def test_missing_state_is_carried_then_channel_reset_by_elapsed_time():
    detector = _fit_detector(z_clip=10.0, missing_reset_s=2.5, threshold_h=0.5)
    high = 3.0 * detector.calibration_[EAST]["mad"]
    scored = detector.score_rows(
        _features(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, high, np.nan, np.nan, np.nan],
            [0.0] * 5,
        )
    )
    state = f"{EAST}_cusum_positive"
    missing_reset = f"{EAST}_missing_reset"

    assert scored.loc[2, state] == scored.loc[1, state]
    assert scored.loc[3, state] == scored.loc[1, state]
    assert not scored.loc[2:3, missing_reset].any()
    assert scored.loc[4, missing_reset]
    assert scored.loc[4, state] == 0.0


def test_all_channels_missing_is_not_evaluable_and_cannot_emit_fresh_alarm():
    detector = _fit_detector(z_clip=10.0, missing_reset_s=30.0, threshold_h=0.5)
    high = 3.0 * detector.calibration_[EAST]["mad"]
    scored = detector.score_rows(
        _features([0.0, 1.0, 2.0], [0.0, high, np.nan], [0.0, 0.0, np.nan])
    )
    assert scored.loc[2, "cusum_joint_score"] > 0.0
    assert not scored.loc[2, "cusum_evaluable"]
    assert not scored.loc[2, "cusum_joint_alarm"]


def test_unknown_ground_status_resets_and_is_not_evaluable():
    detector = _fit_detector(z_clip=10.0)
    high = 3.0 * detector.calibration_[EAST]["mad"]
    scored = detector.score_rows(
        _features([0.0, 1.0, 2.0], [0.0, high, high], [0.0] * 3, on_ground=[False, None, False])
    )
    assert scored["cusum_reset_reason"].tolist() == [
        "flight_start",
        "unknown_ground_status",
        "unknown_ground_status",
    ]
    assert not scored["cusum_evaluable"].any()
    assert scored["cusum_joint_score"].eq(0.0).all()


def test_scoring_is_prefix_invariant():
    detector = _fit_detector(z_clip=10.0, missing_reset_s=2.5)
    high = 3.0 * detector.calibration_[EAST]["mad"]
    test = _features(
        [0.0, 1.0, 2.0, 2.0, 3.0, 4.0],
        [0.0, high, np.nan, 100.0 * high, -high, -high],
        [0.0, 0.0, 0.0, 100.0 * high, 0.0, 0.0],
    )
    full = detector.score_rows(test)
    prefix = detector.score_rows(test.iloc[:5])
    pd.testing.assert_frame_equal(full.iloc[:5], prefix)


def test_roundtrip_preserves_config_calibration_and_scores():
    detector = _fit_detector(z_clip=10.0)
    clone = VectorPageCUSUM.from_dict(detector.to_dict())
    test = _features([0.0, 1.0, 2.0], [0.0, 5.0, -5.0], [0.0, 1.0, -1.0])
    pd.testing.assert_frame_equal(detector.score_rows(test), clone.score_rows(test))
    assert clone.to_dict() == detector.to_dict()
    assert clone.to_dict()["axis_coverage_status"] == "complete_two_axis"


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_vector_shift_mps": 0.0},
        {"threshold_h": 0.0},
        {"max_gap_s": -1.0},
        {"missing_reset_s": -1.0},
        {"z_clip": 0.0},
        {"channels": (EAST, EAST)},
    ],
)
def test_config_rejects_invalid_or_non_four_state_contract(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)
