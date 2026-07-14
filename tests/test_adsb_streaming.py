from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adsb.streaming import (
    BoundedFramePrioritySampler,
    BoundedPrioritySampler,
    CusumBurdenCalibration,
    count_alarm_episodes,
    deterministic_file_sample,
    dkw_quantile_error_bound,
    moving_block_burden_rows,
    prefixed_flight_id,
    robust_sample_calibration,
    scoreable_row_exposure_seconds,
    select_cusum_threshold,
    stable_fit_role,
)


def test_day_prefix_and_hash_split_are_order_independent():
    assert prefixed_flight_id("2026-02-28", "abc_000") == "2026-02-28:abc_000"
    assert stable_fit_role("abc_000", seed=7) == stable_fit_role("abc_000", seed=7)


def test_file_sample_is_reproducible_and_value_independent():
    values = np.arange(1000)
    a = deterministic_file_sample(values, probability=0.2, seed=1, file_key="p", purpose="x")
    b_mask_values = deterministic_file_sample(values + 10_000, probability=0.2, seed=1, file_key="p", purpose="x")
    assert np.array_equal(a + 10_000, b_mask_values)


def test_robust_calibration_excludes_exact_zero_mad_without_floor():
    result = robust_sample_calibration({"good": [0.0, 1.0, 2.0], "zero": [4.0, 4.0, 4.0]})
    assert result["excluded_channels"] == ["zero"]
    assert result["calibration"]["good"]["mad"] == pytest.approx(1.4826)
    assert dkw_quantile_error_bound(1000) > 0


def test_episode_and_exposure_contracts_are_separate():
    times = np.array([0.0, 10.0, 20.0, 100.0])
    assert count_alarm_episodes(times, np.array([False, True, True, True]), merge_gap_s=60) == 2
    assert scoreable_row_exposure_seconds(times, np.ones(4, bool), max_gap_s=60) == pytest.approx(20.0)


def test_moving_blocks_and_threshold_selection_use_natural_budget_only():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0, 10.0),
        advisory_budget_episodes_per_hour=0.1,
        bootstrap_repetitions=20,
        moving_block_s=300.0,
        moving_block_stride_s=150.0,
    )
    times = np.arange(0.0, 1201.0, 10.0)
    scores = np.full(len(times), 5.0)
    rows = moving_block_burden_rows(
        "f", times, scores, np.ones(len(times), bool), contract=contract, max_gap_s=60.0
    )
    frame = pd.DataFrame(rows)
    result = select_cusum_threshold(frame, contract=contract)
    assert result["selected_h"] == 10.0
    assert result["candidates"][0]["meets_advisory_budget"] is False
    assert result["candidates"][1]["meets_advisory_budget"] is True


def test_moving_blocks_count_continuing_alarm_episode_onset_only_once():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0,), bootstrap_repetitions=5,
        moving_block_s=300.0, moving_block_stride_s=150.0,
    )
    times = np.arange(0.0, 1201.0, 10.0)
    rows = moving_block_burden_rows(
        "f", times, np.full(len(times), 5.0), np.ones(len(times), bool),
        contract=contract, max_gap_s=60.0,
    )
    assert sum(row["h_1"] for row in rows) == 1


def test_moving_blocks_anchor_tail_and_use_half_open_nonfinal_boundary():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0,),
        bootstrap_repetitions=5,
        moving_block_s=300.0,
        moving_block_stride_s=300.0,
    )
    times = np.array([0.0, 300.0, 350.0, 650.0])
    scores = np.array([0.0, 5.0, 0.0, 5.0])
    rows = moving_block_burden_rows(
        "f",
        times,
        scores,
        np.ones(len(times), bool),
        contract=contract,
        max_gap_s=400.0,
    )
    assert [row["block_start"] for row in rows] == [0.0, 300.0, 350.0]
    # t=300 is excluded from [0,300); t=650 is included by the final block.
    assert [row["h_1"] for row in rows] == [0, 1, 1]
    # The valid interval ending exactly at the final block's left boundary is
    # retained by endpoint attribution.
    assert rows[-1]["exposure_s"] == pytest.approx(350.0)


def test_full_flight_counters_define_observed_burden_not_overlapping_blocks():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0, 10.0),
        advisory_budget_episodes_per_hour=100.0,
        bootstrap_repetitions=10,
    )
    blocks = pd.DataFrame(
        {
            "exposure_s": [1800.0, 1800.0, 1800.0],
            "h_1": [4, 4, 4],
            "h_10": [0, 0, 0],
        }
    )
    result = select_cusum_threshold(
        blocks,
        contract=contract,
        observed_exposure_s=3600.0,
        observed_episodes_by_h={1.0: 1, 10.0: 0},
    )
    assert result["observed_burden_source"] == "full_flight_counters"
    assert result["observed_exposure_hours"] == pytest.approx(1.0)
    assert result["candidates"][0]["observed_episode_count"] == 1
    assert result["candidates"][0]["observed_episodes_per_hour"] == pytest.approx(1.0)
    assert result["candidates"][1]["observed_episodes_per_hour"] == pytest.approx(0.0)


def test_conservative_upper_cannot_fall_below_full_flight_observed_rate():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0, 10.0),
        advisory_budget_episodes_per_hour=0.5,
        bootstrap_repetitions=10,
    )
    # The bootstrap sample has no h=1 episode, while the authoritative
    # full-flight counter has one episode/hour.  The raw quantile therefore
    # cannot be used by itself as an upper bound or for threshold selection.
    blocks = pd.DataFrame(
        {
            "exposure_s": [1800.0, 1800.0],
            "h_1": [0, 0],
            "h_10": [0, 0],
        }
    )
    result = select_cusum_threshold(
        blocks,
        contract=contract,
        observed_exposure_s=3600.0,
        observed_episodes_by_h={1.0: 1, 10.0: 0},
    )

    low_h = result["candidates"][0]
    assert low_h["bootstrap_raw_quantile_95_episodes_per_hour"] == 0.0
    assert low_h["observed_episodes_per_hour"] == pytest.approx(1.0)
    assert low_h["conservative_upper_95_episodes_per_hour"] == pytest.approx(1.0)
    assert low_h["meets_advisory_budget"] is False
    assert result["selected_h"] == 10.0
    assert "bootstrap_upper_95_episodes_per_hour" not in low_h


def test_conservative_upper_uses_raw_bootstrap_quantile_when_it_is_larger():
    contract = CusumBurdenCalibration(
        candidate_h=(1.0,),
        advisory_budget_episodes_per_hour=100.0,
        bootstrap_repetitions=10,
    )
    blocks = pd.DataFrame({"exposure_s": [3600.0], "h_1": [3]})
    result = select_cusum_threshold(
        blocks,
        contract=contract,
        observed_exposure_s=3600.0,
        observed_episodes_by_h={1.0: 1},
    )
    candidate = result["candidates"][0]
    assert candidate["bootstrap_raw_quantile_95_episodes_per_hour"] == pytest.approx(3.0)
    assert candidate["conservative_upper_95_episodes_per_hour"] == pytest.approx(3.0)


def test_full_flight_observed_arguments_are_all_or_nothing_and_exact():
    contract = CusumBurdenCalibration(candidate_h=(1.0,), bootstrap_repetitions=2)
    blocks = pd.DataFrame({"exposure_s": [60.0], "h_1": [0]})
    with pytest.raises(ValueError, match="supplied together"):
        select_cusum_threshold(blocks, contract=contract, observed_exposure_s=60.0)
    with pytest.raises(ValueError, match="exactly match"):
        select_cusum_threshold(
            blocks,
            contract=contract,
            observed_exposure_s=60.0,
            observed_episodes_by_h={},
        )


def _add_scalar_streams(sampler):
    sampler.add(
        np.r_[np.arange(400.0), np.nan],
        probability=0.37,
        seed=7,
        file_key="a",
        purpose="x",
    )
    sampler.add(
        np.arange(400.0, 1000.0),
        probability=0.37,
        seed=7,
        file_key="a",
        purpose="x",
    )
    sampler.add(
        np.arange(10_000.0, 10_500.0),
        probability=0.37,
        seed=7,
        file_key="b",
        purpose="x",
    )


def test_bounded_priority_sampler_is_chunk_exact_deterministic_and_hard_capped():
    chunked = BoundedPrioritySampler(capacity=25)
    _add_scalar_streams(chunked)

    one_shot = BoundedPrioritySampler(capacity=25)
    one_shot.add(
        np.arange(1000.0),
        probability=0.37,
        seed=7,
        file_key="a",
        purpose="x",
    )
    one_shot.add(
        np.arange(10_000.0, 10_500.0),
        probability=0.37,
        seed=7,
        file_key="b",
        purpose="x",
    )
    assert len(chunked.values) == 25
    assert chunked.finite_seen == 1500
    assert np.array_equal(chunked.values, one_shot.values)
    assert np.array_equal(chunked.priorities, one_shot.priorities)

    unbounded = BoundedPrioritySampler(capacity=2000)
    _add_scalar_streams(unbounded)
    assert np.array_equal(chunked.values, unbounded.values[:25])
    assert np.array_equal(chunked.priorities, unbounded.priorities[:25])


def test_bounded_priority_sampler_is_independent_of_file_processing_order():
    forward = BoundedPrioritySampler(capacity=40)
    reverse = BoundedPrioritySampler(capacity=40)
    streams = [("a", np.arange(500.0)), ("b", np.arange(1000.0, 1500.0))]
    for key, values in streams:
        forward.add(values, probability=1.0, seed=11, file_key=key, purpose="p")
        assert len(forward.values) <= forward.capacity
    for key, values in reversed(streams):
        reverse.add(values, probability=1.0, seed=11, file_key=key, purpose="p")
        assert len(reverse.values) <= reverse.capacity
    assert np.array_equal(forward.values, reverse.values)
    assert np.array_equal(forward.priorities, reverse.priorities)


def test_bounded_frame_sampler_is_chunk_exact_order_independent_and_hard_capped():
    source_a = pd.DataFrame({"row_id": np.arange(100), "value": np.arange(100) * 2})
    source_b = pd.DataFrame(
        {"row_id": np.arange(100, 180), "value": np.arange(100, 180) * 2}
    )

    chunked = BoundedFramePrioritySampler(capacity=30, seed=19)
    chunked.add(source_a.iloc[:35], file_key="a", purpose="blocks")
    chunked.add(source_a.iloc[35:], file_key="a", purpose="blocks")
    chunked.add(source_b, file_key="b", purpose="blocks")

    reordered = BoundedFramePrioritySampler(capacity=30, seed=19)
    reordered.add(source_b, file_key="b", purpose="blocks")
    reordered.add(source_a, file_key="a", purpose="blocks")

    assert chunked.rows_seen == reordered.rows_seen == 180
    assert len(chunked.frame) == len(reordered.frame) == 30
    pd.testing.assert_frame_equal(chunked.frame, reordered.frame)

    unbounded = BoundedFramePrioritySampler(capacity=180, seed=19)
    unbounded.add(source_a, file_key="a", purpose="blocks")
    unbounded.add(source_b, file_key="b", purpose="blocks")
    pd.testing.assert_frame_equal(
        chunked.frame,
        unbounded.frame.iloc[:30].reset_index(drop=True),
    )


def test_bounded_frame_sampler_rejects_schema_drift_and_reserved_column():
    sampler = BoundedFramePrioritySampler(capacity=2)
    sampler.add(pd.DataFrame({"a": [1]}), file_key="x", purpose="p")
    with pytest.raises(ValueError, match="identical ordered columns"):
        sampler.add(pd.DataFrame({"b": [2]}), file_key="y", purpose="p")
    with pytest.raises(ValueError, match="reserved column"):
        sampler.add(
            pd.DataFrame({"a": [2], "_sample_priority": [0.1]}),
            file_key="z",
            purpose="p",
        )
