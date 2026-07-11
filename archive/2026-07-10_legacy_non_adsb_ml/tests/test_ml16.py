"""ML-16 self-conditioned detector discipline tests."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from scripts import build_ml16_residual_channels as chronos_builder
from scripts import build_ml16_tabfm_residual as tabfm_builder
from src.ml.evaluation import score_fusion


def test_tabfm_family_exclusion_removes_only_same_root_columns():
    columns = [
        "source_id",
        "t_rel_s",
        "actuator_output_std",
        "actuator_output_range",
        "actuator_output_std_5s_max",
        "actuator_thrust_cmd",
        "hgt_test_ratio",
        "hgt_test_ratio_5s_max",
        "gps_speed_calc_mps",
    ]

    excluded = tabfm_builder.excluded_family_columns(columns, "actuator_output_std")
    predictors = tabfm_builder.predictor_columns(columns, "actuator_output_std")

    assert excluded == [
        "actuator_output_range",
        "actuator_output_std",
        "actuator_output_std_5s_max",
    ]
    assert "actuator_thrust_cmd" in predictors
    assert "actuator_output_range" not in predictors
    assert "source_id" not in predictors


def test_tabfm_hgt_family_exclusion_uses_prefix_root():
    columns = ["hgt_test_ratio", "hgt_test_ratio_5s_max", "hgt_test_flag", "alt"]
    assert tabfm_builder.excluded_family_columns(columns, "hgt_test_ratio") == [
        "hgt_test_flag",
        "hgt_test_ratio",
        "hgt_test_ratio_5s_max",
    ]


def test_tabfm_context_indices_are_seeded_deterministic_and_bounded():
    first = tabfm_builder.deterministic_context_indices(10_000, 4096, seed=3)
    second = tabfm_builder.deterministic_context_indices(10_000, 4096, seed=3)
    other = tabfm_builder.deterministic_context_indices(10_000, 4096, seed=4)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 4096
    assert len(set(first.tolist())) == 4096
    assert first.min() >= 0
    assert first.max() < 10_000
    assert np.all(first[:-1] <= first[1:])
    assert not np.array_equal(first, other)


def test_chronos_past_context_positions_ignore_future_values():
    times = np.arange(80, dtype=float)
    values = np.sin(times / 9.0)
    changed_future = values.copy()
    changed_future[40:] = 1_000_000.0

    original = chronos_builder.past_context_positions(
        values, times, stride_s=1.0, min_context=8,
    )
    changed = chronos_builder.past_context_positions(
        changed_future, times, stride_s=1.0, min_context=8,
    )

    np.testing.assert_array_equal(original[original < 40], changed[changed < 40])


def test_chronos_channel_selection_is_deterministic_by_completeness():
    frame = pd.DataFrame({
        "gps_speed_calc_mps": [1.0, 2.0, np.nan, 4.0],
        "vel_m_s": [1.0, 2.0, 3.0, 4.0],
    })
    selection = chronos_builder.choose_channel_by_completeness(
        frame,
        ("gps_speed_calc_mps", "vel_m_s"),
        completeness_floor=0.99,
    )

    assert selection["selected_channel"] == "vel_m_s"
    assert selection["passed"] is True


def test_ml16_builders_reuse_score_fusion_and_have_no_training_loop():
    tabfm_source = inspect.getsource(tabfm_builder)
    chronos_source = inspect.getsource(chronos_builder)

    assert tabfm_builder.last_causal_per_bucket is score_fusion.last_causal_per_bucket
    assert "optimizer.step" not in tabfm_source
    assert ".backward(" not in tabfm_source
    assert "loss.backward" not in tabfm_source
    assert ".train(" not in tabfm_source
    assert "TabFMRegressor" in tabfm_source
    assert "BaseChronosPipeline" not in chronos_source
