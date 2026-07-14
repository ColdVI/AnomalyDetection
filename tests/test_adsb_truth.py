"""Truth v2 satir/pencere sozlesmesi testleri."""

from __future__ import annotations

import numpy as np
import pandas as pd

from adsb.truth import (
    TRUTH_V2_COLUMNS,
    attach_clean_truth_v2,
    attach_event_truth_v2,
    paired_observable_changed,
    refresh_observable_truth_v2,
    score_support_mask,
    summarize_window_truth,
)
from adsb.windowing import build_windows


def test_paired_observable_changed_treats_paired_null_as_unchanged():
    clean = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    corrupt = pd.DataFrame({"x": [1.0, np.nan, np.nan]})

    changed = paired_observable_changed(clean, corrupt, columns=["x"])

    assert changed.tolist() == [False, False, True]


def test_observation_fn_applies_feature_or_serialization_resolution():
    clean = pd.DataFrame({"x": [1.01, 1.11]})
    corrupt = pd.DataFrame({"x": [1.02, 1.19]})

    def one_decimal(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"observed": frame["x"].round(1)})

    changed = paired_observable_changed(
        clean,
        corrupt,
        columns=["observed"],
        observation_fn=one_decimal,
    )

    assert changed.tolist() == [False, True]


def test_attach_event_truth_separates_active_from_observable_and_bounds_event():
    clean = pd.DataFrame({"timestamp_utc": [0.0, 10.0, 20.0], "x": [0.0, np.nan, 2.0]})
    corrupt = clean.copy()
    corrupt.loc[2, "x"] = 5.0

    truth = attach_event_truth_v2(
        clean,
        corrupt,
        event_type="example",
        event_id="event-1",
        injection_active=[False, True, True],
        observable_cols=["x"],
    )

    assert truth["injection_active"].tolist() == [False, True, True]
    assert truth["observable_changed"].tolist() == [False, False, True]
    assert truth["evaluable_truth"].all()
    assert truth["event_id"].eq("event-1").all()
    assert truth["event_type"].eq("example").all()
    assert truth["attack_onset"].iloc[0] == 10.0
    assert truth["observable_onset"].iloc[0] == 20.0
    assert truth["event_end"].iloc[0] == 20.0


def test_clean_truth_has_complete_negative_contract():
    truth = attach_clean_truth_v2(pd.DataFrame({"x": [1.0, 2.0]}))

    assert set(TRUTH_V2_COLUMNS).issubset(truth.columns)
    assert not truth["injection_active"].any()
    assert not truth["observable_changed"].any()
    assert truth["evaluable_truth"].all()
    assert truth["event_id"].isna().all()


def test_refresh_observable_truth_uses_post_transform_pair():
    clean = pd.DataFrame({"timestamp_utc": [0.0, 1.0], "x": [1.01, 1.11]})
    corrupt_raw = clean.copy()
    corrupt_raw["x"] = [1.02, 1.19]
    truth = attach_event_truth_v2(
        clean,
        corrupt_raw,
        event_type="quantized",
        injection_active=[True, True],
        observable_cols=["x"],
    )

    refreshed = refresh_observable_truth_v2(
        clean,
        truth,
        observable_cols=["observed"],
        observation_fn=lambda frame: pd.DataFrame({"observed": frame["x"].round(1)}),
    )

    assert refreshed["observable_changed"].tolist() == [False, True]
    assert refreshed["observable_onset"].iloc[0] == 1.0


def test_rule_and_ae_use_full_window_q():
    truth = pd.DataFrame({
        "observable_changed": [False, False, True, True],
        "evaluable_truth": [True, True, True, True],
    })

    summary = summarize_window_truth(truth, architecture="rule")

    assert summary["q_w"] == 0.5
    assert summary["y_any"] is True
    assert summary["steady_subset"] is False
    assert summary["history_contaminated"] is False
    assert score_support_mask(4, architecture="dense_ae").tolist() == [True] * 4


def test_forecaster_uses_only_target_and_separates_contaminated_history():
    truth = pd.DataFrame({
        "observable_changed": [True, True, False, False],
        "evaluable_truth": [True, True, True, True],
    })

    summary = summarize_window_truth(
        truth,
        architecture="forecaster",
        forecast_target_rows=2,
    )

    assert summary["q_w"] == 0.0
    assert summary["y_any"] is False
    assert summary["steady_subset"] is True
    assert summary["steady_label"] is False
    assert summary["history_contaminated"] is True


def test_zero_evaluable_support_is_unscoreable_truth():
    truth = pd.DataFrame({
        "observable_changed": [True, False, False, False],
        "evaluable_truth": [True, True, False, False],
    })

    summary = summarize_window_truth(
        truth,
        architecture="lstm_forecaster",
        forecast_target_rows=2,
    )

    assert np.isnan(summary["q_w"])
    assert summary["truth_scoreable"] is False
    assert pd.isna(summary["y_any"])
    assert summary["steady_subset"] is False
    assert summary["history_contaminated"] is True


def test_build_windows_integrates_forecaster_support_without_history_leakage():
    df = pd.DataFrame({
        "flight_id": "F1",
        "timestamp_utc": np.arange(6, dtype=float),
        "f1": np.arange(6, dtype=float),
        "observable_changed": [True, True, False, False, False, False],
        "evaluable_truth": True,
    })

    _, _, meta = build_windows(
        df,
        ["f1"],
        window=4,
        stride=2,
        max_gap_s=60.0,
        truth_architecture="forecaster",
        forecast_target_rows=2,
    )

    assert meta["q_w"].tolist() == [0.0, 0.0]
    assert meta["y_any"].tolist() == [False, False]
    assert meta["history_contaminated"].tolist() == [True, False]
    assert meta["t_end"].tolist() == [3.0, 5.0]
