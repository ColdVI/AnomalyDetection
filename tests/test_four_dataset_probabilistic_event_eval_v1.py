from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.four_dataset_probabilistic_event_eval_v1 import (
    TruthContext,
    _attach_truth,
    _bootstrap_source_intervals,
    _contiguous_true_intervals,
    _evaluate_point,
    _false_event_rate,
    _flight_metrics,
    _uav_sead_ranges,
)


def test_persistence_and_merge_are_time_based() -> None:
    times = np.arange(0.0, 3.1, 0.5)
    active = np.array([False, True, True, True, False, True, True])
    events = _contiguous_true_intervals(
        times,
        active,
        max_gap_s=1.0,
        minimum_duration_s=0.5,
        merge_gap_s=1.0,
    )
    assert len(events) == 1
    assert events[0].start_s == pytest.approx(0.5)
    assert events[0].alarm_s == pytest.approx(1.0)
    assert events[0].end_s == pytest.approx(3.0)


def test_validation_false_event_rate_counts_events_not_alarm_windows() -> None:
    validation = pd.DataFrame(
        {
            "source_id": ["normal"] * 7,
            "target_time_s": np.arange(7, dtype=float),
            "score": [0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
            "is_anomaly_flight": [False] * 7,
        }
    )
    rate, count, hours = _false_event_rate(
        validation,
        threshold=1.0,
        persistence_s=1.0,
        merge_gap_s=0.5,
        max_gap_s=5.0,
    )
    assert count == 2
    assert hours == pytest.approx(6.0 / 3600.0)
    assert rate == pytest.approx(1200.0)


def test_event_metric_does_not_point_adjust_entire_truth_range() -> None:
    test = pd.DataFrame(
        {
            "source_id": ["fault"] * 6 + ["normal"] * 6,
            "target_time_s": list(np.arange(6, dtype=float)) * 2,
            "score": [0, 0, 2, 0, 0, 0] + [0, 0, 0, 0, 0, 0],
            "is_anomaly_flight": [True] * 6 + [False] * 6,
            "truth_available": [True] * 12,
            "truth_active": [False, False, True, True, True, True] + [False] * 6,
        }
    )
    result = _evaluate_point(
        test,
        threshold=1.0,
        persistence_s=0.0,
        merge_gap_s=0.0,
        max_gap_s=5.0,
    )
    assert result["event_recall"] == pytest.approx(1.0)
    assert result["range_recall_time_weighted"] == pytest.approx(0.0)
    assert result["detected_events"] == 1


def test_flight_aggregation_family_is_reported_without_selection() -> None:
    test = pd.DataFrame(
        {
            "source_id": ["normal"] * 4 + ["fault"] * 4,
            "score": [0.0, 0.1, 0.2, 0.3, 0.0, 0.2, 0.4, 0.8],
            "is_anomaly_flight": [False] * 4 + [True] * 4,
        }
    )
    result = _flight_metrics(test)
    assert set(result) == {
        "max",
        "mean",
        "q99",
        "q995",
        "top_0_5pct_mean",
        "top_1pct_mean",
        "top_2pct_mean",
    }
    assert all(value["roc_auc"] == pytest.approx(1.0) for value in result.values())


def test_uav_sead_nested_ranges_keep_categories_and_union() -> None:
    ranges, categories, labels = _uav_sead_ranges(
        {
            "flight": {
                "label": "external_position_anomaly",
                "ranges": [
                    [
                        ["Position.X", [[10, 20], [30, 40]]],
                        ["Position.Y", [[15, 25]]],
                    ]
                ],
            }
        }
    )
    assert ranges["flight"] == ((10.0, 20.0), (15.0, 25.0), (30.0, 40.0))
    assert categories["flight"] == ("Position.X", "Position.Y")
    assert labels["flight"] == "external_position_anomaly"


def test_uav_sead_truth_uses_absolute_microsecond_mapping() -> None:
    context = TruthContext(
        t0_by_source={"flight": 1_000_000.0},
        intervals_by_source={"flight": ((2_000_000.0, 3_000_000.0),)},
        categories_by_source={"flight": ("Position.X",)},
        labels_by_source={"flight": "external_position_anomaly"},
        evidence={},
    )
    truth = _attach_truth(
        Path("."),
        None,
        "uav_sead",
        "flight",
        "external_position_anomaly",
        np.array([0.0, 1.0, 2.0, 3.0]),
        "normal",
        None,
        context,
    )
    assert truth["truth_available"].tolist() == [True] * 4
    assert truth["truth_active"].tolist() == [False, True, True, False]


def test_alfa_failure_truth_latches_from_first_active_status() -> None:
    context = TruthContext(
        t0_by_source={"flight": 0.0},
        intervals_by_source={"flight": ((5.0, float("inf")),)},
        categories_by_source={"flight": ("elevator",)},
        labels_by_source={},
        evidence={},
    )
    truth = _attach_truth(
        Path("."),
        None,
        "alfa",
        "flight",
        "elevator_fault",
        np.array([4.9, 5.0, 6.0]),
        "normal",
        None,
        context,
    )
    assert truth["truth_active"].tolist() == [False, True, True]


def test_source_bootstrap_is_stratified_and_deterministic() -> None:
    records = [
        {
            "source_id": "normal",
            "is_anomaly_flight": False,
            "event_count": 0,
            "detected_events": 0,
            "false_events": 1,
            "normal_exposure_s": 3600.0,
            "detection_delays_s": [],
            "predicted_duration_s": 1.0,
            "truth_duration_s": 0.0,
            "overlap_duration_s": 0.0,
        },
        {
            "source_id": "fault",
            "is_anomaly_flight": True,
            "event_count": 2,
            "detected_events": 1,
            "false_events": 0,
            "normal_exposure_s": 0.0,
            "detection_delays_s": [2.0],
            "predicted_duration_s": 2.0,
            "truth_duration_s": 4.0,
            "overlap_duration_s": 2.0,
        },
    ]
    result = _bootstrap_source_intervals(records, 20, 0.95, 7)
    assert result is not None
    assert result["replicates"] == 20
    assert result["percentile_intervals"]["event_recall"] == pytest.approx([0.5, 0.5])
    assert result["percentile_intervals"]["false_events_per_normal_hour"] == pytest.approx([1.0, 1.0])
