import json

import pandas as pd
import pytest

from rfly_full.supervised_sweep import (
    _aggregate,
    _gate_summary,
    extension_decision,
    validation_fold_for_outer,
)


def test_validation_fold_mapping_is_cyclic_and_disjoint():
    assert [validation_fold_for_outer(fold) for fold in range(5)] == [4, 0, 1, 2, 3]
    with pytest.raises(ValueError, match="0..4"):
        validation_fold_for_outer(5)


def test_extension_decision_uses_validation_boundary_only():
    plateau = pd.DataFrame({
        "epoch": range(1, 13),
        "validation_loss": [1.0, 0.8, *([0.81] * 10)],
    })
    assert extension_decision(plateau, cap=12)["extend"] is False

    boundary = pd.DataFrame({
        "epoch": range(1, 13),
        "validation_loss": [1.0] * 10 + [0.9, 0.8],
    })
    decision = extension_decision(boundary, cap=12)
    assert decision["extend"] is True
    assert decision["next_cap"] == 25

    tiny_change = pd.DataFrame({
        "epoch": range(1, 13),
        "validation_loss": [1.0] * 10 + [0.99996, 0.99995],
    })
    assert extension_decision(tiny_change, cap=12)["extend"] is False


def test_extension_decision_stops_at_final_cap():
    history = pd.DataFrame({
        "epoch": range(1, 51),
        "validation_loss": list(reversed(range(1, 51))),
    })
    decision = extension_decision(history, cap=50)
    assert decision["extend"] is False
    assert decision["next_cap"] is None


def test_gate_summary_is_json_serializable():
    rows = []
    for policy in ("critical", "advisory"):
        row = {"policy": policy}
        for metric in (
            "event_recall", "all_nonfault_fa_per_hour", "wind_fa_per_hour",
            "real_motor_recall", "real_sensor_recall", "real_macro_recall",
            "real_normal_fa_per_hour",
        ):
            for statistic in ("mean", "min", "max"):
                row[f"{metric}_{statistic}"] = 0.5
        rows.append(row)
    gates = _gate_summary(pd.DataFrame(rows))
    encoded = json.dumps(gates)
    assert "critical_development_gate" in encoded


def test_aggregate_reports_nonmissing_fold_support():
    metrics = (
        "event_recall", "all_nonfault_fa_per_hour", "wind_fa_per_hour",
        "real_motor_recall", "real_sensor_recall", "real_macro_recall",
        "real_normal_fa_per_hour", "real_normal_alarm_flight_rate",
        "median_detection_delay_s",
    )
    rows = []
    for outer_fold in range(2):
        row = {"policy": "critical", "outer_fold": outer_fold}
        row.update({metric: 0.5 for metric in metrics})
        rows.append(row)
    rows[1]["real_sensor_recall"] = float("nan")
    aggregate = _aggregate(pd.DataFrame(rows)).iloc[0]
    assert aggregate["event_recall_n"] == 2
    assert aggregate["real_sensor_recall_n"] == 1
