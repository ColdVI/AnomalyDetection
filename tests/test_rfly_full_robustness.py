from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import rfly_full.robustness as robustness

from rfly_full.robustness import (
    _combine_windows,
    _evaluation_scores,
    _gate_summary,
    _fine_tune_until_convergence,
    _nested_normal_split,
    _nested_wind_split,
    select_rw1_components,
)
from rfly_full.normal_ae import TemporalConvAutoencoder


def _manifest() -> pd.DataFrame:
    rows = []
    for domain in ("Real", "HIL", "SIL"):
        for group in range(5):
            rows.append({
                "canonical_case_id": f"normal_{domain}_{group}",
                "domain": domain,
                "split": "development",
                "evaluation_role": "normal_reference",
                "split_group_id": f"normal:{domain}:{group}",
            })
    for domain in ("HIL", "SIL"):
        for group in range(5):
            rows.append({
                "canonical_case_id": f"wind_{domain}_{group}",
                "domain": domain,
                "split": "development",
                "evaluation_role": "environment_robustness",
                "split_group_id": f"wind:{domain}:{group}",
            })
    rows.append({
        "canonical_case_id": "fault_Real_0",
        "domain": "Real",
        "split": "development",
        "evaluation_role": "fault_detection",
        "split_group_id": "fault:Real:0",
    })
    return pd.DataFrame(rows)


def test_nested_normal_split_has_disjoint_whole_groups_and_no_faults():
    train, inner, outer, groups = _nested_normal_split(_manifest(), rotation=2)
    assert train.evaluation_role.eq("normal_reference").all()
    assert inner.evaluation_role.eq("normal_reference").all()
    assert outer.evaluation_role.eq("normal_reference").all()
    ids = [set(part.canonical_case_id) for part in (train, inner, outer)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    for domain, selected in groups.items():
        assert inner.loc[inner.domain.eq(domain), "split_group_id"].eq(selected["inner"]).all()
        assert outer.loc[outer.domain.eq(domain), "split_group_id"].eq(selected["outer"]).all()


def test_nested_wind_split_has_disjoint_whole_groups():
    train, inner, outer, groups = _nested_wind_split(_manifest(), rotation=4)
    assert train.evaluation_role.eq("environment_robustness").all()
    assert set(train.canonical_case_id).isdisjoint(inner.canonical_case_id)
    assert set(train.canonical_case_id).isdisjoint(outer.canonical_case_id)
    assert set(inner.canonical_case_id).isdisjoint(outer.canonical_case_id)
    assert set(groups) == {"HIL", "SIL"}


def test_wind_training_combination_marks_equal_sampler_strata():
    normal = (
        np.zeros((3, 2, 4), np.float32), np.zeros((3, 2, 2), np.float32),
        np.ones((3, 2, 2), np.float32), np.array(["Real", "HIL", "SIL"]),
    )
    wind = (
        np.zeros((7, 2, 4), np.float32), np.zeros((7, 2, 2), np.float32),
        np.ones((7, 2, 2), np.float32), np.array(["HIL"] * 7),
    )
    combined = _combine_windows(normal, wind)
    assert combined[0].shape[0] == 10
    assert list(combined[3]).count("NoFault") == 3
    assert list(combined[3]).count("Wind") == 7


def test_outer_wind_filter_and_normal_holdout_flag():
    scored = pd.DataFrame({
        "canonical_case_id": ["n1", "n2", "w1", "w2", "f1"],
        "evaluation_role": [
            "normal_reference", "normal_reference", "environment_robustness",
            "environment_robustness", "fault_detection",
        ],
    })
    result = _evaluation_scores(
        scored, outer_normal_ids={"n2"}, outer_wind_ids={"w1"}
    )
    assert set(result.canonical_case_id) == {"n1", "n2", "w1", "f1"}
    assert result.set_index("canonical_case_id").loc["n2", "normal_calibration_holdout"]
    assert not result.set_index("canonical_case_id").loc["n1", "normal_calibration_holdout"]


def test_gate_summary_requires_every_frozen_condition():
    passing = pd.DataFrame([{
        "rotation": rotation,
        "policy": "critical",
        "event_recall": 0.60,
        "all_nonfault_fa_per_hour": 1.0,
        "wind_fa_per_hour": 10.0,
        "real_motor_recall": 0.50,
        "real_sensor_recall": 0.40,
        "real_macro_recall": 0.45,
        "real_normal_fa_per_hour": 2.0,
        "real_normal_alarm_flight_rate": 0.20,
    } for rotation in range(5)])
    gates = _gate_summary(passing)
    assert gates["real_research_gate"]["passed"]
    assert gates["wind_intermediate_gate"]["passed"]
    failing = passing.copy()
    failing.loc[0, "real_macro_recall"] = 0.20
    assert not _gate_summary(failing)["real_research_gate"]["passed"]


def test_rw1_selection_uses_frozen_precedence(monkeypatch):
    summaries = {}
    for name in ("R1", "W1", "W2", "R2", "R3"):
        summaries[name] = {
            "real_research_gate_passed": name in {"R1", "R2"},
            "wind_intermediate_gate_passed": name in {"W1", "W2"},
        }
    monkeypatch.setattr(
        robustness, "_candidate_summary",
        lambda root, candidate: summaries.get(candidate),
    )
    monkeypatch.setattr(robustness, "_atomic_json", lambda path, value: None)
    decision = select_rw1_components(Path("."))
    assert decision["rw1_required"]
    assert decision["real_component"] == "R1"
    assert decision["wind_component"] == "W1"


def test_convergence_fine_tune_stops_on_validation_patience(monkeypatch):
    validation_losses = iter([1.0, 0.9, 0.91, 0.92])
    monkeypatch.setattr(
        robustness, "_validation_loss",
        lambda model, x, target, mask: next(validation_losses),
    )
    x = np.zeros((4, 8, 4), np.float32)
    target = np.zeros((4, 8, 2), np.float32)
    mask = np.ones((4, 8, 2), np.float32)
    strata = np.array(["Real"] * 4, dtype=object)
    model = TemporalConvAutoencoder(channels_in=4, channels_out=2)
    _, history, convergence = _fine_tune_until_convergence(
        model, (x, target, mask, strata), (x, target, mask, strata),
        max_epochs=10, patience=2, min_delta=1e-4,
    )
    assert [row["epoch"] for row in history] == [0, 1, 2, 3]
    assert convergence["best_epoch"] == 1
    assert convergence["epochs_completed"] == 3
    assert convergence["stop_reason"] == "validation_patience_exhausted"
    assert not convergence["best_is_unmodified_base"]
