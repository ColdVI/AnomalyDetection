"""RFLY-0 exploratory-run discipline tests."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.run_rfly0_exploratory_evaluation import (
    MODULE_HYPOTHESES,
    RFLY_EXPLORATORY_QUOTA,
    RFLY_OFFICIAL_QUOTA,
    _build_rfly_splits,
    _dataset_group,
    _diagnosis_rows,
    _exclude_invalid_ids_from_folds,
    _invalid_rfly_interval_truth,
    _truth_mask,
)


def test_rfly_exploratory_quota_fits_current_51_normal_shape():
    rows = []
    for i in range(51):
        rows.append({
            "source_id": f"Real-No_Fault/hover/{i:03d}/log",
            "label": "normal",
            "t_rel_s": 0.0,
        })
    for i in range(20):
        rows.append({
            "source_id": f"Real-Motor/hover/{i:03d}/log",
            "label": "motor_fault",
            "t_rel_s": 0.0,
        })
    features = pd.DataFrame(rows)
    folds = _build_rfly_splits(features)
    split = folds["split_00"]
    assert RFLY_EXPLORATORY_QUOTA == (15, 15)
    assert len(split["val"]) == 15
    assert len(split["test_normal"]) == 15
    assert len(split["train"]) == 21
    assert len(split["test_anomalous"]) == 20
    assert not set(split["train"]) & set(split["val"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["val"]) & set(split["test"])


def test_rfly_official_amended_quota_leaves_meaningful_train_normals():
    rows = []
    for i in range(51):
        rows.append({
            "source_id": f"Real-No_Fault/hover/{i:03d}/log",
            "label": "normal",
            "t_rel_s": 0.0,
        })
    for i in range(20):
        rows.append({
            "source_id": f"Real-Motor/hover/{i:03d}/log",
            "label": "motor_fault",
            "t_rel_s": 0.0,
        })
    features = pd.DataFrame(rows)
    folds = _build_rfly_splits(
        features,
        quota=RFLY_OFFICIAL_QUOTA,
        final_holdout_fraction=0.30,
    )
    split = folds["split_00"]
    assert RFLY_OFFICIAL_QUOTA == (12, 12)
    assert len(split["val"]) == 12
    assert len(split["test_normal"]) == 12
    assert len(split["train"]) == 27
    assert len(split["final_holdout_anomalous"]) == round(20 * 0.30)
    assert not set(split["final_holdout"]) & (
        set(split["train"]) | set(split["val"]) | set(split["test"])
    )


def test_dataset_group_keeps_detection_and_diagnosis_labels_separate():
    assert _dataset_group("Real-Motor/hover/1/log", "motor_fault") == "rfly_motor"
    assert _dataset_group("Real-Sensors/hover-GPS/1/log", "sensor_hover_gps_fault") == "rfly_sensor"
    assert _dataset_group("Real-No_Fault/hover/1/log", "normal") == "rfly_normal"
    assert _dataset_group("2019-01-01/00_00_00", "external_position_anomaly") == "sead_anomaly"


def test_diagnosis_rows_are_hypotheses_not_supervised_classes():
    streams = pd.DataFrame({
        "source_id": ["f1", "f1", "f2", "f2"],
        "kontrol_cevabi": [0.2, 0.9, 0.1, 0.2],
        "nav_butunlugu": [0.1, 0.3, 0.8, 0.7],
    })
    labels = {"f1": "motor_fault", "f2": "sensor_hover_gps_fault"}
    rows = _diagnosis_rows(
        streams,
        {"f1", "f2"},
        labels,
        ["kontrol_cevabi", "nav_butunlugu"],
    )
    by_id = {row["source_id"]: row for row in rows}
    assert by_id["f1"]["hypothesis"] == MODULE_HYPOTHESES["kontrol_cevabi"]
    assert by_id["f2"]["hypothesis"] == MODULE_HYPOTHESES["nav_butunlugu"]
    assert np.isclose(by_id["f1"]["dominant_module_score"], 0.9)


def test_rfly_truth_mask_uses_fault_interval_not_whole_flight():
    group = pd.DataFrame({"t_rel_s": [0.0, 4.9, 5.0, 6.0, 7.0, 7.1, 10.0]})

    truth = _truth_mask(
        "Real-Motor/hover/1/log",
        group,
        "motor_fault",
        sead_t0={},
        sead_ranges={},
        rfly_intervals={"Real-Motor/hover/1/log": (5.0, 7.0, "rfly_ctrl_lxl")},
    )

    assert truth.tolist() == [False, False, True, True, True, False, False]


def test_rfly_truth_mask_keeps_normal_flights_all_false():
    group = pd.DataFrame({"t_rel_s": [0.0, 1.0, 2.0]})

    truth = _truth_mask(
        "Real-No_Fault/hover/1/log",
        group,
        "normal",
        sead_t0={},
        sead_ranges={},
        rfly_intervals={},
    )

    assert truth.tolist() == [False, False, False]


def test_rfly_truth_mask_fails_if_anomaly_interval_is_missing():
    group = pd.DataFrame({"t_rel_s": [0.0, 1.0, 2.0]})

    with pytest.raises(AssertionError, match="Missing interval truth"):
        _truth_mask(
            "Real-Sensors/mag/1/log",
            group,
            "sensor_mag_fault",
            sead_t0={},
            sead_ranges={},
            rfly_intervals={},
        )


def test_invalid_interval_truth_is_excluded_from_all_split_parts():
    folds = {
        "split_00": {
            "seed": 0,
            "train": ["valid_train", "bad"],
            "val": ["valid_val", "bad"],
            "test": ["valid_test", "bad"],
            "test_anomalous": ["valid_test", "bad"],
            "final_holdout": ["holdout", "bad"],
        }
    }

    cleaned = _exclude_invalid_ids_from_folds(folds, {"bad"})

    assert cleaned["split_00"]["train"] == ["valid_train"]
    assert cleaned["split_00"]["val"] == ["valid_val"]
    assert cleaned["split_00"]["test"] == ["valid_test"]
    assert cleaned["split_00"]["test_anomalous"] == ["valid_test"]
    assert cleaned["split_00"]["final_holdout"] == ["holdout"]
    assert folds["split_00"]["test"] == ["valid_test", "bad"]


def test_invalid_interval_truth_detects_no_active_fault_without_using_proxy():
    root = Path(".tmp_test_rfly0_exploratory") / uuid.uuid4().hex
    try:
        root.mkdir(parents=True)
        table = pd.DataFrame({
            "source_id": ["Real-Motor/hover/bad/log", "Real-Motor/hover/good/log", "Real-No_Fault/hover/ok/log"],
            "label": ["motor_fault", "motor_fault", "normal"],
            "fault_onset_s": [np.nan, 5.0, np.nan],
            "fault_end_s": [np.nan, 10.0, np.nan],
            "fault_interval_source": ["rfly_ctrl_lxl_no_active_fault", "rfly_ctrl_lxl", "normal_no_fault"],
        })
        path = root / "rfly.parquet"
        table.to_parquet(path, index=False)

        invalid = _invalid_rfly_interval_truth(set(table["source_id"]), silver_path=path)

        assert set(invalid) == {"Real-Motor/hover/bad/log"}
        assert "no active fault trigger" in invalid["Real-Motor/hover/bad/log"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
