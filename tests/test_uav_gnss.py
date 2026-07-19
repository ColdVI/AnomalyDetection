"""UAV GNSS katalog/degerlendirme testleri

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

from uav_gnss.catalog import FLIGHT_MODES, _assign_balanced

import numpy as np

import pandas as pd

import pytest

from uav_gnss.evaluation import deadline_event_metrics, natural_burden, wilson_interval



# ===== kaynak: test_uav_gnss_catalog =====

def _normal_rows():
    rows = []
    for mode in FLIGHT_MODES:
        for index in range(10 + (1 if mode == "hover" else 0)):
            rows.append(
                {
                    "flight_id": f"normal/{mode}/{index:02d}",
                    "class": "normal",
                    "flight_mode": mode,
                    "fault_mode": None,
                }
            )
    return rows


def _fault_rows():
    counts = {
        ("acce", 3): 2,
        ("acce", 4): 2,
        ("circling", 3): 2,
        ("circling", 4): 5,
        ("hover", 3): 5,
        ("hover", 4): 5,
        ("velocity", 3): 2,
        ("velocity", 4): 4,
        ("waypoint", 3): 5,
        ("waypoint", 4): 5,
    }
    rows = []
    for (mode, fault_mode), count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "flight_id": f"fault/{mode}/{fault_mode}/{index:02d}",
                    "class": "gps_fault",
                    "flight_mode": mode,
                    "fault_mode": fault_mode,
                }
            )
    return rows


def test_balanced_roles_match_frozen_contract():
    rows = _normal_rows() + _fault_rows()
    rows.append(
        {
            "flight_id": "quarantine/one",
            "class": "quarantine_magnetometer",
            "flight_mode": "hover",
            "fault_mode": 4,
        }
    )
    assignments = _assign_balanced(rows)
    counts = {}
    for role in assignments.values():
        counts[role] = counts.get(role, 0) + 1
    assert counts == {
        "fit": 20,
        "calibration": 10,
        "development": 23,
        "rehearsal": 15,
        "holdout": 20,
        "quarantine": 1,
    }
    for mode in FLIGHT_MODES:
        for fault_mode in (3, 4):
            cell = [
                row["flight_id"]
                for row in rows
                if row["class"] == "gps_fault"
                and row["flight_mode"] == mode
                and row["fault_mode"] == fault_mode
            ]
            assert sum(assignments[flight_id] == "rehearsal" for flight_id in cell) == 1
            assert sum(assignments[flight_id] == "holdout" for flight_id in cell) == 1



# ===== kaynak: test_uav_gnss_evaluation =====

def test_episode_burden_uses_scoreable_flight_time():
    frame = pd.DataFrame(
        {
            "flight_id": ["A"] * 5,
            "timestamp_s": [0.0, 1.0, 2.0, 20.0, 21.0],
            "dt_s": [0.0, 1.0, 1.0, 18.0, 1.0],
            "landed": [0, 0, 0, 0, 0],
            "evaluable": [False, True, True, False, True],
        }
    )
    burden = natural_burden(
        frame,
        np.array([False, True, True, False, True]),
        merge_gap_s=10.0,
    )
    assert burden["n_alert_episodes"] == 2
    assert burden["scoreable_flight_hours"] == pytest.approx(3 / 3600)
    assert burden["episodes_per_scoreable_flight_hour"] == pytest.approx(2400)


def test_deadline_recall_and_fault_mode_are_separate():
    frame = pd.DataFrame(
        {
            "flight_id": ["A"] * 4 + ["B"] * 4,
            "timestamp_s": [9, 10, 12, 20, 9, 10, 12, 20],
            "fault_onset_s": [10] * 8,
            "fault_end_s": [20] * 8,
            "flight_mode": ["hover"] * 8,
            "fault_mode": [3] * 4 + [4] * 4,
            "fault_mode_name": ["noise"] * 4 + ["scale_factor"] * 4,
            "evaluable": [True] * 8,
        }
    )
    alarms = np.array([False, False, True, False, False, False, False, True])
    result = deadline_event_metrics(frame, alarms, deadline_s=5)
    assert result["recall"] == 0.5
    assert result["by_fault_mode"]["3"]["recall"] == 1.0
    assert result["by_fault_mode"]["4"]["recall"] == 0.0


def test_wilson_interval_is_not_naive_recall():
    lower, upper = wilson_interval(7, 10)
    assert lower < 0.7 < upper

