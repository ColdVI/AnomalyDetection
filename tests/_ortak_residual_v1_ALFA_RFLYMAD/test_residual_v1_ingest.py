"""RESIDUAL-V1 ingest/split/profil testleri

Bu dosya birden fazla test modulunun birlesimidir (repo sadelestirme).
"""

import json

import numpy as np

import pandas as pd

from gecmis_calismalar.residual_v1.ingest.alfa import (
    fault_class_from_events,
    ingest_alfa_flight,
    normalise_alfa_topic,
)

from pathlib import Path

from types import SimpleNamespace

from gecmis_calismalar.residual_v1.ingest.rfly import (
    extract_rfly_events,
    infer_fault_class,
    ingest_rfly_flight,
    load_exclusions,
    read_ulog_topics,
)

from gecmis_calismalar.residual_v1.ingest.rfly_channels import CHANNELS

from gecmis_calismalar.residual_v1.ingest.splits import split_flights

from gecmis_calismalar.residual_v1.ingest.common import write_json

from gecmis_calismalar.residual_v1.ingest.profile import profile_dataset, stale_segments



# ===== kaynak: test_residual_v1_alfa_ingest =====

def test_alfa_ingest_drops_non_monotonic_wraps_quaternion_and_extracts_onset(tmp_path):
    flight_id = "carbonZ_2018-07-18-15-53-31_1_engine_failure"
    frames = {
        "mavros-nav_info-roll": pd.DataFrame(
            {
                "%time": [1_000_000_000, 2_000_000_000, 1_500_000_000, 3_000_000_000],
                "field.commanded": [190.0, 180.0, 0.0, -190.0],
                "field.measured": [-190.0, -180.0, 0.0, 190.0],
            }
        ),
        "mavros-imu-data": pd.DataFrame(
            {
                "%time": [1_000_000_000, 2_000_000_000],
                "field.orientation.x": [0.1, -0.1],
                "field.orientation.y": [0.2, -0.2],
                "field.orientation.z": [0.3, -0.3],
                "field.orientation.w": [0.9, -0.9],
                "field.angular_velocity.x": [1.0, 1.0],
                "field.angular_velocity.y": [2.0, 2.0],
                "field.angular_velocity.z": [3.0, 3.0],
            }
        ),
        "failure_status-engines": pd.DataFrame(
            {"%time": [2_500_000_000, 2_700_000_000], "field.data": [1, 1]}
        ),
    }
    report = ingest_alfa_flight(flight_id, frames, tmp_path)

    roll = pd.read_parquet(tmp_path / flight_id / "mavros-nav_info-roll.parquet")
    assert roll["t"].tolist() == [0.0, 1.0, 2.0]
    assert report["dropped_non_monotonic"]["mavros-nav_info-roll"] == 1
    assert np.all((roll[["roll_cmd", "roll"]].to_numpy() > -np.pi))
    assert np.all((roll[["roll_cmd", "roll"]].to_numpy() <= np.pi))

    imu = pd.read_parquet(tmp_path / flight_id / "mavros-imu-data.parquet")
    q0 = imu.loc[0, ["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy(float)
    q1 = imu.loc[1, ["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy(float)
    assert np.dot(q0, q1) > 0

    events = json.loads((tmp_path / flight_id / "events.json").read_text(encoding="utf-8"))
    assert events == [{"end_s": 1.7, "fault_class": "engine", "onset_s": 1.5}]


def test_rc_commands_are_flight_centered():
    frame = pd.DataFrame(
        {
            "%time": [1.0, 2.0, 3.0],
            "field.channels0": [1500, 1500, 1500],
            "field.channels1": [1500, 1500, 1500],
            "field.channels2": [1000, 1200, 1400],
            "field.channels3": [1450, 1500, 1550],
            "field.channels4": [1400, 1500, 1600],
            "field.channels5": [1420, 1500, 1580],
        }
    )
    result = normalise_alfa_topic(frame, "mavros-rc-out", 1.0)
    assert result["aileron_cmd"].tolist() == [-90.0, 0.0, 90.0]
    assert result["aileron_left_cmd"].tolist() == [-100.0, 0.0, 100.0]
    assert result["aileron_right_cmd"].tolist() == [-80.0, 0.0, 80.0]
    assert result["throttle_pwm"].tolist() == [1000, 1200, 1400]


def test_rc_commands_use_only_pre_onset_trim_samples():
    frame = pd.DataFrame(
        {
            "%time": [1e9, 2e9, 3e9, 4e9],
            "field.channels1": [1490, 1510, 1700, 1700],
            "field.channels2": [1200, 1200, 1000, 1000],
            "field.channels3": [1480, 1520, 1800, 1800],
            "field.channels4": [1400, 1600, 1900, 1900],
            "field.channels5": [1420, 1580, 1900, 1900],
        }
    )
    result = normalise_alfa_topic(
        frame,
        "mavros-rc-out",
        1e9,
        trim_before_s=2.0,
    )
    assert result["elevator_cmd"].tolist()[:2] == [-10.0, 10.0]
    assert result["rudder_cmd"].tolist()[:2] == [-20.0, 20.0]
    assert result["aileron_cmd"].tolist()[:2] == [-90.0, 90.0]


def test_multi_surface_events_keep_a_separate_class():
    events = [
        {"fault_class": "rudder", "onset_s": 2.0, "end_s": 3.0},
        {"fault_class": "aileron", "onset_s": 2.0, "end_s": 3.0},
    ]
    assert fault_class_from_events(events, "flight") == "aileron_rudder"



# ===== kaynak: test_residual_v1_rfly_ingest =====

class FakeDataset:
    def __init__(self, name, data, multi_id=0):
        self.name = name
        self.data = data
        self.multi_id = multi_id


def fake_ulog_factory(_path, requested):
    assert "battery_status" in requested
    return SimpleNamespace(
        data_list=[
            FakeDataset(
                "vehicle_attitude",
                {
                    "timestamp": np.array([1_000_000, 2_000_000]),
                    "q[0]": np.array([0.9, -0.9]),
                    "q[1]": np.array([0.1, -0.1]),
                    "q[2]": np.array([0.2, -0.2]),
                    "q[3]": np.array([0.3, -0.3]),
                },
            ),
            FakeDataset(
                "battery_status",
                {"timestamp": np.array([1_000_000]), "voltage_v": np.array([15.2])},
            ),
            FakeDataset(
                "rfly_ctrl_lxl",
                {
                    "timestamp": np.array([1_000_000, 1_500_000, 2_000_000]),
                    "id": np.array([1500, 2, 1500]),
                    "mode": np.array([1500, 3, 1500]),
                },
            ),
        ]
    )


def test_rfly_mock_ulog_ingest_extracts_interval_and_quaternion(tmp_path):
    frames = read_ulog_topics(tmp_path / "fake.ulg", ulog_factory=fake_ulog_factory)
    case_id = "Real-Motor/waypoint/42_1/log_1_2023-5-23-10-00-00"
    report = ingest_rfly_flight(case_id, frames, tmp_path)
    assert report["event_count"] == 1

    flight_root = tmp_path / Path(case_id)
    attitude = pd.read_parquet(flight_root / "vehicle_attitude.parquet")
    q0 = attitude.loc[0, ["attitude_qw", "attitude_qx", "attitude_qy", "attitude_qz"]].to_numpy(float)
    q1 = attitude.loc[1, ["attitude_qw", "attitude_qx", "attitude_qy", "attitude_qz"]].to_numpy(float)
    assert np.dot(q0, q1) > 0
    events = json.loads((flight_root / "events.json").read_text(encoding="utf-8"))
    assert events == [{"end_s": 0.5, "fault_class": "motor", "onset_s": 0.5}]


def test_rfly_exclusions_and_battery_role_are_frozen():
    exclusions = load_exclusions()
    assert len(exclusions) == 5
    assert "Real-Motor/hover/2_2/log_7_2023-5-18-10-29-00" in exclusions
    battery = next(channel for channel in CHANNELS if channel.name == "battery_voltage")
    assert battery.role == "context"
    assert infer_fault_class("Real-Sensors/acce-baro/1/log_1_2023-1-1") == "sensor"



# ===== kaynak: test_residual_v1_splits =====

def _flights():
    rows = []
    for session in range(12):
        rows.append(
            {
                "flight_id": f"normal_{session}",
                "session": f"session_{session}",
                "fault_class": "normal",
            }
        )
        rows.append(
            {
                "flight_id": f"engine_{session}",
                "session": f"session_{session}",
                "fault_class": "engine",
            }
        )
    rows.extend(
        {
            "flight_id": f"rudder_{index}",
            "session": f"rare_session_{index}",
            "fault_class": "rudder",
        }
        for index in range(4)
    )
    return rows


def test_split_is_session_isolated_deterministic_and_stratified():
    first = split_flights(_flights(), seed=11)
    second = split_flights(_flights(), seed=11)
    assert first == second
    partitions = first["partitions"]
    session_sets = {name: set(value["sessions"]) for name, value in partitions.items()}
    assert session_sets["development"].isdisjoint(session_sets["test"])
    assert session_sets["development"].isdisjoint(session_sets["holdout"])
    assert session_sets["test"].isdisjoint(session_sets["holdout"])
    assert all(session.startswith("rare_session_") for session in session_sets["development"] if session.startswith("rare_"))
    assert not any(session.startswith("rare_session_") for session in session_sets["test"] | session_sets["holdout"])
    for partition in ("development", "test", "holdout"):
        assert partitions[partition]["class_counts"]["engine"] > 0



# ===== kaynak: test_residual_v1_profile =====

def test_stale_segments_and_quarantine(tmp_path):
    silver = tmp_path / "silver"
    flight = silver / "flight_1"
    flight.mkdir(parents=True)
    write_json(
        flight / "flight.json",
        {"dataset": "alfa", "flight_id": "flight_1", "session": "s1"},
    )
    pd.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0],
            "airspeed": [100.0, 100.0, 100.0, 100.0],
        }
    ).to_parquet(flight / "mavros-nav_info-airspeed.parquet", index=False)

    segments = stale_segments(pd.Series([0.0, 1.0, 2.0]), pd.Series([1.0, 1.0, 1.0]))
    assert segments == [{"start_s": 0.0, "end_s": 2.0}]
    summary = profile_dataset(silver, tmp_path / "profile", dataset="alfa")
    assert summary["flight_count"] == 1
    assert summary["quarantine_count"] == 1

