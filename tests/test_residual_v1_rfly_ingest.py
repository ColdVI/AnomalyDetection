import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from residual_v1.ingest.rfly import (
    extract_rfly_events,
    infer_fault_class,
    ingest_rfly_flight,
    load_exclusions,
    read_ulog_topics,
)
from residual_v1.ingest.rfly_channels import CHANNELS


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
