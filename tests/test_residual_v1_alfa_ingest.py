import json

import numpy as np
import pandas as pd

from residual_v1.ingest.alfa import (
    fault_class_from_events,
    ingest_alfa_flight,
    normalise_alfa_topic,
)


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
