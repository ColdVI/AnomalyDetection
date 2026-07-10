import numpy as np
import pandas as pd
import pytest

from src.common.minio_io import write_bronze
from src.common.provenance import add_provenance
from src.processing.uav_attack_silver import _log_id_and_topic, build_log_table, build_uav_attack_silver


# Real filename conventions observed in the IEEE DataPort UAV Attack zip (verified
# 2026-07-01): log_id prefixes are not uniform and contain their own underscores, which
# is exactly what breaks a generic "last _<word>_<n>.csv" regex.
@pytest.mark.parametrize(
    "source_file,expected_log_id,expected_topic",
    [
        (
            "Simulated - OTU Survey/PX4-H480-SITL/Normal/log_12_2020-8-2-14-18-24_vehicle_global_position_0.csv",
            "log_12_2020-8-2-14-18-24",
            "vehicle_global_position",
        ),
        (
            "Simulated - OTU Survey/PX4-H480-SITL/Normal/log_12_2020-8-2-14-18-24_vehicle_global_position_groundtruth_0.csv",
            "log_12_2020-8-2-14-18-24",
            "vehicle_global_position_groundtruth",
        ),
        (
            "Live GPS Spoofing and Jamming/Benign Flight/ace-benign-log_0_2033-8-19-16-27-30_battery_status_0.csv",
            "ace-benign-log_0_2033-8-19-16-27-30",
            "battery_status",
        ),
        (
            "Simulated - OTU Survey/PX4-H480-SITL/Ping DoS/001-2021-01-27-09-08-37-708_vehicle_attitude_0.csv",
            "001-2021-01-27-09-08-37-708",
            "vehicle_attitude",
        ),
        (
            "Live GPS Spoofing and Jamming/GPS Spoofing/ace-spoofing-hackrf-log_5_2033-8-19-17-14-18_vehicle_gps_position_0.csv",
            "ace-spoofing-hackrf-log_5_2033-8-19-17-14-18",
            "vehicle_gps_position",
        ),
    ],
)
def test_log_id_and_topic_handles_real_naming_conventions(source_file, expected_log_id, expected_topic):
    log_id, topic = _log_id_and_topic(source_file)
    assert log_id == expected_log_id
    assert topic == expected_topic


def test_log_id_and_topic_returns_none_for_unrelated_topics():
    log_id, topic = _log_id_and_topic("Simulated - OTU Survey/PX4-H480-SITL/Normal/log_12_2020-8-2-14-18-24_actuator_controls_0_0.csv")
    assert log_id is None
    assert topic is None


def test_build_log_table_merges_attitude_battery_gps_and_groundtruth():
    position = pd.DataFrame({
        "timestamp": [0, 100_000, 200_000],
        "lat": [36.0, 36.0001, 36.0002],
        "lon": [138.0, 138.0001, 138.0002],
        "alt": [50.0, 51.0, 52.0],
        "eph": [1.0, 1.0, 1.0],
        "epv": [1.5, 1.5, 1.5],
    })
    attitude = pd.DataFrame({
        "timestamp": [0, 200_000],
        "q[0]": [1.0, 1.0], "q[1]": [0.0, 0.0], "q[2]": [0.0, 0.0], "q[3]": [0.0, 0.0],
    })
    battery = pd.DataFrame({"timestamp": [0], "voltage_v": [16.0], "remaining": [0.9], "current_a": [5.0]})
    gps = pd.DataFrame({
        "timestamp": [0, 200_000],
        "lat": [360000000, 360002000],  # x1e7-scaled ints
        "lon": [1380000000, 1380002000],
        "jamming_indicator": [10, 12],
        "satellites_used": [11, 11],
        "time_utc_usec": [1_600_000_000_000_000, 1_600_000_000_200_000],
    })
    groundtruth = pd.DataFrame({
        "timestamp": [0, 200_000],
        "lat": [36.00005, 36.00025],
        "lon": [138.00005, 138.00025],
        "alt": [50.1, 52.1],
    })

    table = build_log_table("log_1_2020", {
        "vehicle_global_position": position,
        "vehicle_attitude": attitude,
        "battery_status": battery,
        "vehicle_gps_position": gps,
        "vehicle_global_position_groundtruth": groundtruth,
    })

    assert table is not None
    assert len(table) == 3
    # quaternion (1,0,0,0) => zero roll/pitch/yaw
    assert np.allclose(table["roll_deg"], 0.0, atol=1e-6)
    assert np.allclose(table["pitch_deg"], 0.0, atol=1e-6)
    assert np.allclose(table["yaw_deg"], 0.0, atol=1e-6)
    assert "q[0]" not in table.columns
    # raw GPS lat/lon converted from x1e7 ints to plain degrees
    assert abs(table["raw_gps_lat"].iloc[0] - 36.0) < 1e-6
    assert abs(table["raw_gps_lon"].iloc[0] - 138.0) < 1e-6
    # groundtruth columns present and distinct from the fused-estimate lat/lon
    assert "gt_lat" in table.columns and "gt_lon" in table.columns and "gt_alt" in table.columns
    assert table["timestamp_is_real_utc"].iloc[0]
    assert table["log_id"].iloc[0] == "log_1_2020"


def test_build_log_table_returns_none_without_position_topic():
    result = build_log_table("log_x", {"battery_status": pd.DataFrame({"timestamp": [0], "voltage_v": [16.0]})})
    assert result is None


def _write_uav_bronze_topic(df, *, source_file, label, attack_type, platform, collection, client):
    tagged = add_provenance(df, source_type="uav_attack", source_file=source_file)
    tagged["_attack_label"] = label
    tagged["_attack_type"] = attack_type
    tagged["_attack_platform"] = platform
    tagged["_attack_collection"] = collection
    write_bronze(tagged, "uav_attack", client=client)


def test_build_uav_attack_silver_end_to_end_via_fake_bronze(fake_minio_client):
    logs = [
        ("Simulated - OTU Survey/PX4-QUAD-SITL/Normal/log_6_2020-8-1-21-26-31", "benign", "normal", "PX4-QUAD-SITL", "simulated"),
        ("Simulated - OTU Survey/PX4-QUAD-SITL/GPS Spoofing/log_0_2020-8-2-10-39-13", "malicious", "gps_spoofing", "PX4-QUAD-SITL", "simulated"),
    ]
    for base, label, attack_type, platform, collection in logs:
        position = pd.DataFrame({
            "timestamp": [0, 100_000],
            "lat": [36.0, 36.0001],
            "lon": [138.0, 138.0001],
            "alt": [50.0, 51.0],
        })
        _write_uav_bronze_topic(
            position, source_file=f"{base}_vehicle_global_position_0.csv",
            label=label, attack_type=attack_type, platform=platform, collection=collection, client=fake_minio_client,
        )

    silver = build_uav_attack_silver(fake_minio_client)

    assert set(silver["log_id"].unique()) == {b.split("/")[-1] for b, *_ in logs}
    assert set(silver["_attack_type"].unique()) == {"normal", "gps_spoofing"}


def test_build_uav_attack_silver_returns_empty_dataframe_when_bronze_is_empty(fake_minio_client):
    result = build_uav_attack_silver(fake_minio_client)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
