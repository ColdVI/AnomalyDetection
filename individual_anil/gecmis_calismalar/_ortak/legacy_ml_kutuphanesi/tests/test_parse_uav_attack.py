import io
import zipfile

import pandas as pd

from src.common.minio_io import write_bronze_bytes
from src.silver.parse_uav_attack import (
    build_uav_attack_silver,
    infer_label_from_path,
    parse_zip_bytes,
    split_log_and_topic,
)


# Real filename conventions observed in the IEEE DataPort UAV Attack zip (verified
# 2026-07-01): log_id prefixes are not uniform and contain their own underscores, which
# is exactly what breaks a generic "last _<word>_<n>.csv" regex.
def test_split_log_and_topic_handles_real_naming_conventions():
    assert split_log_and_topic(
        "Simulated - OTU Survey/PX4-H480-SITL/Normal/log_12_2020-8-2-14-18-24_vehicle_global_position_0.csv"
    ) == ("log_12_2020-8-2-14-18-24", "vehicle_global_position")
    assert split_log_and_topic(
        "Live GPS Spoofing and Jamming/Benign Flight/ace-benign-log_0_2033-8-19-16-27-30_battery_status_0.csv"
    ) == ("ace-benign-log_0_2033-8-19-16-27-30", "battery_status")
    assert split_log_and_topic(
        "Simulated - OTU Survey/PX4-H480-SITL/Ping DoS/001-2021-01-27-09-08-37-708_vehicle_attitude_0.csv"
    ) == ("001-2021-01-27-09-08-37-708", "vehicle_attitude")


def test_split_log_and_topic_returns_none_for_unrelated_topics():
    log_id, topic = split_log_and_topic("Normal/log_12_2020-8-2-14-18-24_actuator_controls_0_0.csv")
    assert log_id is None
    assert topic is None


def test_infer_label_from_path_prefers_nearest_folder():
    assert infer_label_from_path("Live GPS Spoofing and Jamming/Benign Flight") == "benign"
    assert infer_label_from_path("Simulated - OTU Survey/PX4-H480-SITL/GPS Spoofing") == "gps_spoofing"
    assert infer_label_from_path("Live GPS Spoofing and Jamming/GPS Jamming") == "gps_jamming"


def _write_log_csvs(zf, folder: str, log_id: str, *, lat0: float) -> None:
    position = pd.DataFrame({
        "timestamp": [0, 100_000, 200_000],
        "lat": [lat0, lat0 + 0.0001, lat0 + 0.0002],
        "lon": [30.0, 30.0001, 30.0002],
        "alt": [50.0, 51.0, 52.0],
    })
    zf.writestr(f"{folder}/{log_id}_vehicle_global_position_0.csv", position.to_csv(index=False))
    attitude = pd.DataFrame({
        "timestamp": [0, 200_000],
        "q[0]": [1.0, 1.0], "q[1]": [0.0, 0.0], "q[2]": [0.0, 0.0], "q[3]": [0.0, 0.0],
    })
    zf.writestr(f"{folder}/{log_id}_vehicle_attitude_0.csv", attitude.to_csv(index=False))
    battery = pd.DataFrame({"timestamp": [0], "voltage_v": [16.0], "remaining": [0.9], "current_a": [5.0]})
    zf.writestr(f"{folder}/{log_id}_battery_status_0.csv", battery.to_csv(index=False))


def _make_uav_attack_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        _write_log_csvs(zf, "Simulated - OTU Survey/PX4-QUAD-SITL/Normal", "log_1_2020-1-1-00-00-00", lat0=39.0)
        _write_log_csvs(zf, "Simulated - OTU Survey/PX4-QUAD-SITL/GPS Spoofing", "log_2_2020-1-1-01-00-00", lat0=40.0)
    return buf.getvalue()


def test_parse_zip_bytes_produces_correct_logs_and_labels():
    data = _make_uav_attack_zip_bytes()

    df = parse_zip_bytes(data)

    assert set(df["source_id"].unique()) == {"log_1_2020-1-1-00-00-00", "log_2_2020-1-1-01-00-00"}
    benign_rows = df[df["source_id"] == "log_1_2020-1-1-00-00-00"]
    assert (benign_rows["label"] == "benign").all()
    spoofing_rows = df[df["source_id"] == "log_2_2020-1-1-01-00-00"]
    assert (spoofing_rows["label"] == "gps_spoofing").all()
    assert "roll_deg" in df.columns and "pitch_deg" in df.columns and "yaw_deg" in df.columns
    assert "voltage_v" in df.columns


def test_build_uav_attack_silver_downloads_from_bronze_and_adds_provenance(fake_minio_client):
    write_bronze_bytes(_make_uav_attack_zip_bytes(), "uav_attack/UAVAttackData.zip", client=fake_minio_client)

    silver = build_uav_attack_silver(fake_minio_client)

    assert not silver.empty
    assert set(silver["label"].unique()) == {"benign", "gps_spoofing"}
    assert (silver["_source_type"] == "uav_attack").all()
    assert (silver["_source_file"] == "uav_attack/UAVAttackData.zip").all()


def test_build_uav_attack_silver_returns_empty_when_no_zip_in_bronze(fake_minio_client):
    result = build_uav_attack_silver(fake_minio_client)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
