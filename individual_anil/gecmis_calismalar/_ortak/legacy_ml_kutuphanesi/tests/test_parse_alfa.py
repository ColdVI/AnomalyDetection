import io
import zipfile

import pandas as pd

from src.common.minio_io import write_bronze_bytes
from src.silver.parse_alfa import build_alfa_silver, infer_fault_from_seq_name, parse_zip_bytes


def _make_alfa_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # Sequence 1: normal flight, no failure_status file.
        seq1 = "carbonZ_2020-01-01-00-00-00_no_failure"
        pos1 = pd.DataFrame({
            "%time": [0, 10_000_000, 20_000_000],
            "field.latitude": [40.0, 40.001, 40.002],
            "field.longitude": [29.0, 29.001, 29.002],
            "field.altitude": [100.0, 101.0, 102.0],
        })
        zf.writestr(f"processed/{seq1}/{seq1}-mavros-global_position-global.csv", pos1.to_csv(index=False))
        roll1 = pd.DataFrame({
            "%time": [0, 10_000_000, 20_000_000],
            "field.commanded": [1.0, 1.1, 1.2],
            "field.measured": [0.9, 1.0, 1.1],
        })
        zf.writestr(f"processed/{seq1}/{seq1}-mavros-nav_info-roll.csv", roll1.to_csv(index=False))

        # Sequence 2: engine failure, with a failure_status onset.
        seq2 = "carbonZ_2020-01-01-01-00-00_1_engine_failure"
        pos2 = pd.DataFrame({
            "%time": [0, 10_000_000, 20_000_000],
            "field.latitude": [41.0, 41.001, 41.002],
            "field.longitude": [30.0, 30.001, 30.002],
            "field.altitude": [200.0, 199.0, 198.0],
        })
        zf.writestr(f"processed/{seq2}/{seq2}-mavros-global_position-global.csv", pos2.to_csv(index=False))
        failure2 = pd.DataFrame({"%time": [10_000_000], "field.data": [1]})
        zf.writestr(f"processed/{seq2}/{seq2}-failure_status-engines.csv", failure2.to_csv(index=False))

    return buf.getvalue()


def test_infer_fault_from_seq_name_maps_known_labels():
    assert infer_fault_from_seq_name("carbonZ_x_no_failure") == "normal"
    assert infer_fault_from_seq_name("carbonZ_x_no_ground_truth") == "unknown"
    assert infer_fault_from_seq_name("carbonZ_x_1_engine_failure") == "engine_fault"
    assert infer_fault_from_seq_name("carbonZ_x_left_aileron__right_aileron__failure") == "aileron_fault"
    assert infer_fault_from_seq_name("carbonZ_x_1_rudder_zero__left_aileron_failure") == "aileron_rudder_fault"


def test_parse_zip_bytes_produces_correct_sequences_and_labels():
    data = _make_alfa_zip_bytes()

    df = parse_zip_bytes(data)

    assert set(df["source_id"].unique()) == {
        "carbonZ_2020-01-01-00-00-00_no_failure",
        "carbonZ_2020-01-01-01-00-00_1_engine_failure",
    }
    normal_rows = df[df["source_id"] == "carbonZ_2020-01-01-00-00-00_no_failure"]
    assert (normal_rows["label"] == "normal").all()
    assert "roll_measured" in df.columns
    assert "roll_commanded" in df.columns

    fault_rows = df[df["source_id"] == "carbonZ_2020-01-01-01-00-00_1_engine_failure"]
    assert (fault_rows["label"] == "engine_fault").all()
    assert (fault_rows["source_type"] == "alfa").all()


def test_build_alfa_silver_downloads_from_bronze_and_adds_provenance(fake_minio_client):
    write_bronze_bytes(_make_alfa_zip_bytes(), "alfa/processed.zip", client=fake_minio_client)

    silver = build_alfa_silver(fake_minio_client)

    assert not silver.empty
    assert set(silver["source_id"].unique()) == {
        "carbonZ_2020-01-01-00-00-00_no_failure",
        "carbonZ_2020-01-01-01-00-00_1_engine_failure",
    }
    assert (silver["_source_type"] == "alfa").all()
    assert (silver["_source_file"] == "alfa/processed.zip").all()
    assert (silver["_schema_version"] == "silver_v1").all()


def test_build_alfa_silver_returns_empty_when_no_zip_in_bronze(fake_minio_client):
    result = build_alfa_silver(fake_minio_client)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
