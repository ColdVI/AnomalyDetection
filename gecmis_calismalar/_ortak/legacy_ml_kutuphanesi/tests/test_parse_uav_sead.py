"""Tests for src/silver/parse_uav_sead.py."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pandas as pd

from src.common.fakes import FakeMinioClient
from src.common.minio_io import write_bronze_bytes
from src.silver.parse_uav_sead import SOURCE_TYPE, build_uav_sead_silver


def _make_fake_flight_df(source_id: str = "flight_001", label: str = "normal") -> pd.DataFrame:
    """Minimal DataFrame as if parse_ulg_bytes had succeeded."""
    return pd.DataFrame(
        {
            "timestamp": [0, 100_000, 200_000],
            "lat": [39.0, 39.001, 39.002],
            "lon": [32.0, 32.001, 32.002],
            "alt": [100.0, 101.0, 102.0],
            "vel_m_s": [5.0, 5.1, 5.2],
            "yaw_deg": [90.0, 91.0, 92.0],
            "vertical_rate_mps": [0.0, 0.1, -0.1],
            "source_type": [SOURCE_TYPE] * 3,
            "source_id": [source_id] * 3,
            "label": [label] * 3,
            "timestamp_utc": [0.0, 0.1, 0.2],
            "timestamp_is_real_utc": [False] * 3,
        }
    )


def _seed_bronze(client: FakeMinioClient, flights: dict[str, str]) -> None:
    """Write labels.json + placeholder .ulg files to bronze."""
    labels = {
        flight_id: {
            "object_name": f"uav_sead/{flight_id}.ulg",
            "label": label,
        }
        for flight_id, label in flights.items()
    }
    labels_bytes = json.dumps(labels).encode("utf-8")
    client.make_bucket("bronze")
    client.put_object(
        "bronze", "uav_sead/labels.json",
        io.BytesIO(labels_bytes), length=len(labels_bytes),
    )
    for flight_id in flights:
        stub = b"fake-ulg-bytes"
        client.put_object(
            "bronze", f"uav_sead/{flight_id}.ulg",
            io.BytesIO(stub), length=len(stub),
        )


def test_build_uav_sead_silver_adds_provenance(fake_minio_client: FakeMinioClient):
    _seed_bronze(fake_minio_client, {"flight_001": "normal"})

    with patch(
        "src.silver.parse_uav_sead.parse_ulg_bytes",
        return_value=_make_fake_flight_df("flight_001", "normal"),
    ):
        silver = build_uav_sead_silver(fake_minio_client)

    assert not silver.empty
    assert (silver["_source_type"] == SOURCE_TYPE).all()
    assert "_source_file" in silver.columns


def test_build_uav_sead_silver_source_type_column(fake_minio_client: FakeMinioClient):
    _seed_bronze(fake_minio_client, {"flight_001": "jamming"})

    with patch(
        "src.silver.parse_uav_sead.parse_ulg_bytes",
        return_value=_make_fake_flight_df("flight_001", "jamming"),
    ):
        silver = build_uav_sead_silver(fake_minio_client)

    assert (silver["source_type"] == "uav_sead").all()
    assert (silver["label"] == "jamming").all()


def test_build_uav_sead_silver_multiple_flights(fake_minio_client: FakeMinioClient):
    _seed_bronze(fake_minio_client, {"f1": "normal", "f2": "spoofing"})

    def _fake_parse(data, source_id, label):
        return _make_fake_flight_df(source_id, label)

    with patch("src.silver.parse_uav_sead.parse_ulg_bytes", side_effect=_fake_parse):
        silver = build_uav_sead_silver(fake_minio_client)

    assert set(silver["source_id"].unique()) == {"f1", "f2"}
    assert len(silver) == 6  # 3 rows per flight


def test_build_uav_sead_silver_returns_empty_when_no_labels_json(
    fake_minio_client: FakeMinioClient,
):
    result = build_uav_sead_silver(fake_minio_client)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
