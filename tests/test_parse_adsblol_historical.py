"""Tests for src/silver/parse_adsblol_historical.py"""

from __future__ import annotations

import gzip
import io
import json
import tarfile

import pandas as pd
import pytest

from src.common.fakes import FakeMinioClient
from src.silver.parse_adsblol_historical import (
    SILVER_SCHEMA_VERSION,
    _parse_tar_fileobj,
    parse_trace_bytes,
)


def _make_tar_bytes(aircraft_records: list[dict]) -> bytes:
    """Build an in-memory tar with one gzip-compressed JSON per record."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i, rec in enumerate(aircraft_records):
            payload = gzip.compress(json.dumps(rec).encode("utf-8"))
            info = tarfile.TarInfo(name=f"traces/ab/{rec.get('icao', 'xx')}.json")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


SAMPLE_AIRCRAFT = {
    "icao": "abc123",
    "timestamp": 1_700_000_000,
    "r": "TC-ABC",
    "t": "B738",
    "trace": [
        [0, 40.0, 29.0, 35000, 450, 90, 0, 0, None, "adsb_icao", 35100, 0, None, None],
        [10, 40.1, 29.1, "ground", 0, 0, 0, 0, None, "adsb_icao", None, 0, None, None],
        [20, 40.2, 29.2, 36000, 460, 91, 0, -500, None, "adsb_icao", 36100, -500, 440, 2.5],
    ],
}


def test_parse_trace_bytes_unit_conversions():
    raw = gzip.compress(json.dumps(SAMPLE_AIRCRAFT).encode())
    df = parse_trace_bytes(raw)

    assert len(df) == 3

    # feet → metres: 35000 * 0.3048 = 10668.0
    assert df.loc[0, "alt"] == pytest.approx(35000 * 0.3048, abs=0.2)
    # on_ground row
    assert bool(df.loc[1, "on_ground"]) is True
    assert pd.isna(df.loc[1, "alt"])
    # ground_speed_ms: 450 knots → m/s
    assert df.loc[0, "ground_speed_ms"] == pytest.approx(450 * 0.5144, abs=0.01)
    # vertical_rate_ms: -500 fpm → m/s
    assert df.loc[2, "vertical_rate_ms"] == pytest.approx(-500 * 0.00508, abs=0.001)


def test_parse_trace_bytes_source_fields():
    raw = gzip.compress(json.dumps(SAMPLE_AIRCRAFT).encode())
    df = parse_trace_bytes(raw)

    assert (df["source_type"] == "adsblol_historical").all()
    assert (df["source_id"] == "abc123").all()
    assert (df["registration"] == "TC-ABC").all()
    assert (df["label"].isna()).all()


def test_parse_trace_bytes_timestamp():
    raw = gzip.compress(json.dumps(SAMPLE_AIRCRAFT).encode())
    df = parse_trace_bytes(raw)

    assert df.loc[0, "timestamp_utc"] == 1_700_000_000 + 0
    assert df.loc[2, "timestamp_utc"] == 1_700_000_000 + 20


def test_sparse_s2_updates_are_distinct_from_forward_filled_values():
    record = {
        "icao": "fresh1",
        "timestamp": 1_700_000_000,
        "trace": [
            [
                0, 40.0, 29.0, 10000, 200, 90, 0, 0,
                {
                    "squawk": "7700", "emergency": "general", "nic": 8,
                    "rc": 186, "nac_p": 9, "sil": 3, "version": 2,
                    "sil_type": "perhour", "sda": 2, "nac_v": 2,
                },
                "adsb_icao", 10100, 0, None, None,
            ],
            [
                10, 40.1, 29.1, 10010, 200, 90, 0, 0,
                {"nic": 7}, "adsb_icao", 10110, 0, None, None,
            ],
            [
                30, 40.2, 29.2, 10020, 200, 90, 2, 0,
                None, "adsb_icao", 10120, 0, None, None,
            ],
        ],
    }

    df = parse_trace_bytes(gzip.compress(json.dumps(record).encode()))

    # Values remain forward-filled, but only actual key presence is an update.
    assert df["squawk"].tolist() == ["7700", "7700", "7700"]
    assert df["squawk_updated"].tolist() == [True, False, False]
    assert df["squawk_update_timestamp_utc"].tolist() == [
        1_700_000_000,
        1_700_000_000,
        1_700_000_000,
    ]
    assert df["squawk_update_age_s"].tolist() == [0.0, 10.0, 30.0]

    assert df["nic"].tolist() == [8, 7, 7]
    assert df["nic_updated"].tolist() == [True, True, False]
    assert df["nic_update_timestamp_utc"].tolist() == [
        1_700_000_000,
        1_700_000_010,
        1_700_000_010,
    ]
    assert df["nic_update_age_s"].tolist() == [0.0, 0.0, 20.0]

    # A new-leg flag does not masquerade as a new S2 transmission.
    assert bool(df.loc[2, "flags_new_leg"]) is True
    assert bool(df.loc[2, "emergency_updated"]) is False
    assert df.loc[2, "emergency_update_age_s"] == 30.0

    for field in ("rc", "nac_p", "sil", "adsb_version", "sil_type", "sda", "nac_v"):
        assert bool(df.loc[0, f"{field}_updated"]) is True
        assert bool(df.loc[1, f"{field}_updated"]) is False
        assert df.loc[1, f"{field}_update_age_s"] == 10.0


def test_explicit_null_is_a_fresh_clear_not_an_absent_update():
    record = {
        "icao": "clear1",
        "timestamp": 1000,
        "trace": [
            [0, 1, 1, 1000, 100, 0, 0, 0, {"emergency": "general"}, "adsb_icao"],
            [5, 1, 1, 1000, 100, 0, 0, 0, {"emergency": None}, "adsb_icao"],
            [8, 1, 1, 1000, 100, 0, 0, 0, None, "adsb_icao"],
        ],
    }

    df = parse_trace_bytes(json.dumps(record).encode())

    assert df.loc[0, "emergency"] == "general"
    assert pd.isna(df.loc[1, "emergency"])
    assert pd.isna(df.loc[2, "emergency"])
    assert df["emergency_updated"].tolist() == [True, True, False]
    assert df["emergency_update_timestamp_utc"].tolist() == [1000.0, 1005.0, 1005.0]
    assert df["emergency_update_age_s"].tolist() == [0.0, 0.0, 3.0]


def test_s2_update_state_resets_at_each_trace_boundary():
    first = {
        "icao": "sameicao",
        "timestamp": 1000,
        "trace": [[0, 1, 1, 1000, 100, 0, 0, 0, {"squawk": "7600"}, "adsb_icao"]],
    }
    second = {
        "icao": "sameicao",
        "timestamp": 2000,
        "trace": [[0, 1, 1, 1000, 100, 0, 0, 0, None, "adsb_icao"]],
    }

    assert parse_trace_bytes(json.dumps(first).encode()).loc[0, "squawk"] == "7600"
    parsed_second = parse_trace_bytes(json.dumps(second).encode())
    assert pd.isna(parsed_second.loc[0, "squawk"])
    assert bool(parsed_second.loc[0, "squawk_updated"]) is False
    assert pd.isna(parsed_second.loc[0, "squawk_update_timestamp_utc"])
    assert pd.isna(parsed_second.loc[0, "squawk_update_age_s"])


def test_parse_tar_fileobj_writes_silver(fake_minio_client: FakeMinioClient):
    tar_bytes = _make_tar_bytes([SAMPLE_AIRCRAFT])
    uris = _parse_tar_fileobj(
        io.BytesIO(tar_bytes), "test.tar", batch_size=100, client=fake_minio_client
    )

    assert len(uris) == 1
    assert uris[0].startswith("s3://silver/")

    # Silver Parquet contains provenance
    stored = list(fake_minio_client.buckets["silver"].values())[0]
    df = pd.read_parquet(io.BytesIO(stored))
    assert "_source_type" in df.columns
    assert (df["_source_type"] == "adsblol_historical").all()
    assert (df["_source_file"] == "test.tar").all()
    assert (df["_schema_version"] == SILVER_SCHEMA_VERSION).all()


def test_parse_tar_handles_non_gzip_json(fake_minio_client: FakeMinioClient):
    """Members that are plain (non-gzip) JSON should also parse without error."""
    plain_rec = dict(SAMPLE_AIRCRAFT)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        payload = json.dumps(plain_rec).encode("utf-8")
        info = tarfile.TarInfo(name="traces/ab/plain.json")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    uris = _parse_tar_fileobj(
        io.BytesIO(buf.getvalue()), "plain.tar", batch_size=100, client=fake_minio_client
    )
    assert len(uris) == 1
