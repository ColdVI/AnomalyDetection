"""Tests for src/silver/parse_adsblol_historical.py"""

from __future__ import annotations

import gzip
import io
import json
import tarfile

import pandas as pd
import pytest

from src.common.fakes import FakeMinioClient
from src.silver.parse_adsblol_historical import parse_trace_bytes, _parse_tar_fileobj


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

    assert (df["source_type"] == "adsblol_hist").all()
    assert (df["source_id"] == "abc123").all()
    assert (df["registration"] == "TC-ABC").all()
    assert (df["label"].isna()).all()


def test_parse_trace_bytes_timestamp():
    raw = gzip.compress(json.dumps(SAMPLE_AIRCRAFT).encode())
    df = parse_trace_bytes(raw)

    assert df.loc[0, "timestamp_utc"] == 1_700_000_000 + 0
    assert df.loc[2, "timestamp_utc"] == 1_700_000_000 + 20


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
    assert (df["_source_type"] == "adsblol_hist").all()
    assert (df["_source_file"] == "test.tar").all()


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
