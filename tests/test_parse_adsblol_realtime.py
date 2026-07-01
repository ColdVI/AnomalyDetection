"""Tests for src/silver/parse_adsblol_realtime.py"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from src.common.fakes import FakeMinioClient
from src.silver.parse_adsblol_realtime import (
    _batch_timestamp,
    _parse_ac_record,
    parse_jsonl_bytes,
    run,
)

SAMPLE_AC = {
    "hex": "4b1234",
    "lat": 41.0,
    "lon": 28.5,
    "alt_baro": 35000,
    "alt_geom": 35100,
    "gs": 450,
    "track": 90.5,
    "baro_rate": -500,
    "geom_rate": -480,
    "ias": 440,
    "tas": 455,
    "roll": 2.1,
    "flight": "THY123  ",
    "r": "TC-ABC",
    "t": "B738",
    "category": "A3",
    "squawk": "2112",
    "nic": 8,
    "seen": 1.2,
    "seen_pos": 0.8,
}

GROUND_AC = {
    "hex": "4bffff",
    "lat": 40.0,
    "lon": 29.0,
    "alt_baro": "ground",
    "gs": 0,
    "track": 180,
    "baro_rate": 0,
}


def test_parse_ac_record_unit_conversions():
    rec = _parse_ac_record(SAMPLE_AC, batch_ts=1_700_000_000.0)

    assert rec["source_type"] == "adsblol_rt"
    assert rec["source_id"] == "4b1234"
    assert rec["on_ground"] is False
    assert rec["alt"] == pytest.approx(35000 * 0.3048, abs=0.2)
    assert rec["alt_geom_m"] == pytest.approx(35100 * 0.3048, abs=0.2)
    assert rec["ground_speed_ms"] == pytest.approx(450 * 0.5144, abs=0.01)
    assert rec["vertical_rate_ms"] == pytest.approx(-500 * 0.00508, abs=0.001)
    assert rec["indicated_airspeed_ms"] == pytest.approx(440 * 0.5144, abs=0.01)
    assert rec["flight_callsign"] == "THY123"
    assert rec["label"] is None


def test_parse_ac_record_on_ground():
    rec = _parse_ac_record(GROUND_AC, batch_ts=None)
    assert rec["on_ground"] is True
    assert rec["alt"] is None


def test_batch_timestamp_extraction():
    ts = _batch_timestamp("adsblol_realtime/_landing/states-20260701T120000Z.jsonl")
    assert ts == pytest.approx(1_782_907_200.0, abs=2.0)


def test_batch_timestamp_no_match():
    assert _batch_timestamp("adsblol_realtime/_landing/some_other_name.jsonl") is None


def test_parse_jsonl_bytes():
    lines = [json.dumps(SAMPLE_AC), json.dumps(GROUND_AC), ""]
    raw = "\n".join(lines).encode("utf-8")
    df = parse_jsonl_bytes(raw, "states-20260701T120000Z.jsonl")

    assert len(df) == 2
    assert set(df["source_id"]) == {"4b1234", "4bffff"}
    assert (df["source_type"] == "adsblol_rt").all()


def test_run_writes_silver(fake_minio_client: FakeMinioClient):
    # Seed Bronze with a JSONL file
    lines = json.dumps(SAMPLE_AC).encode("utf-8") + b"\n" + json.dumps(GROUND_AC).encode("utf-8") + b"\n"
    fake_minio_client.make_bucket("bronze")
    fake_minio_client.put_object(
        "bronze", "adsblol_realtime/_landing/states-20260701T120000Z.jsonl",
        io.BytesIO(lines), length=len(lines), content_type="application/x-ndjson",
    )

    uris = run(client=fake_minio_client, bronze_bucket="bronze")

    assert len(uris) == 1
    assert uris[0].startswith("s3://silver/")

    stored = list(fake_minio_client.buckets["silver"].values())[0]
    df = pd.read_parquet(io.BytesIO(stored))
    assert len(df) == 2
    assert "_source_type" in df.columns
    assert (df["_source_type"] == "adsblol_rt").all()
