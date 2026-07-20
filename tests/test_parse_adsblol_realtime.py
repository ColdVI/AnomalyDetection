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

# 2026-07-18: Dashboard/codes/uav_producer.py'nin GERCEKTEN Kafka'ya/Bronze'a
# yazdigi, normalize edilmis (_normalize_common) sema -- ham adsb.lol `ac`
# semasi (hex, alt_baro, gs, baro_rate, dbFlags...) DEGIL. Bkz. modul
# docstring'i (parse_adsblol_realtime.py) -- eski testler yanlis/hayali bir
# semayi test ediyordu.
SAMPLE_AC = {
    "icao24": "4b1234",
    "ts": "2026-07-01T12:00:00.500000+00:00",
    "cycle_id": 7,
    "signal_age_sec": 0.3,
    "source": "adsblol",
    "lat": 41.0,
    "lon": 28.5,
    "alt": 10668.0,
    "velocity": 231.5,
    "track": 90.5,
    "vertical_rate": -2.54,
    "is_ground": False,
    "is_military": False,
    "callsign": "THY123  ",
    "category": "A3",
    "squawk": "2112",
    "emergency": None,
}

GROUND_AC = {
    "icao24": "4bffff",
    "ts": "2026-07-01T12:00:01.000000+00:00",
    "cycle_id": 7,
    "signal_age_sec": 0.5,
    "source": "adsblol",
    "lat": 40.0,
    "lon": 29.0,
    "alt": 0.0,
    "velocity": 0.0,
    "track": 180.0,
    "vertical_rate": 0.0,
    "is_ground": True,
    "is_military": False,
    "callsign": None,
    "category": None,
    "squawk": None,
    "emergency": None,
}


def test_parse_ac_record_fields_pass_through_without_reconversion():
    """alt/velocity/vertical_rate zaten SI birimlerinde geliyor (uav_producer.py
    ceviriyor) -- Silver bunlari TEKRAR donusturmemeli, oldugu gibi almali."""
    rec = _parse_ac_record(SAMPLE_AC, batch_ts=1_700_000_000.0)

    assert rec["source_type"] == "adsblol_realtime"
    assert rec["source_id"] == "4b1234"
    assert rec["on_ground"] is False
    assert rec["alt"] == 10668.0
    assert rec["ground_speed_ms"] == 231.5
    assert rec["vertical_rate_ms"] == -2.54
    assert rec["track_deg"] == 90.5
    assert rec["flight_callsign"] == "THY123"
    assert rec["is_military"] is False
    assert rec["label"] is None
    assert rec["signal_age_sec"] == 0.3
    assert rec["source"] == "adsblol"
    assert rec["cycle_id"] == 7


def test_parse_ac_record_on_ground():
    rec = _parse_ac_record(GROUND_AC, batch_ts=None)
    assert rec["on_ground"] is True
    assert rec["flight_callsign"] is None


def test_parse_ac_record_prefers_precise_record_ts_over_batch_ts():
    rec = _parse_ac_record(SAMPLE_AC, batch_ts=1_700_000_000.0)
    from datetime import datetime

    expected = datetime.fromisoformat(SAMPLE_AC["ts"]).timestamp()
    assert rec["timestamp_utc"] == pytest.approx(expected)
    assert rec["timestamp_utc"] != 1_700_000_000.0


def test_parse_ac_record_falls_back_to_batch_ts_when_ts_missing():
    rec_no_ts = dict(SAMPLE_AC)
    rec_no_ts.pop("ts")
    rec = _parse_ac_record(rec_no_ts, batch_ts=1_700_000_000.0)
    assert rec["timestamp_utc"] == 1_700_000_000.0


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
    assert (df["source_type"] == "adsblol_realtime").all()


@pytest.mark.parametrize("is_military,expected", [
    (True, True),
    (False, False),
])
def test_parse_ac_record_is_military_passthrough(is_military, expected):
    rec = dict(SAMPLE_AC)
    rec["is_military"] = is_military
    assert _parse_ac_record(rec, batch_ts=None)["is_military"] == expected


def test_parse_ac_record_is_military_defaults_false_when_missing():
    rec = dict(SAMPLE_AC)
    rec.pop("is_military", None)
    assert _parse_ac_record(rec, batch_ts=None)["is_military"] is False


def test_parse_jsonl_bytes_skips_malformed_lines_without_crashing():
    lines = [json.dumps(SAMPLE_AC), "{not valid json", json.dumps(GROUND_AC)]
    raw = "\n".join(lines).encode("utf-8")

    df = parse_jsonl_bytes(raw, "states-20260701T120000Z.jsonl")

    assert len(df) == 2
    assert set(df["source_id"]) == {"4b1234", "4bffff"}


def test_parse_jsonl_bytes_all_malformed_returns_empty_dataframe():
    raw = b"{not json\nalso not json"
    df = parse_jsonl_bytes(raw, "states-20260701T120000Z.jsonl")
    assert df.empty


def test_run_returns_empty_list_when_no_jsonl_objects(fake_minio_client: FakeMinioClient):
    fake_minio_client.make_bucket("bronze")
    assert run(client=fake_minio_client, bronze_bucket="bronze") == []


def test_run_deletes_processed_bronze_files_after_writing_silver(fake_minio_client: FakeMinioClient):
    """2026-07-09 karari (bkz. modul docstring'i): Silver'a basariyla yazilan
    Bronze JSONL'leri SILINMELI -- aksi halde run() tekrar cagirildiginda
    ayni dosyalar ikinci kez islenip Silver'da kopya satir uretir."""
    lines = json.dumps(SAMPLE_AC).encode("utf-8") + b"\n"
    fake_minio_client.make_bucket("bronze")
    object_name = "adsblol_realtime/_landing/states-20260701T120000Z.jsonl"
    fake_minio_client.put_object(
        "bronze", object_name, io.BytesIO(lines), length=len(lines),
        content_type="application/x-ndjson",
    )

    run(client=fake_minio_client, bronze_bucket="bronze")

    assert object_name not in fake_minio_client.buckets["bronze"]


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
    assert (df["_source_type"] == "adsblol_realtime").all()
    assert set(df["source_id"]) == {"4b1234", "4bffff"}
