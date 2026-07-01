import io
import json

import pandas as pd

from src.ingestion.adsblol_consumer import flush_batch_to_bronze, land_batch_raw


def test_flush_batch_to_bronze_writes_every_row_with_provenance(fake_minio_client):
    batch = [
        {"hex": "tc001", "lat": 39.0, "lon": 35.0, "alt_baro": 35000},
        {"hex": "ab002", "lat": 51.0, "lon": -10.0, "alt_baro": 30000},
    ]

    uri = flush_batch_to_bronze(batch, source_file="uav.raw.states", client=fake_minio_client)

    assert uri is not None
    bucket, object_name = uri.removeprefix("s3://").split("/", 1)
    df = pd.read_parquet(io.BytesIO(fake_minio_client.buckets[bucket][object_name]))
    # No geo filter (ADR-003): both rows survive, not just the Turkey one.
    assert list(df["hex"]) == ["tc001", "ab002"]
    assert df.loc[0, "_source_type"] == "adsblol_rt"
    assert df.loc[0, "_source_file"] == "uav.raw.states"


def test_flush_batch_to_bronze_returns_none_for_empty_batch(fake_minio_client):
    uri = flush_batch_to_bronze([], source_file="uav.raw.states", client=fake_minio_client)
    assert uri is None


def test_land_batch_raw_writes_every_message_unfiltered(fake_minio_client):
    batch = [
        {"hex": "tc001", "lat": 39.0, "lon": 35.0},
        {"hex": "ab002", "lat": 51.0, "lon": -10.0},
    ]

    uri = land_batch_raw(batch, client=fake_minio_client)

    assert uri is not None
    bucket, object_name = uri.removeprefix("s3://").split("/", 1)
    assert object_name.startswith("adsblol_realtime/_landing/states-")
    lines = fake_minio_client.buckets[bucket][object_name].decode("utf-8").strip().split("\n")
    assert [json.loads(line)["hex"] for line in lines] == ["tc001", "ab002"]


def test_land_batch_raw_returns_none_for_empty_batch(fake_minio_client):
    assert land_batch_raw([], client=fake_minio_client) is None
