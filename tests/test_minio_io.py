import io

import pandas as pd
import pytest

from src.common.minio_io import (
    download_raw_bytes,
    list_layer_objects,
    read_layer,
    read_parquet_object,
    write_bronze,
    write_bronze_bytes,
    write_gold,
    write_silver,
)


def test_write_bronze_round_trip(fake_minio_client):
    source = pd.DataFrame({"raw_value": [1, 2]})

    uri = write_bronze(source, "alfa", "flight=demo", client=fake_minio_client)

    assert uri.startswith("s3://bronze/alfa/flight=demo/part-")
    assert uri.endswith(".parquet")
    bucket, object_name = uri.removeprefix("s3://").split("/", 1)
    written_bytes = fake_minio_client.buckets[bucket][object_name]
    round_tripped = pd.read_parquet(io.BytesIO(written_bytes))
    assert round_tripped.equals(source)


def test_write_bronze_without_partition(fake_minio_client):
    uri = write_bronze(pd.DataFrame({"x": [1]}), "uav_attack", client=fake_minio_client)
    assert uri.startswith("s3://bronze/uav_attack/part-")


def test_write_bronze_creates_bucket_if_missing(fake_minio_client):
    assert not fake_minio_client.bucket_exists("bronze")
    write_bronze(pd.DataFrame({"x": [1]}), "alfa", client=fake_minio_client)
    assert fake_minio_client.bucket_exists("bronze")


@pytest.mark.parametrize("unsafe", ["../alfa", "a/b", "", "with space"])
def test_write_bronze_rejects_unsafe_path_components(fake_minio_client, unsafe):
    with pytest.raises(ValueError):
        write_bronze(pd.DataFrame(), unsafe, client=fake_minio_client)


def test_write_bronze_bytes_round_trip(fake_minio_client):
    uri = write_bronze_bytes(b'{"hex": "tc001"}\n', "adsblol_realtime/_landing/states-1.jsonl", client=fake_minio_client)
    assert uri == "s3://bronze/adsblol_realtime/_landing/states-1.jsonl"
    assert fake_minio_client.buckets["bronze"]["adsblol_realtime/_landing/states-1.jsonl"] == b'{"hex": "tc001"}\n'


def test_write_bronze_bytes_preserves_caller_chosen_object_name(fake_minio_client):
    """Bronze = raw upload (ADR-003): callers keep the original filename, no random part-name."""
    uri = write_bronze_bytes(b"PK\x03\x04fake-zip-bytes", "alfa/ALFA.zip", client=fake_minio_client)
    assert uri == "s3://bronze/alfa/ALFA.zip"


def test_download_raw_bytes_round_trips_write_bronze_bytes(fake_minio_client):
    payload = b"PK\x03\x04fake-zip-bytes"
    write_bronze_bytes(payload, "alfa/ALFA.zip", client=fake_minio_client)

    downloaded = download_raw_bytes(fake_minio_client, "alfa/ALFA.zip")

    assert downloaded == payload


def test_write_silver_and_gold_use_their_own_default_buckets(fake_minio_client):
    silver_uri = write_silver(pd.DataFrame({"x": [1]}), "alfa", client=fake_minio_client)
    gold_uri = write_gold(pd.DataFrame({"x": [1]}), "common_uav_events", client=fake_minio_client)

    assert silver_uri.startswith("s3://silver/alfa/part-")
    assert gold_uri.startswith("s3://gold/common_uav_events/part-")


def test_list_layer_objects_filters_by_source_type_prefix(fake_minio_client):
    write_bronze(pd.DataFrame({"x": [1]}), "alfa", client=fake_minio_client)
    write_bronze(pd.DataFrame({"x": [2]}), "alfa", client=fake_minio_client)
    write_bronze(pd.DataFrame({"x": [3]}), "uav_attack", client=fake_minio_client)

    alfa_objects = list_layer_objects(fake_minio_client, "bronze", "alfa")

    assert len(alfa_objects) == 2
    assert all(name.startswith("alfa/") for name in alfa_objects)


def test_read_parquet_object_round_trips_write_bronze(fake_minio_client):
    source = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    uri = write_bronze(source, "alfa", client=fake_minio_client)
    object_name = uri.removeprefix("s3://bronze/")

    read_back = read_parquet_object(fake_minio_client, "bronze", object_name)

    assert read_back.equals(source)


def test_read_layer_concatenates_all_objects_for_source_type(fake_minio_client):
    write_bronze(pd.DataFrame({"a": [1], "b": ["only-in-first"]}), "alfa", client=fake_minio_client)
    write_bronze(pd.DataFrame({"a": [2], "c": ["only-in-second"]}), "alfa", client=fake_minio_client)
    write_bronze(pd.DataFrame({"a": [99]}), "uav_attack", client=fake_minio_client)

    combined = read_layer(fake_minio_client, "bronze", "alfa")

    assert len(combined) == 2
    assert set(combined.columns) == {"a", "b", "c"}
    assert sorted(combined["a"].tolist()) == [1, 2]


def test_read_layer_returns_empty_dataframe_when_nothing_written(fake_minio_client):
    result = read_layer(fake_minio_client, "silver", "alfa")

    assert isinstance(result, pd.DataFrame)
    assert result.empty
