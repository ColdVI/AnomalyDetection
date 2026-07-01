import io

import pandas as pd
import pytest

from src.common.io import write_bronze, write_bronze_bytes


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
