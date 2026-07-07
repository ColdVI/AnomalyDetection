import io

from src.common.fakes import FakeMinioClient
from src.common.local_store import LocalObjectStoreClient
from src.common.minio_io import delete_layer_objects, write_silver


def _put(client, bucket: str, name: str) -> None:
    client.make_bucket(bucket)
    client.put_object(bucket, name, io.BytesIO(b"x"), length=1)


def test_delete_layer_objects_removes_only_matching_prefix_fake(fake_minio_client):
    _put(fake_minio_client, "silver", "adsblol_historical/part-0.parquet")
    _put(fake_minio_client, "silver", "adsblol_historical/part-1.parquet")
    _put(fake_minio_client, "silver", "alfa/part-0.parquet")

    removed = delete_layer_objects(fake_minio_client, "silver", "adsblol_historical")

    assert removed == 2
    remaining = fake_minio_client.list_object_names("silver")
    assert remaining == ["alfa/part-0.parquet"]


def test_delete_layer_objects_removes_only_matching_prefix_local(tmp_path):
    client = LocalObjectStoreClient(str(tmp_path))
    _put(client, "silver", "adsblol_historical/part-0.parquet")
    _put(client, "silver", "adsblol_historical/part-1.parquet")
    _put(client, "silver", "alfa/part-0.parquet")

    removed = delete_layer_objects(client, "silver", "adsblol_historical")

    assert removed == 2
    remaining = [o.object_name for o in client.list_objects("silver", recursive=True)]
    assert remaining == ["alfa/part-0.parquet"]


def test_rerun_after_clear_does_not_accumulate_stale_parts(fake_minio_client):
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2]})
    write_silver(df, "adsblol_historical", client=fake_minio_client)
    write_silver(df, "adsblol_historical", client=fake_minio_client)
    assert len(fake_minio_client.list_object_names("silver")) == 2

    delete_layer_objects(fake_minio_client, "silver", "adsblol_historical")
    write_silver(df, "adsblol_historical", client=fake_minio_client)

    assert len(fake_minio_client.list_object_names("silver")) == 1
