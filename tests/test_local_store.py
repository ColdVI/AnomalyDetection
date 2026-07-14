"""Tests for src/common/local_store.py (LocalObjectStoreClient).

ONEMLI: bu istemcinin daha once KENDI test dosyasi yoktu -- sadece
test_minio_io_delete.py'de delete_layer_objects()'in fake VE local
istemcide AYNI davrandigini dogrulayan TEK bir dolayli test vardi. Burada
ObjectStoreClient protokolunun HER metodu (bucket_exists/make_bucket/
put_object/get_object/remove_object/list_objects) dogrudan test ediliyor
-- gercek MinIO'ya STORAGE_BACKEND=local ile ihtiyac duymadan ayni arayuzu
saglayan bu sinifin GERCEKTEN dogru calistigindan emin olmak icin (bkz.
get_minio_client() docstring'i, .env.example)."""

from __future__ import annotations

import pytest

from src.common.local_store import LocalObjectStoreClient


def test_bucket_exists_false_before_creation(tmp_path):
    client = LocalObjectStoreClient(str(tmp_path))
    assert client.bucket_exists("bronze") is False


def test_make_bucket_then_bucket_exists(tmp_path):
    client = LocalObjectStoreClient(str(tmp_path))
    client.make_bucket("bronze")
    assert client.bucket_exists("bronze") is True


def test_make_bucket_is_idempotent(tmp_path):
    """Iki kez cagirmak hata FIRLATMAMALI (exist_ok=True) -- write_bronze_bytes
    gibi cagiranlar ensure_bucket() ile HER yazimda bunu cagiriyor."""
    client = LocalObjectStoreClient(str(tmp_path))
    client.make_bucket("bronze")
    client.make_bucket("bronze")
    assert client.bucket_exists("bronze") is True


def test_put_object_without_bucket_raises_runtime_error(tmp_path):
    client = LocalObjectStoreClient(str(tmp_path))
    with pytest.raises(RuntimeError):
        client.put_object("bronze", "alfa/x.parquet", b"data", length=4)


def test_put_and_get_object_round_trips_bytes(tmp_path):
    client = LocalObjectStoreClient(str(tmp_path))
    client.make_bucket("bronze")
    client.put_object("bronze", "alfa/x.parquet", b"hello-bytes", length=11)

    response = client.get_object("bronze", "alfa/x.parquet")

    assert response.read() == b"hello-bytes"


def test_put_object_accepts_file_like_data(tmp_path):
    """put_object'e hem duz bytes HEM DE io.BytesIO gibi .read() saglayan bir
    nesne verilebilmeli -- write_bronze_bytes() ikincisini kullaniyor."""
    import io

    client = LocalObjectStoreClient(str(tmp_path))
    client.make_bucket("bronze")
    client.put_object("bronze", "alfa/x.bin", io.BytesIO(b"stream-data"), length=11)

    assert client.get_object("bronze", "alfa/x.bin").read() == b"stream-data"


def test_put_object_creates_nested_directories(tmp_path):
    client = LocalObjectStoreClient(str(tmp_path))
    client.make_bucket("silver")
    client.put_object("silver", "adsblol_realtime/_landing/states-1.jsonl", b"{}", length=2)

    assert (tmp_path / "silver" / "adsblol_realtime" / "_landing" / "states-1.jsonl").exists()


def test_remove_object_deletes_file(tmp_path):
    client = LocalObjectStoreClient(str(tmp_path))
    client.make_bucket("bronze")
    client.put_object("bronze", "alfa/x.parquet", b"data", length=4)

    client.remove_object("bronze", "alfa/x.parquet")

    assert not (tmp_path / "bronze" / "alfa" / "x.parquet").exists()


def test_remove_object_missing_file_is_a_no_op(tmp_path):
    """Var olmayan bir objeyi silmeye calismak hata FIRLATMAMALI -- MinIO'nun
    kendi remove_object'i de var-olmayan anahtarlar icin sessizce basarili
    olur, delete_layer_objects()'in tekrar tekrar cagirilabilir olmasi buna
    dayanir."""
    client = LocalObjectStoreClient(str(tmp_path))
    client.make_bucket("bronze")
    client.remove_object("bronze", "does/not/exist.parquet")  # raise etmemeli


def test_list_objects_empty_bucket_returns_empty_list(tmp_path):
    client = LocalObjectStoreClient(str(tmp_path))
    client.make_bucket("bronze")
    assert client.list_objects("bronze") == []


def test_list_objects_nonexistent_bucket_returns_empty_list(tmp_path):
    client = LocalObjectStoreClient(str(tmp_path))
    assert client.list_objects("bronze") == []


def test_list_objects_returns_sorted_names_recursively(tmp_path):
    client = LocalObjectStoreClient(str(tmp_path))
    client.make_bucket("bronze")
    client.put_object("bronze", "alfa/b.parquet", b"x", length=1)
    client.put_object("bronze", "alfa/a.parquet", b"x", length=1)
    client.put_object("bronze", "uav_attack/c.parquet", b"x", length=1)

    names = [o.object_name for o in client.list_objects("bronze", recursive=True)]

    assert names == ["alfa/a.parquet", "alfa/b.parquet", "uav_attack/c.parquet"]


def test_list_objects_filters_by_prefix(tmp_path):
    client = LocalObjectStoreClient(str(tmp_path))
    client.make_bucket("bronze")
    client.put_object("bronze", "alfa/a.parquet", b"x", length=1)
    client.put_object("bronze", "uav_attack/b.parquet", b"x", length=1)

    names = [o.object_name for o in client.list_objects("bronze", prefix="alfa/")]

    assert names == ["alfa/a.parquet"]
