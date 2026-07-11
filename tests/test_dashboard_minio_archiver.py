"""Dashboard/minio_archiver.py -- Kafka adsb.flights -> MinIO Bronze
arşivleyici testleri."""

from __future__ import annotations

from Dashboard import minio_archiver as archiver
from dashboard_fakes import FakeMinio


# ------------------------------------------------------------------------ flush --

def test_flush_writes_lines_joined_by_newline_as_ndjson():
    client = FakeMinio()
    client.make_bucket("bronze")
    archiver.flush(client, "bronze", ['{"a":1}', '{"a":2}', '{"a":3}'])

    assert len(client.put_calls) == 1
    call = client.put_calls[0]
    assert call["data"] == b'{"a":1}\n{"a":2}\n{"a":3}'
    assert call["content_type"] == "application/x-ndjson"


def test_flush_object_name_uses_bronze_prefix_and_jsonl_extension():
    client = FakeMinio()
    client.make_bucket("bronze")
    archiver.flush(client, "bronze", ["line"])

    object_name = client.put_calls[0]["object_name"]
    assert object_name.startswith(archiver.BRONZE_PREFIX)
    assert object_name.startswith("adsblol_realtime/_landing/states-")
    assert object_name.endswith(".jsonl")


def test_flush_object_name_timestamp_is_utc_and_filesystem_safe():
    """':' gibi dosya adinda sorun cikarabilecek karakterler OLMAMALI --
    format 'YYYYMMDDTHHMMSSZ' (bkz. flush() gövdesi)."""
    client = FakeMinio()
    client.make_bucket("bronze")
    archiver.flush(client, "bronze", ["line"])

    object_name = client.put_calls[0]["object_name"]
    ts_part = object_name[len(archiver.BRONZE_PREFIX) + len("states-"):-len(".jsonl")]
    assert ":" not in ts_part
    assert ts_part.endswith("Z")
    assert "T" in ts_part


def test_flush_single_message_still_produces_valid_object():
    client = FakeMinio()
    client.make_bucket("bronze")
    archiver.flush(client, "bronze", ["only-one-message"])
    assert client.put_calls[0]["data"] == b"only-one-message"


# ---------------------------------------------------- remove_lifecycle_if_present --

def test_remove_lifecycle_if_present_calls_delete_bucket_lifecycle():
    client = FakeMinio()
    archiver.remove_lifecycle_if_present(client, "bronze")
    assert client.lifecycle_delete_calls == ["bronze"]


def test_remove_lifecycle_if_present_swallows_exceptions():
    """2026-07-09 karari: MinIO'da otomatik silme kurali OLMAMALI --
    kural zaten yoksa (ilk kurulum, ya da daha once kaldirilmis) MinIO
    bir hata dondurebilir, bu ASLA main()'i cokertmemeli (bkz. fonksiyon
    docstring'i)."""
    client = FakeMinio()
    client.raise_on_delete_lifecycle = RuntimeError("kural zaten yok")
    archiver.remove_lifecycle_if_present(client, "bronze")  # exception firlatmamali


# ------------------------------------------------------------------- get_minio --

def test_get_minio_uses_env_var_overrides(monkeypatch):
    captured = {}

    class FakeMinioCtor:
        def __init__(self, endpoint, access_key, secret_key, secure):
            captured.update(endpoint=endpoint, access_key=access_key,
                            secret_key=secret_key, secure=secure)

    monkeypatch.setattr(archiver, "Minio", FakeMinioCtor)
    monkeypatch.setenv("MINIO_ENDPOINT", "minio.internal:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "custom-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "custom-secret")

    archiver.get_minio()

    assert captured == {
        "endpoint": "minio.internal:9000", "access_key": "custom-access",
        "secret_key": "custom-secret", "secure": False,
    }


def test_get_minio_defaults_when_env_vars_missing(monkeypatch):
    captured = {}

    class FakeMinioCtor:
        def __init__(self, endpoint, access_key, secret_key, secure):
            captured.update(endpoint=endpoint, access_key=access_key,
                            secret_key=secret_key, secure=secure)

    monkeypatch.setattr(archiver, "Minio", FakeMinioCtor)
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)

    archiver.get_minio()

    assert captured["endpoint"] == "localhost:9000"
    assert captured["access_key"] == "minioadmin"
    assert captured["secure"] is False


# ------------------------------------------------------------------- sabitler --

def test_batch_size_and_flush_secs_are_positive():
    assert archiver.BATCH_SIZE > 0
    assert archiver.FLUSH_SECS > 0


def test_dashboard_consumer_and_archiver_use_different_kafka_group_ids():
    """dashboard_consumer.py ile AYNI mesajlari BAGIMSIZ okumali (modul
    docstring'i) -- Kafka'da farkli group.id, ayni mesaji IKI kez okumayi
    (consumer group'un ayni mesaji paylasmasi yerine) garanti eder."""
    from Dashboard import dashboard_consumer
    assert archiver.GROUP_ID != "dashboard-consumer"
    assert archiver.GROUP_ID == "minio-archiver"
