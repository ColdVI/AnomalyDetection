"""Tests for apply_realtime_retention() in src/common/minio_io.py."""

from src.common.fakes import FakeMinioClient
from src.common.minio_io import apply_realtime_retention


def test_apply_realtime_retention_sets_correct_prefix_and_days():
    client = FakeMinioClient()
    client.make_bucket("bronze")

    apply_realtime_retention(client, bucket="bronze")

    assert len(client.lifecycle_calls) == 1
    call = client.lifecycle_calls[0]
    assert call["bucket"] == "bronze"
    assert call["prefix"] == "adsblol_realtime/_landing/"
    assert call["days"] == 7


def test_apply_realtime_retention_only_one_rule_called():
    client = FakeMinioClient()
    client.make_bucket("bronze")

    apply_realtime_retention(client, bucket="bronze")

    # Exactly one call -- other prefixes (historical, alfa, uav_attack, uav_sead)
    # must NOT be touched.
    assert len(client.lifecycle_calls) == 1


def test_apply_realtime_retention_custom_days():
    client = FakeMinioClient()
    client.make_bucket("bronze")

    apply_realtime_retention(client, bucket="bronze", days=14)

    assert client.lifecycle_calls[0]["days"] == 14


def test_apply_realtime_retention_uses_default_bronze_bucket(monkeypatch):
    import os

    monkeypatch.setenv("MINIO_BRONZE_BUCKET", "my-bronze")
    client = FakeMinioClient()
    client.make_bucket("my-bronze")

    apply_realtime_retention(client)

    assert client.lifecycle_calls[0]["bucket"] == "my-bronze"
