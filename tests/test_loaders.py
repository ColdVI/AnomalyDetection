"""Tests for src/ingestion/upload_raw.py (Bronze raw upload)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from src.ingestion.upload_raw import merge_tar_parts, upload_raw_file


def test_upload_raw_file_stores_bytes_unchanged(tmp_path, fake_minio_client):
    data = b"raw bytes here"
    f = tmp_path / "dataset.zip"
    f.write_bytes(data)

    uri = upload_raw_file(f, "alfa", client=fake_minio_client)

    assert uri == "s3://bronze/alfa/dataset.zip"
    stored = fake_minio_client.buckets["bronze"]["alfa/dataset.zip"]
    assert stored == data


def test_upload_raw_file_preserves_filename(tmp_path, fake_minio_client):
    f = tmp_path / "UAVAttackData.zip"
    f.write_bytes(b"content")
    uri = upload_raw_file(f, "uav_attack", client=fake_minio_client)
    assert "uav_attack/UAVAttackData.zip" in uri


def test_merge_tar_parts_concatenates_in_order(tmp_path):
    base = tmp_path / "release"
    (tmp_path / "release.tar.aa").write_bytes(b"AAAA")
    (tmp_path / "release.tar.ab").write_bytes(b"BBBB")

    merged = merge_tar_parts(base)

    assert merged == tmp_path / "release.tar"
    assert merged.read_bytes() == b"AAAABBBB"


def test_merge_tar_parts_noop_if_merged_exists(tmp_path):
    base = tmp_path / "release"
    (tmp_path / "release.tar").write_bytes(b"EXISTING")
    (tmp_path / "release.tar.aa").write_bytes(b"AAAA")

    merged = merge_tar_parts(base)

    assert merged.read_bytes() == b"EXISTING"


def test_merge_tar_parts_raises_when_no_parts_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        merge_tar_parts(tmp_path / "missing")
