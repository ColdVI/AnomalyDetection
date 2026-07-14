"""Tests for src/ingestion/upload_raw.py (Bronze raw upload).

ONEMLI: bu dosya onceden test_loaders.py ile bolunmustu -- ikisi de
upload_raw_file'i neredeyse ayni senaryolarla (byte-for-byte kopyalama,
dosya adi korunumu) test ediyordu. Birlesirken merge_tar_parts testleri
(sadece test_loaders.py'de vardi) buraya tasindi, upload_raw_file'in
CAKISAN iki testinden biri (filename-only kontrolu) elendi -- ikisi de
"dosya adi/byte'lar aynen korunuyor mu" sorusuna cevap veriyordu, aralarinda
GERCEK bir fark yoktu."""

from __future__ import annotations

import pytest

from src.ingestion.upload_raw import merge_tar_parts, upload_raw_file


def test_upload_raw_file_preserves_bytes_and_original_filename(tmp_path, fake_minio_client):
    local_file = tmp_path / "processed.zip"
    local_file.write_bytes(b"PK\x03\x04fake-zip-bytes")

    uri = upload_raw_file(local_file, "alfa", client=fake_minio_client)

    assert uri == "s3://bronze/alfa/processed.zip"
    assert fake_minio_client.buckets["bronze"]["alfa/processed.zip"] == b"PK\x03\x04fake-zip-bytes"


def test_upload_raw_file_does_not_alter_binary_content(tmp_path, fake_minio_client):
    """Ayri bir test olarak kaliyor cunku ONCEKINDEN farkli bir riski
    kapsiyor: gecerli bir UTF-8 metni degil, rastgele binary (non-printable/
    non-UTF8) byte dizisi -- yanlislikla bir metin kodlamasi/decode adimi
    eklenirse (str'ye cevirip geri byte'a donmek gibi) SADECE bu test
    yakalar, yukaridaki ASCII-agirlikli test yakalamaz."""
    local_file = tmp_path / "UAVAttackData.zip"
    payload = b"\x00\x01\x02binary-content\xff"
    local_file.write_bytes(payload)

    upload_raw_file(local_file, "uav_attack", client=fake_minio_client)

    assert fake_minio_client.buckets["bronze"]["uav_attack/UAVAttackData.zip"] == payload


def test_merge_tar_parts_concatenates_in_order(tmp_path):
    base = tmp_path / "release"
    (tmp_path / "release.tar.aa").write_bytes(b"AAAA")
    (tmp_path / "release.tar.ab").write_bytes(b"BBBB")

    merged = merge_tar_parts(base)

    assert merged == tmp_path / "release.tar"
    assert merged.read_bytes() == b"AAAABBBB"


def test_merge_tar_parts_noop_if_merged_exists(tmp_path):
    """Zaten birlestirilmis bir .tar varsa, parcalar hala diskte olsa bile
    TEKRAR birlestirilmemeli (idempotent -- ayni komutu iki kez calistirmak
    guvenli olmali)."""
    base = tmp_path / "release"
    (tmp_path / "release.tar").write_bytes(b"EXISTING")
    (tmp_path / "release.tar.aa").write_bytes(b"AAAA")

    merged = merge_tar_parts(base)

    assert merged.read_bytes() == b"EXISTING"


def test_merge_tar_parts_raises_when_no_parts_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        merge_tar_parts(tmp_path / "missing")
