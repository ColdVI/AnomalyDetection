from src.ingestion.upload_raw import upload_raw_file


def test_upload_raw_file_preserves_bytes_and_original_filename(tmp_path, fake_minio_client):
    local_file = tmp_path / "processed.zip"
    local_file.write_bytes(b"PK\x03\x04fake-zip-bytes")

    uri = upload_raw_file(local_file, "alfa", client=fake_minio_client)

    assert uri == "s3://bronze/alfa/processed.zip"
    assert fake_minio_client.buckets["bronze"]["alfa/processed.zip"] == b"PK\x03\x04fake-zip-bytes"


def test_upload_raw_file_does_not_alter_content(tmp_path, fake_minio_client):
    local_file = tmp_path / "UAVAttackData.zip"
    payload = b"\x00\x01\x02binary-content\xff"
    local_file.write_bytes(payload)

    upload_raw_file(local_file, "uav_attack", client=fake_minio_client)

    assert fake_minio_client.buckets["bronze"]["uav_attack/UAVAttackData.zip"] == payload
