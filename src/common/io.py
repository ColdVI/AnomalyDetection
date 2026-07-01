"""Bronze output: every Bronze artifact is written to MinIO, never local disk.

ADR-002 (2026-06-30): the team decided everything (Bronze/Silver/Gold) lives
in MinIO (S3-compatible object storage), matching the architecture diagram
in docs/ (not local Parquet files). This module owns the MinIO client and
the one write path every Bronze loader must use.

The MinIO client is injectable so loaders/tests never need a running MinIO
server: pass `client=<fake or real Minio instance>` to skip env-based
client construction entirely.
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

import pandas as pd
from minio import Minio

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.=-]*$")
DEFAULT_BUCKET = "bronze"


class ObjectStoreClient(Protocol):
    """The subset of the Minio client surface our code depends on.

    Any object satisfying this (real `minio.Minio`, or a test fake) works
    with `write_bronze`/`write_bronze_bytes`.
    """

    def bucket_exists(self, bucket_name: str) -> bool: ...

    def make_bucket(self, bucket_name: str) -> None: ...

    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data,
        length: int,
        content_type: str = "application/octet-stream",
    ): ...


def _validate_component(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{name} must be a non-empty, filesystem/object-key-safe name; got {value!r}"
        )
    return value


def get_minio_client() -> Minio:
    """Build a Minio client from MINIO_* env vars (see .env.example)."""
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure = os.getenv("MINIO_SECURE", "false").strip().lower() == "true"
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def ensure_bucket(client: ObjectStoreClient, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def _object_name(source_type: str, partition: str | None) -> str:
    source = _validate_component(source_type, "source_type")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"part-{timestamp}-{uuid4().hex[:8]}.parquet"
    if partition is not None:
        partition = _validate_component(partition, "partition")
        return f"{source}/{partition}/{filename}"
    return f"{source}/{filename}"


def write_bronze_bytes(
    data: bytes,
    object_name: str,
    *,
    bucket: str | None = None,
    content_type: str = "application/octet-stream",
    client: ObjectStoreClient | None = None,
) -> str:
    """Upload raw bytes to MinIO and return the `s3://bucket/key` URI.

    Used by the realtime consumer for the raw JSONL landing objects, where
    there's no DataFrame to serialize. `bucket` defaults to the
    MINIO_BRONZE_BUCKET env var (falling back to "bronze"), resolved at call
    time so tests/scripts can rely on .env being loaded by the caller first.
    """
    bucket = bucket or os.getenv("MINIO_BRONZE_BUCKET", DEFAULT_BUCKET)
    client = client or get_minio_client()
    ensure_bucket(client, bucket)
    client.put_object(
        bucket,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return f"s3://{bucket}/{object_name}"


def write_bronze(
    df: pd.DataFrame,
    source_type: str,
    partition: str | None = None,
    *,
    bucket: str | None = None,
    client: ObjectStoreClient | None = None,
) -> str:
    """Write one immutable Snappy Parquet object to MinIO; return its `s3://` URI."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    object_name = _object_name(source_type, partition)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, compression="snappy", engine="pyarrow")
    data = buffer.getvalue()
    return write_bronze_bytes(
        data,
        object_name,
        bucket=bucket,
        content_type="application/octet-stream",
        client=client,
    )
