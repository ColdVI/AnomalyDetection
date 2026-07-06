"""MinIO IO: the one place every layer (Bronze/Silver/Gold) reads and writes through.

ADR-002 (2026-06-30) + ADR-003 (2026-07-01, docs/PIPELINE_PLAN.md): everything lives in
MinIO (S3-compatible object storage). Bronze uploads raw files byte-for-byte (no parsing,
no provenance -- see `write_bronze_bytes` / `download_raw_bytes`); Silver parses per source
and is where `add_provenance` (src/common/provenance.py) gets attached, written via
`write_silver`. Renamed from `src/common/io.py` as part of that pivot.

The MinIO client is injectable so loaders/parsers/tests never need a running MinIO
server: pass `client=<fake or real Minio instance>` to skip env-based client construction.
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from minio import Minio

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.=-]*$")
DEFAULT_BUCKET = "bronze"


class ObjectStoreClient(Protocol):
    """The subset of the Minio client surface our code depends on.

    Any object satisfying this (real `minio.Minio`, or a test fake) works with every
    function in this module -- `list_objects`/`get_object` signatures match `minio.Minio`
    exactly.
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

    def list_objects(
        self,
        bucket_name: str,
        prefix: str | None = None,
        recursive: bool = False,
    ): ...

    def get_object(self, bucket_name: str, object_name: str): ...

    def remove_object(self, bucket_name: str, object_name: str) -> None: ...


def _validate_component(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{name} must be a non-empty, filesystem/object-key-safe name; got {value!r}"
        )
    return value


def get_minio_client() -> ObjectStoreClient:
    """Build an object-store client from env vars (see .env.example).

    `STORAGE_BACKEND=local` (default: `minio`) returns a disk-backed client
    (`src.common.local_store.LocalObjectStoreClient`, rooted at `LOCAL_STORAGE_DIR`)
    instead of connecting to a real MinIO server -- for running the pipeline before
    docker-compose (Kafka/MinIO) is set up. Same call sites, same `ObjectStoreClient`
    surface either way.
    """
    load_dotenv()
    backend = os.getenv("STORAGE_BACKEND", "minio").strip().lower()
    if backend == "local":
        from src.common.local_store import LocalObjectStoreClient

        return LocalObjectStoreClient(os.getenv("LOCAL_STORAGE_DIR", "data/objectstore"))

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
    """Upload raw bytes to MinIO unchanged and return the `s3://bucket/key` URI.

    This is Bronze's only write path now (ADR-003): callers pick `object_name` themselves
    (e.g. `f"{source}/{original_filename}"` for a raw zip, or a dated JSONL landing path
    for the realtime consumer) -- no parsing, no provenance, no Parquet conversion here.
    `bucket` defaults to the MINIO_BRONZE_BUCKET env var (falling back to "bronze"),
    resolved at call time so tests/scripts can rely on .env being loaded by the caller first.
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


def download_raw_bytes(
    client: ObjectStoreClient,
    object_name: str,
    *,
    bucket: str | None = None,
) -> bytes:
    """Read one raw object back from MinIO (the Bronze read path -- the inverse of
    `write_bronze_bytes`). Used by Silver parsers to fetch e.g. a raw ALFA/UAV Attack zip
    without ever writing it to local disk.
    """
    bucket = bucket or os.getenv("MINIO_BRONZE_BUCKET", DEFAULT_BUCKET)
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _write_layer(
    df: pd.DataFrame,
    source_type: str,
    partition: str | None,
    *,
    bucket: str | None,
    default_bucket_env: str,
    default_bucket: str,
    client: ObjectStoreClient | None,
) -> str:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    object_name = _object_name(source_type, partition)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, compression="snappy", engine="pyarrow")
    data = buffer.getvalue()
    resolved_bucket = bucket or os.getenv(default_bucket_env, default_bucket)
    return write_bronze_bytes(
        data,
        object_name,
        bucket=resolved_bucket,
        content_type="application/octet-stream",
        client=client,
    )


def write_bronze(
    df: pd.DataFrame,
    source_type: str,
    partition: str | None = None,
    *,
    bucket: str | None = None,
    client: ObjectStoreClient | None = None,
) -> str:
    """Write one immutable Snappy Parquet object to MinIO; return its `s3://` URI.

    Kept for sources that still write parsed DataFrames straight to Bronze (pre-ADR-003
    call sites); new Bronze code should prefer `write_bronze_bytes` with the raw file.
    """
    return _write_layer(
        df, source_type, partition,
        bucket=bucket, default_bucket_env="MINIO_BRONZE_BUCKET", default_bucket=DEFAULT_BUCKET,
        client=client,
    )


def write_silver(
    df: pd.DataFrame,
    source_type: str,
    partition: str | None = None,
    *,
    bucket: str | None = None,
    client: ObjectStoreClient | None = None,
) -> str:
    """Write one immutable Snappy Parquet object to the Silver bucket; return its `s3://` URI."""
    return _write_layer(
        df, source_type, partition,
        bucket=bucket, default_bucket_env="MINIO_SILVER_BUCKET", default_bucket="silver",
        client=client,
    )


def write_gold(
    df: pd.DataFrame,
    source_type: str,
    partition: str | None = None,
    *,
    bucket: str | None = None,
    client: ObjectStoreClient | None = None,
) -> str:
    """Write one immutable Snappy Parquet object to the Gold bucket; return its `s3://` URI."""
    return _write_layer(
        df, source_type, partition,
        bucket=bucket, default_bucket_env="MINIO_GOLD_BUCKET", default_bucket="gold",
        client=client,
    )


def list_layer_objects(
    client: ObjectStoreClient,
    bucket: str,
    source_type: str,
) -> list[str]:
    """List every object key under `<source_type>/` in `bucket` (Bronze/Silver/Gold alike)."""
    prefix = _validate_component(source_type, "source_type") + "/"
    return [obj.object_name for obj in client.list_objects(bucket, prefix=prefix, recursive=True)]


def delete_layer_objects(
    client: ObjectStoreClient,
    bucket: str,
    source_type: str,
) -> int:
    """Delete every object under `<source_type>/` in `bucket`. Returns the count removed.

    Used to clear a prior run's output before rewriting it (e.g. Gold's `unified/`
    prefix, see `src/gold/unify.py:clear_gold_before_unify`) so repeated runs don't
    accumulate duplicate parts alongside the new ones.
    """
    object_names = list_layer_objects(client, bucket, source_type)
    for name in object_names:
        client.remove_object(bucket, name)
    return len(object_names)


def read_parquet_object(client: ObjectStoreClient, bucket: str, object_name: str) -> pd.DataFrame:
    """Read one Parquet object back from MinIO into a DataFrame."""
    response = client.get_object(bucket, object_name)
    try:
        return pd.read_parquet(io.BytesIO(response.read()))
    finally:
        response.close()
        response.release_conn()


def apply_realtime_retention(
    client: ObjectStoreClient,
    bucket: str | None = None,
    prefix: str = "adsblol_realtime/_landing/",
    days: int = 7,
) -> None:
    """Set a MinIO ILM lifecycle rule that expires objects after `days` days.

    Only applied to `prefix` -- never to historical, alfa, uav_attack, or
    uav_sead prefixes, which hold static datasets that must not be auto-deleted.
    """
    from minio.lifecycleconfig import Expiration, Filter, LifecycleConfig, Rule

    resolved_bucket = bucket or os.getenv("MINIO_BRONZE_BUCKET", DEFAULT_BUCKET)
    cfg = LifecycleConfig(
        [
            Rule(
                "Enabled",
                rule_filter=Filter(prefix=prefix),
                rule_id="rt-retention",
                expiration=Expiration(days=days),
            )
        ]
    )
    client.set_bucket_lifecycle(resolved_bucket, cfg)


def read_layer(client: ObjectStoreClient, bucket: str, source_type: str) -> pd.DataFrame:
    """Read and concatenate every `<source_type>/` object in `bucket` into one DataFrame.

    Returns an empty DataFrame if no objects exist yet (e.g. upstream layer not run).
    """
    object_names = list_layer_objects(client, bucket, source_type)
    if not object_names:
        return pd.DataFrame()
    frames = [read_parquet_object(client, bucket, name) for name in object_names]
    return pd.concat(frames, ignore_index=True, sort=False)
