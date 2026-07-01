"""Shared test fixtures: an in-memory fake of the Minio client surface.

None of the Bronze tests need a running MinIO server. `FakeMinioClient`
implements exactly the methods `src.common.io.ObjectStoreClient` declares
(bucket_exists/make_bucket/put_object), plus `get_object` so tests can read
back what was written, all backed by a plain dict.
"""

from __future__ import annotations

import pytest


class _FakeGetResponse:
    """Minimal stand-in for the urllib3 response minio.get_object returns."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        pass

    def release_conn(self) -> None:
        pass


class FakeMinioClient:
    """In-memory object store: {bucket: {object_name: bytes}}."""

    def __init__(self) -> None:
        self.buckets: dict[str, dict[str, bytes]] = {}
        self.put_calls: list[tuple[str, str, str]] = []  # (bucket, object_name, content_type)

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self.buckets

    def make_bucket(self, bucket_name: str) -> None:
        self.buckets.setdefault(bucket_name, {})

    def put_object(self, bucket_name, object_name, data, length, content_type="application/octet-stream", **_kwargs):
        if bucket_name not in self.buckets:
            raise RuntimeError(f"bucket {bucket_name!r} does not exist (call make_bucket first)")
        payload = data.read() if hasattr(data, "read") else bytes(data)
        self.buckets[bucket_name][object_name] = payload
        self.put_calls.append((bucket_name, object_name, content_type))

    def get_object(self, bucket_name: str, object_name: str) -> _FakeGetResponse:
        return _FakeGetResponse(self.buckets[bucket_name][object_name])

    def list_object_names(self, bucket_name: str) -> list[str]:
        return sorted(self.buckets.get(bucket_name, {}).keys())


@pytest.fixture
def fake_minio_client() -> FakeMinioClient:
    return FakeMinioClient()
