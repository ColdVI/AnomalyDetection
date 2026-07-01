"""In-memory fake of the Minio client surface.

Lives outside `tests/` (which has no `__init__.py` and isn't a real importable
package) so both pytest fixtures and standalone scripts -- e.g. `scripts/`'s
local runners, which chain a real upload + Silver parse without a running
MinIO server -- can use the exact same fake. `FakeMinioClient` implements
exactly the methods `src.common.minio_io.ObjectStoreClient` declares
(bucket_exists/make_bucket/put_object/list_objects/get_object), all backed by
a plain dict.
"""

from __future__ import annotations


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


class _FakeObject:
    """Minimal stand-in for the minio.Object entries minio.list_objects yields."""

    def __init__(self, object_name: str) -> None:
        self.object_name = object_name


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

    def list_objects(self, bucket_name: str, prefix: str | None = None, recursive: bool = False) -> list[_FakeObject]:
        names = self.buckets.get(bucket_name, {}).keys()
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return [_FakeObject(n) for n in sorted(names)]
