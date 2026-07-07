"""Disk-backed ObjectStoreClient -- a local stand-in for MinIO (no Docker required).

Unlike `FakeMinioClient` (in-memory, src/common/fakes.py, test-only -- lost as soon as the
process exits), this persists objects to real files under a base directory, so Bronze/Silver/
Gold stay readable across separate commands (e.g. `make bronze-alfa` then `make silver-alfa`)
without a running MinIO server. Selected via `STORAGE_BACKEND=local` (see .env.example) --
`get_minio_client()` in `src.common.minio_io` returns this instead of a real `Minio` client.
Implements exactly the `ObjectStoreClient` protocol surface (bucket_exists/make_bucket/
put_object/list_objects/get_object/remove_object).
"""

from __future__ import annotations

from pathlib import Path


class _LocalGetResponse:
    """Minimal stand-in for the urllib3 response minio.get_object returns."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        pass

    def release_conn(self) -> None:
        pass


class _LocalObject:
    """Minimal stand-in for the minio.Object entries minio.list_objects yields."""

    def __init__(self, object_name: str) -> None:
        self.object_name = object_name


class LocalObjectStoreClient:
    """Object store backed by `<base_dir>/<bucket>/<object_name>` files on disk."""

    def __init__(self, base_dir: str = "data/objectstore") -> None:
        self.base_dir = Path(base_dir)

    def _bucket_dir(self, bucket_name: str) -> Path:
        return self.base_dir / bucket_name

    def bucket_exists(self, bucket_name: str) -> bool:
        return self._bucket_dir(bucket_name).is_dir()

    def make_bucket(self, bucket_name: str) -> None:
        self._bucket_dir(bucket_name).mkdir(parents=True, exist_ok=True)

    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data,
        length: int,
        content_type: str = "application/octet-stream",
        **_kwargs,
    ) -> None:
        if not self.bucket_exists(bucket_name):
            raise RuntimeError(f"bucket {bucket_name!r} does not exist (call make_bucket first)")
        payload = data.read() if hasattr(data, "read") else bytes(data)
        target = self._bucket_dir(bucket_name) / object_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    def get_object(self, bucket_name: str, object_name: str) -> _LocalGetResponse:
        target = self._bucket_dir(bucket_name) / object_name
        return _LocalGetResponse(target.read_bytes())

    def remove_object(self, bucket_name: str, object_name: str) -> None:
        target = self._bucket_dir(bucket_name) / object_name
        target.unlink(missing_ok=True)

    def list_objects(
        self,
        bucket_name: str,
        prefix: str | None = None,
        recursive: bool = False,
    ) -> list[_LocalObject]:
        bucket_dir = self._bucket_dir(bucket_name)
        if not bucket_dir.is_dir():
            return []
        names = [p.relative_to(bucket_dir).as_posix() for p in bucket_dir.rglob("*") if p.is_file()]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return [_LocalObject(n) for n in sorted(names)]
