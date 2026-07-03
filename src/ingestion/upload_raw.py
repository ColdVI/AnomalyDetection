"""Bronze: upload a local raw file to MinIO byte-for-byte, no parsing.

ADR-003 (docs/PIPELINE_PLAN.md): Bronze's only job is to hold raw files
(`.zip`, `.tar`, ...) exactly as downloaded. Every source uses this: ALFA's
`processed.zip`, UAV Attack's `UAVAttackData.zip`, and (whoever migrates them)
Metehan's/Yusuf's own raw adsb.lol files.

The object name preserves the original filename under `<source>/`, e.g.
`--source alfa --input processed.zip` -> `bronze/alfa/processed.zip`, so
Silver parsers (src/silver/parse_alfa.py, parse_uav_attack.py) can find it
back by listing that prefix.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.common.minio_io import ObjectStoreClient, get_minio_client, write_bronze_bytes

logger = logging.getLogger(__name__)


def merge_tar_parts(base_path: str | Path, parts: tuple[str, ...] = ("aa", "ab")) -> Path:
    """Concatenate `<base_path>.tar.aa` + `.tar.ab` ... into `<base_path>.tar`.

    No-op if the merged file already exists. Used before uploading split tars to Bronze.
    """
    base_path = Path(base_path)
    merged = base_path.with_suffix(".tar") if base_path.suffix != ".tar" else base_path
    if merged.exists():
        return merged

    part_paths = [Path(f"{base_path}.tar.{p}") for p in parts if Path(f"{base_path}.tar.{p}").exists()]
    if not part_paths:
        raise FileNotFoundError(f"No split parts found for {base_path}.tar.*")

    with merged.open("wb") as out:
        for p in part_paths:
            logger.info("Merging part: %s", p)
            with p.open("rb") as f:
                while chunk := f.read(16 * 1024 * 1024):
                    out.write(chunk)
    logger.info("Merged %d parts -> %s", len(part_paths), merged)
    return merged


def upload_raw_file(
    input_path: str | Path,
    source: str,
    *,
    client: ObjectStoreClient | None = None,
) -> str:
    """Upload `input_path` unchanged to `bronze/<source>/<input_path.name>`.

    Streams the file rather than loading it fully into RAM so large tars (>1 GB)
    don't exhaust memory.
    """
    from src.common.minio_io import ensure_bucket, get_minio_client as _get
    import os

    input_path = Path(input_path)
    size = input_path.stat().st_size
    object_name = f"{source}/{input_path.name}"
    resolved_client = client or _get()
    bucket = os.getenv("MINIO_BRONZE_BUCKET", "bronze")
    ensure_bucket(resolved_client, bucket)
    with input_path.open("rb") as fh:
        resolved_client.put_object(
            bucket, object_name, fh, length=size,
            content_type="application/octet-stream",
        )
    uri = f"s3://{bucket}/{object_name}"
    logger.info("Uploaded %s (%d bytes) -> %s", input_path, size, uri)
    return uri


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Upload a raw local file to MinIO Bronze, unchanged")
    parser.add_argument("--source", required=True, help="Source name, e.g. alfa, uav_attack")
    parser.add_argument("--input", required=True, help="Path to the local raw file (e.g. a .zip)")
    args = parser.parse_args()

    client = get_minio_client()
    upload_raw_file(args.input, args.source, client=client)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
