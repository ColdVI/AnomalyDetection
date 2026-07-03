"""Upload all raw data files to MinIO Bronze.

Reads from DATA_ROOT env var (default: <project>/data).
Structure expected:  <DATA_ROOT>/bronze/<source>/_input/<file>

- Skips files already in MinIO (idempotent, safe to re-run)
- Skips incomplete split-part files (*.tar.aa, *.tar.ab ...)
- Streams large files — no full-file RAM load
- Retries once on transient errors

Usage:
    python scripts/upload_bronze_all.py                  # all sources
    python scripts/upload_bronze_all.py --source alfa    # one source only
    python scripts/upload_bronze_all.py --dry-run        # preview only
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from minio import Minio
from minio.error import S3Error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPLIT_SUFFIXES = {".aa", ".ab", ".ac", ".ad"}


def get_client() -> Minio:
    return Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        log.info("Created bucket: %s", bucket)


def already_uploaded(client: Minio, bucket: str, object_name: str) -> bool:
    try:
        client.stat_object(bucket, object_name)
        return True
    except S3Error as e:
        if e.code == "NoSuchKey":
            return False
        raise


def upload_file(client: Minio, bucket: str, object_name: str, path: Path, retries: int = 2) -> bool:
    size = path.stat().st_size
    for attempt in range(1, retries + 1):
        try:
            with path.open("rb") as fh:
                client.put_object(bucket, object_name, fh, length=size,
                                  content_type="application/octet-stream")
            return True
        except Exception as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries, object_name, exc)
            if attempt < retries:
                time.sleep(5)
    return False


def find_input_files(data_root: Path, source_filter: str | None) -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    bronze_root = data_root / "bronze"
    if not bronze_root.exists():
        log.error("Bronze input root not found: %s", bronze_root)
        return results

    for source_dir in sorted(bronze_root.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        if source_filter and source != source_filter:
            continue
        input_dir = source_dir / "_input"
        if not input_dir.exists():
            continue
        for f in sorted(input_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix in SPLIT_SUFFIXES:
                log.info("Skipping incomplete split part: %s", f.name)
                continue
            results.append((source, f))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", help="e.g. alfa, uav_attack, adsblol_historical")
    parser.add_argument("--data-root", help="Override DATA_ROOT env var")
    args = parser.parse_args()

    data_root_str = args.data_root or os.getenv("DATA_ROOT") or str(ROOT / "data")
    data_root = Path(data_root_str)
    log.info("Data root: %s", data_root)

    bucket = os.getenv("MINIO_BRONZE_BUCKET", "bronze")
    files = find_input_files(data_root, args.source)

    if not files:
        log.error("No input files found under %s/bronze/*/_input/", data_root)
        sys.exit(1)

    total_gb = sum(f.stat().st_size for _, f in files) / 1e9
    log.info("Found %d file(s) | %.1f GB total", len(files), total_gb)

    if args.dry_run:
        for source, f in files:
            log.info("  [DRY-RUN] %s -> bronze/%s/%s  (%.1f MB)",
                     f.name, source, f.name, f.stat().st_size / 1e6)
        return

    client = get_client()
    ensure_bucket(client, bucket)

    uploaded = skipped = failed = 0
    for source, f in files:
        object_name = f"{source}/{f.name}"
        size_mb = f.stat().st_size / 1e6

        if already_uploaded(client, bucket, object_name):
            log.info("SKIP (exists)  %s  %.0f MB", object_name, size_mb)
            skipped += 1
            continue

        log.info("UPLOADING  %s  %.0f MB ...", object_name, size_mb)
        t0 = time.time()
        ok = upload_file(client, bucket, object_name, f)
        elapsed = time.time() - t0
        if ok:
            speed = size_mb / elapsed if elapsed else 0
            log.info("  OK  %.0fs  %.1f MB/s", elapsed, speed)
            # delete local copy after successful upload
            f.unlink()
            log.info("  Deleted local: %s", f)
            uploaded += 1
        else:
            log.error("  FAILED: %s", object_name)
            failed += 1

    log.info("Done — uploaded=%d  skipped=%d  failed=%d", uploaded, skipped, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
