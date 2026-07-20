"""migrate_gold_partition_layout.py -- one-time migration: moves existing
FLAT `gold/unified/part-*.parquet` objects (written before the 2026-07-18
partitioned-write change in src/gold/unify.py) into their proper
`gold/unified/<source_type>/part-*.parquet` location.

Cheap by design: does NOT re-read Silver or re-apply the 7+3 column map --
each Gold part is already homogeneous in source_type (one write_gold() call
per Silver input file), so this just reads the `source_type` column
(column-pruned) to learn where a part belongs, copies its raw bytes to the
new key (same filename, new prefix), and deletes the old flat key.
Idempotent: a part already under a `<source_type>/` subfolder, or whose
target already exists, is skipped -- safe to re-run after a partial/
interrupted run.

Usage:
    python scripts/migrate_gold_partition_layout.py
    python scripts/migrate_gold_partition_layout.py --dry-run
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.common.minio_io import get_minio_client, read_parquet_object
from src.gold.unify import GOLD_NAME

logger = logging.getLogger(__name__)


def migrate(*, dry_run: bool = False) -> tuple[int, int]:
    client = get_minio_client()
    gold_bucket = os.getenv("MINIO_GOLD_BUCKET", "gold")

    flat_prefix = f"{GOLD_NAME}/"
    all_objects = [obj.object_name for obj in client.list_objects(gold_bucket, prefix=flat_prefix, recursive=True)]

    # "Flat" = directly under unified/ with no further subfolder, i.e. the
    # remainder after stripping the prefix has no "/".
    flat_names = [n for n in all_objects if "/" not in n[len(flat_prefix):]]
    logger.info("Gold/%s: %d object(s) total, %d already-partitioned, %d flat (to migrate)",
                GOLD_NAME, len(all_objects), len(all_objects) - len(flat_names), len(flat_names))

    migrated = skipped = 0
    for name in flat_names:
        df = read_parquet_object(client, gold_bucket, name, columns=["source_type"])
        if df.empty:
            logger.warning("Skipping empty part: %s", name)
            skipped += 1
            continue
        source_type = df["source_type"].mode().iat[0]
        filename = name[len(flat_prefix):]
        new_name = f"{flat_prefix}{source_type}/{filename}"

        if dry_run:
            logger.info("[dry-run] %s -> %s", name, new_name)
            migrated += 1
            continue

        response = client.get_object(gold_bucket, name)
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()
        client.put_object(gold_bucket, new_name, io.BytesIO(raw), length=len(raw),
                           content_type="application/octet-stream")
        client.remove_object(gold_bucket, name)
        migrated += 1
        if migrated % 500 == 0:
            logger.info("  %d/%d migrated", migrated, len(flat_names))

    logger.info("Migration complete: %d migrated, %d skipped", migrated, skipped)
    return migrated, skipped


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only log what would move, don't actually copy/delete")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)
