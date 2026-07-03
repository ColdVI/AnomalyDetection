"""process_tars_sequential.py — parse ADSB historical tars one at a time, delete after success.

Strategy:
  - Tars already uploaded to Bronze MinIO  → local copy deleted (data safe in MinIO)
  - Tars not yet in Bronze MinIO           → parse directly to Silver, then delete local
  - Split .aa + .ab pair                   → joined only after enough disk is freed

Usage:
    # MinIO must be running
    docker compose up -d minio

    python scripts/process_tars_sequential.py [--tar-dir <path>] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TAR_DIR = Path("data/bronze/adsblol_historical/_input")
SPLIT_AA = "v2026.05.01-planes-readsb-prod-0.tar.aa"
SPLIT_AB = "v2026.05.01-planes-readsb-prod-0.tar.ab"
SPLIT_OUT = "v2026.05.01-planes-readsb-prod-0.tar"


def free_gb(path: Path) -> float:
    import shutil
    return shutil.disk_usage(str(path)).free / (1024 ** 3)


def get_bronze_tar_names(client) -> set[str]:
    bronze_bucket = os.getenv("MINIO_BRONZE_BUCKET", "bronze")
    objs = client.list_objects(bronze_bucket, prefix="adsblol_historical/", recursive=True)
    return {Path(o.object_name).name for o in objs if o.object_name.endswith(".tar")}


def join_split_tar(aa_path: Path, ab_path: Path, out_path: Path) -> bool:
    needed = aa_path.stat().st_size + ab_path.stat().st_size
    avail = free_gb(aa_path.parent) * 1024 ** 3
    logger.info("Split join: need %.1fGB, %.1fGB free", needed / 1024 ** 3, avail / 1024 ** 3)
    if avail < needed + 300 * 1024 * 1024:
        logger.error("Not enough disk to join split tar")
        return False
    logger.info("Joining %s + %s", aa_path.name, ab_path.name)
    with open(out_path, "wb") as fout:
        for part in (aa_path, ab_path):
            with open(part, "rb") as fin:
                while chunk := fin.read(8 * 1024 * 1024):
                    fout.write(chunk)
    logger.info("Joined: %.1fGB", out_path.stat().st_size / 1024 ** 3)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--tar-dir", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.common.minio_io import get_minio_client
    from src.silver.parse_adsblol_historical import parse_local_tar

    tar_dir = Path(args.tar_dir) if args.tar_dir else DEFAULT_TAR_DIR
    if not tar_dir.exists():
        logger.error("Tar directory not found: %s", tar_dir)
        sys.exit(1)

    client = get_minio_client()
    bronze_names = get_bronze_tar_names(client)
    logger.info("Already in Bronze MinIO: %s", sorted(bronze_names))

    # Complete local tars (no .aa .ab)
    local_tars = sorted(
        f for f in tar_dir.iterdir()
        if f.suffix == ".tar" and not f.name.endswith((".aa", ".ab"))
    )
    aa_path = tar_dir / SPLIT_AA
    ab_path = tar_dir / SPLIT_AB

    in_bronze = [t for t in local_tars if t.name in bronze_names]
    not_in_bronze = [t for t in local_tars if t.name not in bronze_names]

    logger.info("Local tars in Bronze (will delete local): %d", len(in_bronze))
    logger.info("Local tars not in Bronze (will parse+delete): %d", len(not_in_bronze))

    if args.dry_run:
        for t in in_bronze:
            print(f"  DELETE local (in Bronze): {t.name}  [{t.stat().st_size//1024**2}MB]")
        for t in not_in_bronze:
            print(f"  PARSE+DELETE: {t.name}  [{t.stat().st_size//1024**2}MB]")
        if aa_path.exists() and ab_path.exists():
            total = (aa_path.stat().st_size + ab_path.stat().st_size) // 1024**2
            print(f"  JOIN+PARSE+DELETE: {SPLIT_AA} + {SPLIT_AB}  [{total}MB combined]")
        return

    # Step 1: Delete local copies of tars already in Bronze (data safe in MinIO)
    for t in in_bronze:
        logger.info("Deleting local copy already in Bronze: %s (%.1fGB)",
                    t.name, t.stat().st_size / 1024**3)
        t.unlink()
        logger.info("Deleted. Disk free: %.1fGB", free_gb(tar_dir))

    # Step 2: Parse remaining tars to Silver, delete after success
    for t in not_in_bronze:
        size_gb = t.stat().st_size / 1024**3
        logger.info("=== Processing %s (%.1fGB) | %.1fGB free ===",
                    t.name, size_gb, free_gb(tar_dir))
        try:
            uris = parse_local_tar(str(t), batch_size=args.batch_size, client=client)
            if uris:
                logger.info("OK: %d Silver parts written", len(uris))
                t.unlink()
                logger.info("Deleted %s. Disk free: %.1fGB", t.name, free_gb(tar_dir))
            else:
                logger.error("No Silver parts written — keeping %s", t.name)
        except Exception:
            logger.exception("Error processing %s — keeping file", t.name)

    # Step 3: Handle split tar (needs most free space)
    if aa_path.exists() and ab_path.exists():
        split_combined = tar_dir / SPLIT_OUT
        if free_gb(tar_dir) > (aa_path.stat().st_size + ab_path.stat().st_size) / 1024**3 + 0.3:
            if join_split_tar(aa_path, ab_path, split_combined):
                try:
                    uris = parse_local_tar(str(split_combined), batch_size=args.batch_size, client=client)
                    if uris:
                        logger.info("Split tar: %d Silver parts", len(uris))
                        split_combined.unlink()
                        aa_path.unlink()
                        ab_path.unlink()
                    else:
                        logger.error("No Silver parts from split tar")
                        split_combined.unlink()
                except Exception:
                    logger.exception("Error on split tar")
                    if split_combined.exists():
                        split_combined.unlink()
        else:
            logger.warning("Not enough space to join split tar (%.1fGB free) — skipping",
                           free_gb(tar_dir))
    elif aa_path.exists() or ab_path.exists():
        logger.warning("Only one part of split tar found, cannot join")

    logger.info("=== All done. Disk free: %.1fGB ===", free_gb(tar_dir))


if __name__ == "__main__":
    main()
