"""Dev/validation runner: real UAV Attack zip -> Bronze (in-memory) -> Silver -> local files.

Mirrors `scripts/run_alfa_local.py`. Chains `src.ingestion.upload_raw` +
`src.silver.parse_uav_attack` through the same in-memory `FakeMinioClient`
(src/common/fakes.py) so the ADR-003 pipeline can be exercised against the
real IEEE DataPort UAV Attack zip without a running MinIO server.

Usage:
    python scripts/run_uav_attack_local.py --input "C:\\path\\to\\UAVAttackData.zip" \\
        --local-out data/silver/uav_attack_silver.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common.fakes import FakeMinioClient
from src.ingestion.upload_raw import upload_raw_file
from src.silver.parse_uav_attack import build_uav_attack_silver

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_uav_attack_local")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the real UAVAttackData.zip")
    parser.add_argument("--local-out", default="data/silver/uav_attack_silver.parquet")
    parser.add_argument("--local-out-csv", default=None)
    args = parser.parse_args()

    client = FakeMinioClient()

    logger.info("Bronze: uploading %s (raw, unchanged)", args.input)
    upload_raw_file(args.input, "uav_attack", client=client)

    logger.info("Silver: parsing UAV Attack zip from Bronze")
    silver = build_uav_attack_silver(client)
    if silver.empty:
        logger.error("Silver is empty -- check --input path")
        return

    print(f"\nShape: {silver.shape}")
    print(f"Logs: {silver['source_id'].nunique()}")
    print("\nlabel distribution (rows):")
    print(silver["label"].value_counts())
    print("\ntimestamp_is_real_utc distribution (logs):")
    print(silver.groupby("source_id")["timestamp_is_real_utc"].first().value_counts())
    print(f"\nColumns ({len(silver.columns)}): {list(silver.columns)}")

    if args.local_out:
        Path(args.local_out).parent.mkdir(parents=True, exist_ok=True)
        silver.to_parquet(args.local_out, index=False)
        logger.info("Wrote %s", args.local_out)
    if args.local_out_csv:
        Path(args.local_out_csv).parent.mkdir(parents=True, exist_ok=True)
        silver.to_csv(args.local_out_csv, index=False)
        logger.info("Wrote %s", args.local_out_csv)


if __name__ == "__main__":
    main()
