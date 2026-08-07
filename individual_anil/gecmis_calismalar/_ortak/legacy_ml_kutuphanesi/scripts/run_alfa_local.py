"""Dev/validation runner: real ALFA zip -> Bronze (in-memory) -> Silver -> local files.

Chains `src.ingestion.upload_raw` + `src.silver.parse_alfa` through the same
in-memory `FakeMinioClient` the test suite uses (src/common/fakes.py), so the
whole ADR-003 pipeline (raw zip upload -> Silver parse) can be exercised
against the real ALFA collection without a running MinIO server. The exact
same two functions work against a real `Minio` client once docker-compose is
up -- nothing here is a second/parallel pipeline.

Usage:
    python scripts/run_alfa_local.py --input "C:\\path\\to\\ALFA\\processed.zip" \\
        --local-out data/silver/alfa_silver.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common.fakes import FakeMinioClient
from src.ingestion.upload_raw import upload_raw_file
from src.silver.parse_alfa import build_alfa_silver

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_alfa_local")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the real ALFA processed.zip (or ALFA.zip)")
    parser.add_argument("--local-out", default="data/silver/alfa_silver.parquet")
    parser.add_argument("--local-out-csv", default=None)
    args = parser.parse_args()

    client = FakeMinioClient()

    logger.info("Bronze: uploading %s (raw, unchanged)", args.input)
    upload_raw_file(args.input, "alfa", client=client)

    logger.info("Silver: parsing ALFA zip from Bronze")
    silver = build_alfa_silver(client)
    if silver.empty:
        logger.error("Silver is empty -- check --input path")
        return

    print(f"\nShape: {silver.shape}")
    print(f"Sequences: {silver['source_id'].nunique()}")
    print("\nlabel distribution:")
    print(silver["label"].value_counts())
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
