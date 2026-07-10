"""Dev/validation runner: real ALFA + UAV Attack zips -> Bronze -> Silver -> Gold (local files).

Chains `src.ingestion.upload_raw` + `src.silver.parse_alfa` / `parse_uav_attack` +
`src.gold.unify` through the same in-memory `FakeMinioClient` the test suite uses
(src/common/fakes.py), so the whole ADR-003 pipeline (raw zip upload -> Silver parse
-> Gold unify) can be exercised against the real ALFA/UAV Attack collections without a
running MinIO server. The exact same functions work against a real `Minio` client once
docker-compose is up -- nothing here is a second/parallel pipeline.

Usage:
    python scripts/run_gold_local.py --alfa-input "C:\\path\\to\\ALFA\\processed.zip" \\
        --uav-attack-input "C:\\path\\to\\UAVAttackData.zip" \\
        --local-out data/gold/unified.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common.fakes import FakeMinioClient
from src.common.minio_io import write_silver
from src.gold.unify import unify
from src.ingestion.upload_raw import upload_raw_file
from src.silver.parse_alfa import build_alfa_silver
from src.silver.parse_uav_attack import build_uav_attack_silver

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_gold_local")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alfa-input", default=None, help="Path to the real ALFA processed.zip (or ALFA.zip)")
    parser.add_argument("--uav-attack-input", default=None, help="Path to the real UAVAttackData.zip")
    parser.add_argument("--local-out", default="data/gold/unified.parquet")
    parser.add_argument("--local-out-csv", default=None)
    args = parser.parse_args()

    if not args.alfa_input and not args.uav_attack_input:
        parser.error("Provide at least one of --alfa-input / --uav-attack-input")

    client = FakeMinioClient()

    if args.alfa_input:
        logger.info("Bronze+Silver: ALFA (%s)", args.alfa_input)
        upload_raw_file(args.alfa_input, "alfa", client=client)
        alfa_silver = build_alfa_silver(client)
        if alfa_silver.empty:
            logger.warning("ALFA Silver empty -- check --alfa-input")
        else:
            write_silver(alfa_silver, "alfa", client=client)

    if args.uav_attack_input:
        logger.info("Bronze+Silver: UAV Attack (%s)", args.uav_attack_input)
        upload_raw_file(args.uav_attack_input, "uav_attack", client=client)
        uav_silver = build_uav_attack_silver(client)
        if uav_silver.empty:
            logger.warning("UAV Attack Silver empty -- check --uav-attack-input")
        else:
            write_silver(uav_silver, "uav_attack", client=client)

    logger.info("Gold: unifying available Silver sources to the 7+3 schema")
    gold = unify(client)
    if gold.empty:
        logger.error("Gold is empty -- no Silver source produced data")
        return

    print(f"\nShape: {gold.shape}")
    print("\nsource_type distribution:")
    print(gold["source_type"].value_counts())
    print("\nlabel distribution:")
    print(gold["label"].value_counts(dropna=False))
    print(f"\nColumns: {list(gold.columns)}")
    print("\nnull count per column:")
    print(gold.isna().sum())

    if args.local_out:
        Path(args.local_out).parent.mkdir(parents=True, exist_ok=True)
        gold.to_parquet(args.local_out, index=False)
        logger.info("Wrote %s", args.local_out)
    if args.local_out_csv:
        Path(args.local_out_csv).parent.mkdir(parents=True, exist_ok=True)
        gold.to_csv(args.local_out_csv, index=False)
        logger.info("Wrote %s", args.local_out_csv)


if __name__ == "__main__":
    main()
