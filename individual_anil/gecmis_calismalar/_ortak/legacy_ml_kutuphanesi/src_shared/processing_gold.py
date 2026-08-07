"""Gold: `common_uav_events` -- a UNION (never a JOIN) of every source's Silver table.

NOT the active pipeline (see docs/PIPELINE_PLAN.md / ADR-003, docs/decisions.md):
the team's Gold step is `src/gold/unify.py` (not yet built -- explicitly gated
on the whole team's Silver review first) and aligns every source to a 7+3
common-column schema, not a wide-sparse union of each source's full Silver
columns. Kept as a validated reference for the "union, not join, across
mismatched sources" approach once that Gold work starts.

ALFA (2018 fixed-wing actuator faults) and UAV Attack (PX4 multirotor GPS
attacks) share no timestamps, no locations, and mostly disjoint feature
spaces. Rows are stacked with a `source_type` discriminator column; each
source's columns stay in its own namespace and are NaN for every other
source's rows. This is intentionally "wide but sparse" -- downstream
modeling already trains one model per source_type (see docs/MEMORY.md), so
Gold's only job is to be the single place that holds all of them together.

A missing source's Silver table (not yet built/run) is skipped with a
warning rather than treated as an error, so Gold can be produced with
whatever sources are actually available.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from src.common.minio_io import DEFAULT_BUCKET, ObjectStoreClient, get_minio_client, read_layer, write_gold

logger = logging.getLogger(__name__)

GOLD_NAME = "common_uav_events"
KNOWN_SOURCE_TYPES = ("alfa", "uav_attack")


def build_gold(
    client: ObjectStoreClient,
    *,
    silver_bucket: str | None = None,
    source_types: tuple[str, ...] = KNOWN_SOURCE_TYPES,
) -> pd.DataFrame:
    """Read each source's Silver table and UNION them into one Gold table."""
    bucket = silver_bucket or os.getenv("MINIO_SILVER_BUCKET", "silver")

    frames = []
    for source_type in source_types:
        df = read_layer(client, bucket, source_type)
        if df.empty:
            logger.warning("No Silver data for source_type=%s under bucket=%s -- skipping", source_type, bucket)
            continue
        frames.append(df)
        logger.info("Gold input: source_type=%s, %d rows, %d columns", source_type, len(df), df.shape[1])

    if not frames:
        logger.error("No Silver tables found for any of %s -- nothing to union", source_types)
        return pd.DataFrame()

    gold = pd.concat(frames, ignore_index=True, sort=False)
    logger.info(
        "Gold %s: %d rows, %d columns, source_type distribution:\n%s",
        GOLD_NAME, len(gold), gold.shape[1], gold["source_type"].value_counts().to_string(),
    )
    return gold


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Silver -> Gold UNION (common_uav_events)")
    parser.add_argument("--local-out", default=None, help="Optional local Parquet path")
    parser.add_argument("--local-out-csv", default=None, help="Optional local CSV path")
    args = parser.parse_args()

    client = get_minio_client()
    gold = build_gold(client)
    if gold.empty:
        logger.error("Nothing to write: Gold is empty (did any Silver step run?)")
        return

    uri = write_gold(gold, GOLD_NAME, client=client)
    logger.info("Wrote Gold -> %s", uri)

    if args.local_out:
        Path(args.local_out).parent.mkdir(parents=True, exist_ok=True)
        gold.to_parquet(args.local_out, index=False)
        logger.info("Local copy written: %s", args.local_out)
    if args.local_out_csv:
        Path(args.local_out_csv).parent.mkdir(parents=True, exist_ok=True)
        gold.to_csv(args.local_out_csv, index=False)
        logger.info("Local CSV copy written: %s", args.local_out_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
