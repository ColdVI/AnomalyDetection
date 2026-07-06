"""Gold: N Silver sources aligned to a 7 common-column + 3 metadata schema.

Per `docs/PIPELINE_PLAN (1).md` ("ORTAK -- Gold, hep birlikte, Silver review'dan sonra")
and ADR-003 (`docs/decisions.md`): Gold's only job is aligning every source's Silver table
to 7 common columns (`timestamp_utc`, `lat`, `lon`, `altitude_m`, `velocity_mps`,
`heading_deg`, `vertical_rate_mps`) + 3 metadata columns (`source_type`, `source_id`,
`label`). Source-specific columns (`squawk`, `roll_deg`, `jamming_indicator`, ...) stay in
Silver and are dropped here. Adding a new dataset means adding one `COLUMN_MAPS` entry, not
a new code path.

This is NOT `src/processing/gold.py` (ADR-004): that file is the old wide-sparse
UNION-of-everything reference approach, kept only as a possible future re-enrichment path.
This module is the pipeline's actual Gold step.

Column values in `COLUMN_MAPS` are either the Silver column name to copy from, or `None`
if that source has no such field (filled with NaN/None rather than guessed).
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import pandas as pd

from src.common.minio_io import (
    ObjectStoreClient,
    delete_layer_objects,
    get_minio_client,
    list_layer_objects,
    read_parquet_object,
    write_gold,
)

logger = logging.getLogger(__name__)

GOLD_NAME = "unified"

GOLD_COLUMNS = [
    "timestamp_utc",
    "lat",
    "lon",
    "altitude_m",
    "velocity_mps",
    "heading_deg",
    "vertical_rate_mps",
    "source_type",
    "source_id",
    "label",
]

# docs/PIPELINE_PLAN (1).md, "Gold ortak sema (7 temel kolon + metadata)" tablosu.
# Keys must match the MinIO Silver prefix (= first arg of write_silver() call in each parser).
COLUMN_MAPS: dict[str, dict[str, str | None]] = {
    "adsblol_historical": {
        "timestamp_utc": "timestamp_utc",
        "lat": "lat",
        "lon": "lon",
        "altitude_m": "alt",
        "velocity_mps": "ground_speed_ms",
        "heading_deg": "track_deg",
        "vertical_rate_mps": "vertical_rate_ms",
        "source_type": "source_type",
        "source_id": "source_id",
        "label": None,  # adsb.lol'da etiket yok -- her zaman null.
    },
    # Historical MinIO prefix alias (some tests use 'adsblol_hist')
    "adsblol_hist": {
        "timestamp_utc": "timestamp_utc",
        "lat": "lat",
        "lon": "lon",
        "altitude_m": "alt",
        "velocity_mps": "ground_speed_ms",
        "heading_deg": "track_deg",
        "vertical_rate_mps": "vertical_rate_ms",
        "source_type": "source_type",
        "source_id": "source_id",
        "label": None,  # adsb.lol'da etiket yok -- her zaman null.
    },

    "adsblol_realtime": {
        "timestamp_utc": "timestamp_utc",
        "lat": "lat",
        "lon": "lon",
        "altitude_m": "alt",
        "velocity_mps": "ground_speed_ms",
        "heading_deg": "track_deg",
        "vertical_rate_mps": "vertical_rate_ms",
        "source_type": "source_type",
        "source_id": "source_id",
        "label": None,  # adsb.lol'da etiket yok -- her zaman null.
    },
    # Realtime MinIO prefix alias (some tests use 'adsblol_rt')
    "adsblol_rt": {
        "timestamp_utc": "timestamp_utc",
        "lat": "lat",
        "lon": "lon",
        "altitude_m": "alt",
        "velocity_mps": "ground_speed_ms",
        "heading_deg": "track_deg",
        "vertical_rate_mps": "vertical_rate_ms",
        "source_type": "source_type",
        "source_id": "source_id",
        "label": None,  # adsb.lol'da etiket yok -- her zaman null.
    },
    "alfa": {
        "timestamp_utc": "timestamp_utc",
        "lat": "lat",
        "lon": "lon",
        "altitude_m": "alt",
        # COZULDU (2026-07-02): kok neden nav_info-velocity kolon adlarinin
        # "meas_x/des_x" olmasiydi (find_col(['measured']) eslesmiyordu).
        # parse_alfa.py artik bilesenlerden velocity_measured'i hesapliyor;
        # vfr_hud'dan climb_rate_ms de dikey hizi sagliyor.
        "velocity_mps": "velocity_measured",
        "heading_deg": "yaw_measured",
        "vertical_rate_mps": "climb_rate_ms",
        "source_type": "source_type",
        "source_id": "source_id",
        "label": "label",
    },
    "uav_attack": {
        "timestamp_utc": "timestamp_utc",
        "lat": "lat",
        "lon": "lon",
        "altitude_m": "alt",
        # COZULDU (2026-07-02): vehicle_gps_position'daki vel_m_s / vel_d_m_s
        # kolonlari parse_uav_attack.py'ye eklendi.
        "velocity_mps": "vel_m_s",
        "heading_deg": "yaw_deg",
        "vertical_rate_mps": "vertical_rate_mps",
        "source_type": "source_type",
        "source_id": "source_id",
        "label": "label",
    },
    # UAV-SEAD: ayni PX4 uORB semantigi -- uav_attack ile birebir ayni eslesme.
    "uav_sead": {
        "timestamp_utc": "timestamp_utc",
        "lat": "lat",
        "lon": "lon",
        "altitude_m": "alt",
        "velocity_mps": "vel_m_s",
        "heading_deg": "yaw_deg",
        "vertical_rate_mps": "vertical_rate_mps",
        "source_type": "source_type",
        "source_id": "source_id",
        "label": "label",
    },
}


def _apply_column_map(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for gold_col in GOLD_COLUMNS:
        silver_col = mapping.get(gold_col)
        if silver_col is not None and silver_col in df.columns:
            out[gold_col] = df[silver_col]
        else:
            out[gold_col] = None
    return out


def clear_gold_before_unify(client: ObjectStoreClient, *, gold_bucket: str | None = None) -> int:
    """Delete every existing object under `<gold_bucket>/unified/` before a fresh unify run.

    Both `stream_unify()` and `main()`'s in-memory path call `write_gold()`, which always
    creates a new timestamped+uuid part name (never overwrites). Without this step, each
    rerun leaves the previous run's parts in place alongside the new ones, so `unified/`
    accumulates duplicate data and every downstream read (Gold review, the individual
    project's `load_adsb_gold_data()`) silently double/N-counts rows. Returns the number
    of objects removed.
    """
    g_bucket = gold_bucket or os.getenv("MINIO_GOLD_BUCKET", "gold")
    removed = delete_layer_objects(client, g_bucket, GOLD_NAME)
    if removed:
        logger.info("Gold: cleared %d stale part(s) from %s/%s/ before unify", removed, g_bucket, GOLD_NAME)
    return removed


def stream_unify(
    client: ObjectStoreClient,
    *,
    silver_bucket: str | None = None,
    source_types: tuple[str, ...] = tuple(COLUMN_MAPS),
    gold_bucket: str | None = None,
) -> int:
    """Stream each Silver Parquet file through the 7+3 column map and write one Gold
    Parquet per input file. Returns the total number of rows written.

    Uses file-by-file streaming so large datasets (>64M rows) don't exhaust RAM.
    Clears any prior `unified/` output first (see `clear_gold_before_unify`) so reruns
    don't double-count.
    """
    s_bucket = silver_bucket or os.getenv("MINIO_SILVER_BUCKET", "silver")
    clear_gold_before_unify(client, gold_bucket=gold_bucket)
    total_rows = 0
    total_parts = 0

    for source_type in source_types:
        mapping = COLUMN_MAPS.get(source_type)
        if mapping is None:
            raise ValueError(f"No Gold column mapping registered for source_type={source_type!r}")

        object_names = list_layer_objects(client, s_bucket, source_type)
        if not object_names:
            logger.warning("No Silver data for source_type=%s -- skipping", source_type)
            continue

        logger.info("Gold: streaming %d Silver part(s) for source_type=%s", len(object_names), source_type)
        for obj_name in object_names:
            df = read_parquet_object(client, s_bucket, obj_name)
            aligned = _apply_column_map(df, mapping)
            write_gold(aligned, GOLD_NAME, client=client, bucket=gold_bucket)
            total_rows += len(aligned)
            total_parts += 1

        logger.info("Gold: source_type=%s done -- %d parts written so far", source_type, total_parts)

    logger.info("Gold %s complete: %d total rows in %d parts", GOLD_NAME, total_rows, total_parts)
    return total_rows


def unify(
    client: ObjectStoreClient,
    *,
    silver_bucket: str | None = None,
    source_types: tuple[str, ...] = tuple(COLUMN_MAPS),
) -> pd.DataFrame:
    """Read each source's Silver table and align it to the 7+3 Gold schema, then UNION.

    WARNING: loads all Silver into RAM. For large datasets use stream_unify() instead.
    """
    from src.common.minio_io import read_layer

    bucket = silver_bucket or os.getenv("MINIO_SILVER_BUCKET", "silver")

    frames = []
    for source_type in source_types:
        mapping = COLUMN_MAPS.get(source_type)
        if mapping is None:
            raise ValueError(f"No Gold column mapping registered for source_type={source_type!r}")

        df = read_layer(client, bucket, source_type)
        if df.empty:
            logger.warning("No Silver data for source_type=%s under bucket=%s -- skipping", source_type, bucket)
            continue

        aligned = _apply_column_map(df, mapping)
        frames.append(aligned)
        logger.info("Gold input: source_type=%s, %d rows aligned to 7+3 schema", source_type, len(aligned))

    if not frames:
        logger.error("No Silver tables found for any of %s -- nothing to unify", source_types)
        return pd.DataFrame(columns=GOLD_COLUMNS)

    gold = pd.concat(frames, ignore_index=True, sort=False)[GOLD_COLUMNS]
    logger.info(
        "Gold %s: %d rows, source_type distribution:\n%s",
        GOLD_NAME, len(gold), gold["source_type"].value_counts().to_string(),
    )
    return gold


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Silver -> Gold (7+3 common-column unify)")
    parser.add_argument("--local-out", default=None, help="Optional local Parquet path (small datasets only)")
    parser.add_argument("--local-out-csv", default=None, help="Optional local CSV path (small datasets only)")
    args = parser.parse_args()

    client = get_minio_client()

    if args.local_out or args.local_out_csv:
        # In-memory path: only for small datasets.
        gold = unify(client)
        if gold.empty:
            logger.error("Nothing to write: Gold is empty (did any Silver step run?)")
            return
        clear_gold_before_unify(client)
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
    else:
        total = stream_unify(client)
        if total == 0:
            logger.error("Nothing written: Gold is empty (did any Silver step run?)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
