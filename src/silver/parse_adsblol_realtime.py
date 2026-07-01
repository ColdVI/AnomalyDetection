"""parse_adsblol_realtime.py -- adsb.lol realtime Silver parser.

Reads raw JSONL files from MinIO Bronze (bronze/adsblol_realtime/_landing/),
applies unit conversions, and writes Silver Parquet to MinIO.

Each JSONL line is one raw `ac` entry from the adsb.lol v2 API response,
as published by src/ingestion/adsblol_producer.py.

Unit conversions (Silver's job per ADR-003):
  alt_baro  feet → metres (* 0.3048), "ground" → on_ground=True, alt=None
  alt_geom  feet → metres (* 0.3048)
  gs        knots → m/s (* 0.5144)
  baro_rate fpm → m/s (* 0.00508)
  geom_rate fpm → m/s (* 0.00508)
  ias, tas  knots → m/s (* 0.5144)

Usage:
    python -m src.silver.parse_adsblol_realtime
"""

from __future__ import annotations

import json
import logging
import os
import re

import pandas as pd

from src.common.minio_io import (
    ObjectStoreClient,
    download_raw_bytes,
    get_minio_client,
    write_silver,
)
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

SOURCE_TYPE = "adsblol_rt"
_LANDING_PREFIX = "adsblol_realtime/_landing/"
_TS_RE = re.compile(r"states-(\d{8}T\d{6})")


def _batch_timestamp(object_name: str) -> float | None:
    """Extract Unix-ish epoch from the JSONL file name (states-YYYYMMDDTHHMMSS...).

    Returns None if the name doesn't match the expected pattern.
    """
    m = _TS_RE.search(object_name)
    if not m:
        return None
    from datetime import datetime, timezone
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _knots_to_ms(v) -> float | None:
    return round(float(v) * 0.5144, 2) if v is not None else None


def _fpm_to_ms(v) -> float | None:
    return round(float(v) * 0.00508, 3) if v is not None else None


def _feet_to_m(v) -> float | None:
    return round(float(v) * 0.3048, 1) if v is not None else None


def _parse_ac_record(record: dict, batch_ts: float | None) -> dict:
    alt_baro_raw = record.get("alt_baro")
    on_ground = alt_baro_raw == "ground"
    alt_m = None if (on_ground or alt_baro_raw is None) else _feet_to_m(alt_baro_raw)
    alt_geom_raw = record.get("alt_geom")
    alt_geom_m = None if alt_geom_raw is None else _feet_to_m(alt_geom_raw)
    return {
        "source_type": SOURCE_TYPE,
        "source_id": record.get("hex"),
        "timestamp_utc": batch_ts,
        "lat": record.get("lat"),
        "lon": record.get("lon"),
        "alt": alt_m,
        "alt_geom_m": alt_geom_m,
        "on_ground": on_ground,
        "label": None,
        "ground_speed_ms": _knots_to_ms(record.get("gs")),
        "track_deg": record.get("track"),
        "vertical_rate_ms": _fpm_to_ms(record.get("baro_rate")),
        "geom_vertical_rate_ms": _fpm_to_ms(record.get("geom_rate")),
        "indicated_airspeed_ms": _knots_to_ms(record.get("ias")),
        "true_airspeed_ms": _knots_to_ms(record.get("tas")),
        "roll_deg": record.get("roll"),
        "flight_callsign": (record.get("flight") or "").strip() or None,
        "category": record.get("category"),
        "squawk": record.get("squawk"),
        "emergency": record.get("emergency"),
        "registration": record.get("r"),
        "aircraft_type": record.get("t"),
        "nic": record.get("nic"),
        "rc": record.get("rc"),
        "nac_p": record.get("nac_p"),
        "sil": record.get("sil"),
        "adsb_version": record.get("version"),
        "seen": record.get("seen"),
        "seen_pos": record.get("seen_pos"),
        "rssi": record.get("rssi"),
    }


def parse_jsonl_bytes(raw: bytes, object_name: str) -> pd.DataFrame:
    """Parse one JSONL blob (bytes) into a Silver DataFrame."""
    batch_ts = _batch_timestamp(object_name)
    rows = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            rows.append(_parse_ac_record(record, batch_ts))
        except (json.JSONDecodeError, Exception):
            logger.warning("Skipping malformed line in %s", object_name)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run(
    bronze_prefix: str = _LANDING_PREFIX,
    *,
    client: ObjectStoreClient | None = None,
    bronze_bucket: str | None = None,
) -> list[str]:
    """Parse all JSONL landing files in Bronze and write Silver. Returns s3:// URIs."""
    client = client or get_minio_client()
    bronze_bucket = bronze_bucket or os.getenv("MINIO_BRONZE_BUCKET", "bronze")

    jsonl_objects = [
        obj.object_name
        for obj in client.list_objects(bronze_bucket, prefix=bronze_prefix, recursive=True)
        if obj.object_name.endswith(".jsonl")
    ]

    if not jsonl_objects:
        logger.warning("No .jsonl objects found under %s/%s", bronze_bucket, bronze_prefix)
        return []

    logger.info("Found %d JSONL file(s) to parse", len(jsonl_objects))
    all_uris: list[str] = []

    for obj_name in sorted(jsonl_objects):
        raw = download_raw_bytes(client, obj_name, bucket=bronze_bucket)
        df = parse_jsonl_bytes(raw, obj_name)
        if df.empty:
            logger.warning("No rows parsed from %s", obj_name)
            continue
        df = add_provenance(
            df, source_type=SOURCE_TYPE, source_file=obj_name, schema_version="silver_v1"
        )
        uri = write_silver(df, "adsblol_realtime", client=client)
        all_uris.append(uri)
        logger.info("Parsed %s -> %d rows -> %s", obj_name, len(df), uri)

    logger.info("Done: %d JSONL files → %d Silver objects", len(jsonl_objects), len(all_uris))
    return all_uris


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="adsb.lol realtime JSONL → Silver Parquet")
    parser.add_argument(
        "--bronze-prefix", default=_LANDING_PREFIX, help="MinIO Bronze prefix for JSONL files"
    )
    args = parser.parse_args()
    run(args.bronze_prefix)
