"""parse_adsblol_historical.py -- adsb.lol historical Silver parser.

Moved from src/bronze2silverParsers/parse_adsb_traces_from_tar_v2.py per ADR-003
(docs/PIPELINE_PLAN.md, METEHAN REHBERİ). parse_trace_bytes() is UNCHANGED --
only the IO layer changed: input is a MinIO Bronze tar object (or a local tar
path for development); output goes to MinIO Silver via write_silver().

See docs/PIPELINE_PLAN.md for the full Silver column spec.

Usage (from MinIO Bronze):
    python -m src.silver.parse_adsblol_historical

Usage (local tar file, no MinIO needed):
    python -m src.silver.parse_adsblol_historical --local-tar data/bronze/adsblol_historical/_input/v2026.06.28-planes-readsb-prod-0.tar
"""

from __future__ import annotations

import gc
import gzip
import io
import json
import logging
import os
import tarfile
from pathlib import Path

import pandas as pd

from src.common.minio_io import (
    ObjectStoreClient,
    download_raw_bytes,
    get_minio_client,
    write_silver,
)
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

SOURCE_TYPE = "adsblol_historical"

TRACE_COLS = [
    "t_offset", "lat", "lon", "alt_raw", "gs", "track", "flags", "vrate",
    "ac_dict", "ads_source_type", "alt_geom", "vrate_geom", "ias", "roll",
]


def parse_trace_bytes(raw: bytes) -> pd.DataFrame:
    """Parse one gzip-compressed (or plain) per-aircraft trace JSON into Silver rows.

    UNCHANGED from src/bronze2silverParsers/parse_adsb_traces_from_tar_v2.py.
    Unit conversions (feet→m, knots→m/s, fpm→m/s) happen here -- Silver's job.
    """
    try:
        data = json.loads(gzip.decompress(raw))
    except OSError:
        data = json.loads(raw)

    icao = data.get("icao")
    file_ts = data.get("timestamp")
    trace = data.get("trace", [])

    rows = []
    last_ac: dict = {}
    for row in trace:
        row = list(row) + [None] * (14 - len(row))
        rec = dict(zip(TRACE_COLS, row[:14]))

        if rec["ac_dict"]:
            last_ac.update(rec["ac_dict"])

        alt_raw = rec["alt_raw"]
        on_ground = alt_raw == "ground"
        alt_m = None if (on_ground or alt_raw is None) else round(float(alt_raw) * 0.3048, 1)
        alt_geom_m = (
            round(float(rec["alt_geom"]) * 0.3048, 1)
            if rec["alt_geom"] not in (None, "ground") else None
        )

        rows.append({
            "source_type": SOURCE_TYPE,
            "source_id": icao,
            "timestamp_utc": (file_ts + rec["t_offset"]) if file_ts is not None else None,
            "lat": rec["lat"],
            "lon": rec["lon"],
            "alt": alt_m,
            "alt_geom_m": alt_geom_m,
            "on_ground": on_ground,
            "label": None,
            "ground_speed_ms": round(float(rec["gs"]) * 0.5144, 2) if rec["gs"] is not None else None,
            "track_deg": rec["track"],
            "vertical_rate_ms": round(float(rec["vrate"]) * 0.00508, 3) if rec["vrate"] is not None else None,
            "indicated_airspeed_ms": round(float(rec["ias"]) * 0.5144, 2) if rec["ias"] is not None else None,
            "roll_deg": rec["roll"],
            "flags_stale": bool(rec["flags"] & 1) if rec["flags"] is not None else None,
            "flags_new_leg": bool(rec["flags"] & 2) if rec["flags"] is not None else None,
            "ads_source_type": rec["ads_source_type"],
            "registration": data.get("r"),
            "aircraft_type": data.get("t"),
            "aircraft_desc": data.get("desc"),
            "no_reg_data": bool(data.get("noRegData", False)),
            "flight_callsign": (last_ac.get("flight") or "").strip() or None,
            "category": last_ac.get("category"),
            "squawk": last_ac.get("squawk"),
            "emergency": last_ac.get("emergency"),
            "nic": last_ac.get("nic"),
            "rc": last_ac.get("rc"),
            "nac_p": last_ac.get("nac_p"),
            "sil": last_ac.get("sil"),
            "adsb_version": last_ac.get("version"),
        })

    return pd.DataFrame(rows)


def _parse_tar_fileobj(
    fileobj: io.IOBase,
    tar_name: str,
    *,
    batch_size: int,
    client: ObjectStoreClient | None,
) -> list[str]:
    """Stream-parse one tar (from any file-like object), writing Silver per batch."""
    uris: list[str] = []
    part_num = 0
    total_rows = 0
    errors = 0
    batch_dfs: list[pd.DataFrame] = []

    def _flush() -> None:
        nonlocal batch_dfs, part_num, total_rows
        if not batch_dfs:
            return
        batch = pd.concat(batch_dfs, ignore_index=True)
        batch = add_provenance(
            batch, source_type=SOURCE_TYPE, source_file=tar_name, schema_version="silver_v1"
        )
        uri = write_silver(batch, SOURCE_TYPE, client=client)
        uris.append(uri)
        total_rows += len(batch)
        logger.info("Silver part %05d: %d rows (total %d so far)", part_num, len(batch), total_rows)
        part_num += 1
        batch_dfs.clear()
        gc.collect()

    with tarfile.open(fileobj=fileobj, mode="r:*") as tar:
        members = [
            m for m in tar.getmembers()
            if "traces" in m.name and (m.name.endswith(".json") or m.name.endswith(".json.gz"))
        ]
        logger.info("%s: %d trace member(s) found", tar_name, len(members))

        for i, m in enumerate(members):
            try:
                f = tar.extractfile(m)
                if f is None:
                    continue
                df = parse_trace_bytes(f.read())
                if len(df):
                    batch_dfs.append(df)
            except Exception:
                errors += 1
                if errors <= 10:
                    logger.warning("Error parsing %s", m.name, exc_info=True)

            if (i + 1) % batch_size == 0:
                logger.info("  Progress: %d/%d members", i + 1, len(members))
                _flush()

        _flush()

    logger.info(
        "Done %s: %d Silver part(s), %d total rows, %d error(s)",
        tar_name, part_num, total_rows, errors,
    )
    return uris


def parse_local_tar(
    tar_path: str | Path,
    *,
    batch_size: int = 300,
    client: ObjectStoreClient | None = None,
) -> list[str]:
    """Parse a local tar file and write Silver to MinIO. Returns s3:// URIs."""
    tar_path = Path(tar_path)
    logger.info("Opening local tar: %s", tar_path)
    with open(tar_path, "rb") as f:
        return _parse_tar_fileobj(f, tar_path.name, batch_size=batch_size, client=client)


def run(
    bronze_prefix: str = "adsblol_historical/",
    *,
    batch_size: int = 300,
    client: ObjectStoreClient | None = None,
    bronze_bucket: str | None = None,
) -> list[str]:
    """Download all tars from MinIO Bronze and parse each to Silver.

    NOTE: downloads each tar fully into memory (BytesIO) before processing.
    For 3GB+ tars this requires sufficient RAM. Use parse_local_tar() directly
    during development to avoid the download overhead.
    """
    client = client or get_minio_client()
    bronze_bucket = bronze_bucket or os.getenv("MINIO_BRONZE_BUCKET", "bronze")

    tar_objects = [
        obj.object_name
        for obj in client.list_objects(bronze_bucket, prefix=bronze_prefix, recursive=True)
        if obj.object_name.endswith(".tar")
    ]

    if not tar_objects:
        logger.warning("No .tar objects found under %s/%s", bronze_bucket, bronze_prefix)
        return []

    logger.info("Found %d tar(s): %s", len(tar_objects), tar_objects)
    all_uris: list[str] = []

    for tar_object in tar_objects:
        tar_name = tar_object.split("/")[-1]
        logger.info("Downloading %s from MinIO bronze/%s ...", tar_name, bronze_prefix)
        data = download_raw_bytes(client, tar_object, bucket=bronze_bucket)
        uris = _parse_tar_fileobj(
            io.BytesIO(data), tar_name, batch_size=batch_size, client=client
        )
        all_uris.extend(uris)

    return all_uris


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="adsb.lol historical Bronze tar → Silver Parquet")
    parser.add_argument("--local-tar", help="Parse a local .tar file directly (skips MinIO download)")
    parser.add_argument("--bronze-prefix", default="adsblol_historical/", help="MinIO Bronze prefix")
    parser.add_argument("--batch-size", type=int, default=300, help="Aircraft per Silver Parquet part")
    args = parser.parse_args()

    if args.local_tar:
        parse_local_tar(args.local_tar, batch_size=args.batch_size)
    else:
        run(args.bronze_prefix, batch_size=args.batch_size)
