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
from typing import Callable

import pandas as pd

from src.common.minio_io import (
    ObjectStoreClient,
    delete_layer_objects,
    download_raw_bytes,
    get_minio_client,
    write_silver,
)
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

SOURCE_TYPE = "adsblol_historical"

# 2026-07-13 (kullanici istegi -- "dur deyince dursam ertesi gun kaldigi yerden
# devam edemez miyiz"): tar-bazli checkpoint dosyasi. Onceden run() HER
# cagrida TUM Silver'i silip TUM tar'lari sifirdan isliyordu (bkz. eski
# docstring notu) -- 11 tar icin bu kabul edilebilirdi ama 30 tar'a cikinca
# (~17 saat) kullanicinin PC'yi o kadar acik tutamamasi gercek bir sorun oldu.
# Artik HER Silver parcasi yazildikca checkpoint'e ekleniyor (part-duzeyinde
# degil ama flush-duzeyinde durabilite) -- bir tar tamamlaninca "completed"a
# tasiniyor. Islem KESILIRSE (Ctrl+C / kill / PC kapanmasi): tamamlanmis
# tar'lar checkpoint'te kalir, YARIM kalan tek tar'in parcalari bir sonraki
# calistirmada silinip o tar sifirdan islenir -- 30 tar'in TAMAMI degil,
# sadece kesinti anindaki tek tar kaybedilir.
CHECKPOINT_PATH = Path("data/state/silver_historical_checkpoint.json")


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"completed_tars": [], "in_progress": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Checkpoint dosyasi okunamadi (%s), sifirdan basliyor", path)
        return {"completed_tars": [], "in_progress": {}}
    state.setdefault("completed_tars", [])
    state.setdefault("in_progress", {})
    return state


def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _delete_uris(client: ObjectStoreClient, uris: list[str]) -> None:
    """Delete specific `s3://bucket/key` objects -- used to clean up an
    interrupted tar's partial Silver parts before reprocessing it."""
    for uri in uris:
        # s3://bucket/key -> (bucket, key)
        _, _, rest = uri.partition("s3://")
        bucket, _, key = rest.partition("/")
        try:
            client.remove_object(bucket, key)
        except Exception:
            logger.warning("Yarim kalan parca silinemedi: %s", uri, exc_info=True)

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

    # 2026-07-10 (kullanici istegi): adsb.lol/readsb'nin dbFlags bit alaninin
    # 1. biti askeri ucak demek (bkz. Dashboard/uav_producer.py'deki AYNI
    # mantik) -- dbFlags dosya-seviyesinde (icao/timestamp gibi), trace
    # icindeki HER satir icin sabit. Eksikse (ucak topluluk veritabaninda
    # yoksa, ~%10 vaka) varsayilan False -- "askeri OLDUGU DOGRULANMAMIS"
    # ile "sivil" ayni kefeye konur (Dashboard'daki ile ayni varsayim).
    try:
        is_military = bool(int(data.get("dbFlags", 0) or 0) & 1)
    except (TypeError, ValueError):
        is_military = False

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
            "is_military": is_military,
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
    on_part_written: Callable[[str], None] | None = None,
) -> list[str]:
    """Stream-parse one tar (from any file-like object), writing Silver per batch.

    `on_part_written(uri)`, if given, is called right after each part is durably
    written to Silver -- `run()` uses this to persist the checkpoint incrementally,
    so a kill mid-tar only loses that tar's (not-yet-flushed) partial batch.
    """
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
        if on_part_written is not None:
            on_part_written(uri)

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
    fresh: bool = False,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> list[str]:
    """Download all tars from MinIO Bronze and parse each to Silver.

    NOTE: downloads each tar fully into memory (BytesIO) before processing.
    For 3GB+ tars this requires sufficient RAM. Use parse_local_tar() directly
    during development to avoid the download overhead.

    Tar-level resume checkpoint (`checkpoint_path`, default
    `data/state/silver_historical_checkpoint.json`): already-completed tars are
    skipped on rerun, so stopping the process (Ctrl+C, `Stop-Process`, a PC
    shutdown) and restarting the next day resumes from the next unfinished tar
    instead of reprocessing everything from scratch. Only the ONE tar that was
    in-flight when interrupted is redone (its partial Silver parts, tracked in
    the checkpoint, are deleted first so they don't sit alongside the retry's
    fresh parts). Pass `fresh=True` (or `--fresh` on the CLI) to ignore the
    checkpoint and reprocess every tar from scratch, e.g. after a parsing-logic
    change that invalidates all existing Silver output.
    """
    client = client or get_minio_client()
    bronze_bucket = bronze_bucket or os.getenv("MINIO_BRONZE_BUCKET", "bronze")
    silver_bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")

    if fresh:
        cleared = delete_layer_objects(client, silver_bucket, SOURCE_TYPE)
        if cleared:
            logger.info("--fresh: cleared %d stale Silver part(s), ignoring checkpoint", cleared)
        state = {"completed_tars": [], "in_progress": {}}
        _save_checkpoint(checkpoint_path, state)
    else:
        state = _load_checkpoint(checkpoint_path)
        for tar_name, partial_uris in list(state["in_progress"].items()):
            logger.info(
                "Onceki calistirma '%s' islenirken kesilmis -- %d yarim parca siliniyor, tar sifirdan islenecek",
                tar_name, len(partial_uris),
            )
            _delete_uris(client, partial_uris)
            del state["in_progress"][tar_name]
        _save_checkpoint(checkpoint_path, state)

    tar_objects = [
        obj.object_name
        for obj in client.list_objects(bronze_bucket, prefix=bronze_prefix, recursive=True)
        if obj.object_name.endswith(".tar")
    ]

    if not tar_objects:
        logger.warning("No .tar objects found under %s/%s", bronze_bucket, bronze_prefix)
        return []

    logger.info("Found %d tar(s): %s", len(tar_objects), tar_objects)
    completed = set(state["completed_tars"])
    all_uris: list[str] = []

    for tar_object in tar_objects:
        tar_name = tar_object.split("/")[-1]
        if tar_name in completed:
            logger.info("'%s' zaten tamamlanmis (checkpoint), atlaniyor", tar_name)
            continue

        logger.info("Downloading %s from MinIO bronze/%s ...", tar_name, bronze_prefix)
        state["in_progress"][tar_name] = []
        _save_checkpoint(checkpoint_path, state)

        def _on_part_written(uri: str, _tar_name: str = tar_name) -> None:
            state["in_progress"][_tar_name].append(uri)
            _save_checkpoint(checkpoint_path, state)

        data = download_raw_bytes(client, tar_object, bucket=bronze_bucket)
        uris = _parse_tar_fileobj(
            io.BytesIO(data), tar_name, batch_size=batch_size, client=client,
            on_part_written=_on_part_written,
        )
        all_uris.extend(uris)

        del state["in_progress"][tar_name]
        state["completed_tars"].append(tar_name)
        _save_checkpoint(checkpoint_path, state)

    return all_uris


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="adsb.lol historical Bronze tar → Silver Parquet")
    parser.add_argument("--local-tar", help="Parse a local .tar file directly (skips MinIO download)")
    parser.add_argument("--bronze-prefix", default="adsblol_historical/", help="MinIO Bronze prefix")
    parser.add_argument("--batch-size", type=int, default=300, help="Aircraft per Silver Parquet part")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore the tar-level checkpoint and reprocess every tar from scratch",
    )
    args = parser.parse_args()

    if args.local_tar:
        parse_local_tar(args.local_tar, batch_size=args.batch_size)
    else:
        run(args.bronze_prefix, batch_size=args.batch_size, fresh=args.fresh)
