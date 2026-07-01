"""
parse_uav_attack.py -- UAV Attack Silver parser.

Moved from `src/bronze2silverParsers/parse_uav_attack.py` per ADR-003
(docs/PIPELINE_PLAN.md, ANIL REHBERİ): IO layer changed (download the raw zip
from MinIO Bronze instead of a local path, write to MinIO Silver instead of a
local parquet). One transform fix kept (not a rewrite of the rest): the
topic-suffix regex (`TOPIC_SUFFIX_PATTERN`/`split_log_and_topic`) is proven
broken against real files -- verified against the real 683.9 MB IEEE
DataPort zip on 2026-07-01 (see src/processing/uav_attack_silver.py's
docstring for the full investigation). Log-id prefixes are not uniform
across collections (`log_12_2020-8-2-14-18-24_...` for Simulated,
`ace-benign-log_0_2033-8-19-16-27-30_...` for Live, `001-2021-01-27-09-08-
37-708_...` for live Ping DoS) and contain their own underscores, so the old
lazy "last `_<word>_<n>.csv`" regex, searched left-to-right, matched the
FIRST underscore instead of the true topic boundary. Fixed by matching each
of the 4 known topic names this parser actually merges, anchored to the
filename's end -- everything else (quaternion->euler, label-from-path,
battery/gps merge) is unchanged.

UAVAttackData.zip icindeki ulog2csv ciktilarini (her log icin onlarca
per-topic CSV) log bazinda gruplar; vehicle_global_position'i omurga
(zaman/lat/lon/alt) olarak alir, vehicle_attitude (quaternion->euler),
battery_status ve vehicle_gps_position (jamming_indicator, noise_per_ms,
hdop, satellites_used) ile zenginlestirir.

ONEMLI: PX4 "timestamp" kolonu gercek UTC DEGIL, acilistan beri gecen
mikrosaniye sayacidir. Gercek UTC icin vehicle_gps_position.time_utc_usec
kullanilir (varsa); yoksa timestamp_utc sadece o log icinde goreceli kalir.

Etiket dosya yolundaki anahtar kelimelerden (benign/spoofing/jamming)
cikarilir -- klasor derinligi onemli degil, sadece en yakin klasor bakilir.

Kullanim:
    python -m src.silver.parse_uav_attack [--bronze-object uav_attack/UAVAttackData.zip] [--local-out silver/uav_attack.parquet]
"""

from __future__ import annotations

import logging
import re
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.minio_io import (
    ObjectStoreClient,
    download_raw_bytes,
    get_minio_client,
    list_layer_objects,
    write_silver,
)
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

SOURCE_TYPE = "uav_attack"

# Fix (verified against real data 2026-07-01): match each known topic name exactly,
# anchored to the filename's end, instead of a generic lazy "last _<word>_<n>.csv" regex
# (which mis-split on the first underscore for non-uniform log_id prefixes).
_KNOWN_TOPICS = ("vehicle_global_position", "vehicle_attitude", "battery_status", "vehicle_gps_position")
_TOPIC_PATTERNS = {topic: re.compile(rf"_{re.escape(topic)}_(\d+)\.csv$", re.IGNORECASE) for topic in _KNOWN_TOPICS}


def split_log_and_topic(filename: str):
    name = Path(filename).name
    for topic, pattern in _TOPIC_PATTERNS.items():
        m = pattern.search(name)
        if m:
            return name[: m.start()], topic
    return None, None


def infer_label_from_path(path: str) -> str:
    # ONEMLI: tum path'e degil, SADECE EN YAKIN klasor adina bak.
    # Ust klasor "GPS Spoofing and Jamming" olsa bile alt klasor "Benign Flight"
    # ise gercek etiket benign'dir -- tum path'i taramak yanlis pozitif uretiyordu
    # (orn. "Live GPS Spoofing and Jamming/Benign Flight/..." icinde "spoof"
    # gectigi icin benign loglar yanlislikla gps_spoofing olarak etiketleniyordu).
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    nearest = parts[-1].lower() if parts else ""

    if "benign" in nearest or "normal" in nearest:
        return "benign"
    if "spoof" in nearest:
        return "gps_spoofing"
    if "jam" in nearest:
        return "gps_jamming"
    if "ping" in nearest or "dos" in nearest:
        return "ping_dos"
    if "malicious" in nearest or "attack" in nearest:
        return "malicious_unspecified"

    # En yakin klasorde eslesme yoksa, guvenlik icin bir ust klasore de bak
    # (bazi log'lar dogrudan senaryo klasorunun icinde olabilir, alt klasorsuz).
    joined = "/".join(parts).lower()
    if "benign" in joined or "normal" in joined:
        return "benign"
    if "spoof" in joined:
        return "gps_spoofing"
    if "jam" in joined:
        return "gps_jamming"
    if "ping" in joined or "dos" in joined:
        return "ping_dos"
    return "unknown"


def quat_to_euler_deg(q0, q1, q2, q3):
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
    pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1, 1))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def read_csv_member(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_csv(f)


def merge_topic(base, zf, files_for_log, log_id, topic, cols_keep,
                 rename=None, tol_us=200_000):
    match = next((f for f in files_for_log if split_log_and_topic(f) == (log_id, topic)), None)
    if match is None:
        return base
    try:
        df = read_csv_member(zf, match)
    except Exception:
        return base
    if "timestamp" not in df.columns:
        return base
    cols_present = [c for c in cols_keep if c in df.columns]
    if not cols_present:
        return base
    small = df[["timestamp"] + cols_present].sort_values("timestamp")
    if rename:
        small = small.rename(columns=rename)
    return pd.merge_asof(base.sort_values("timestamp"), small,
                          on="timestamp", direction="nearest", tolerance=tol_us)


def parse_log(zf, log_id: str, label: str, files_for_log: list):
    pos_file = next((f for f in files_for_log
                      if split_log_and_topic(f) == (log_id, "vehicle_global_position")), None)
    if pos_file is None:
        return None

    base = read_csv_member(zf, pos_file)
    needed = ["timestamp", "lat", "lon", "alt"]
    if not all(c in base.columns for c in needed):
        return None
    extra_cols = [c for c in ["eph", "epv"] if c in base.columns]
    base = base[needed + extra_cols].sort_values("timestamp")

    base = merge_topic(base, zf, files_for_log, log_id, "vehicle_attitude",
                        ["q[0]", "q[1]", "q[2]", "q[3]"])
    if all(c in base.columns for c in ["q[0]", "q[1]", "q[2]", "q[3]"]):
        roll, pitch, yaw = quat_to_euler_deg(
            base["q[0]"], base["q[1]"], base["q[2]"], base["q[3]"])
        base["roll_deg"], base["pitch_deg"], base["yaw_deg"] = roll, pitch, yaw
        base = base.drop(columns=["q[0]", "q[1]", "q[2]", "q[3]"])

    base = merge_topic(base, zf, files_for_log, log_id, "battery_status",
                        ["voltage_v", "remaining", "current_a"])

    base = merge_topic(
        base, zf, files_for_log, log_id, "vehicle_gps_position",
        ["jamming_indicator", "noise_per_ms", "hdop", "vdop",
         "satellites_used", "s_variance_m_s", "eph", "epv", "time_utc_usec"],
        rename={"eph": "raw_gps_eph", "epv": "raw_gps_epv"},
    )

    base["source_type"] = "uav_attack"
    base["source_id"] = log_id
    base["label"] = label

    if "time_utc_usec" in base.columns and base["time_utc_usec"].notna().any():
        base["timestamp_utc"] = base["time_utc_usec"] / 1e6
        base["timestamp_is_real_utc"] = True
    else:
        # acilistan beri gecen sure -- gercek UTC degil, sadece log-ici siralama icin
        base["timestamp_utc"] = base["timestamp"] / 1e6
        base["timestamp_is_real_utc"] = False

    return base


def parse_zip_bytes(data: bytes) -> pd.DataFrame:
    """Parse an in-memory UAVAttackData.zip into the flat Silver table."""
    with zipfile.ZipFile(BytesIO(data)) as zf:
        all_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]

        by_folder = defaultdict(list)
        for n in all_names:
            by_folder[str(Path(n).parent)].append(n)

        results = []
        total_logs = 0
        for folder, files in by_folder.items():
            label = infer_label_from_path(folder)
            log_ids = sorted({lid for lid, _ in
                              (split_log_and_topic(f) for f in files) if lid})
            for log_id in log_ids:
                total_logs += 1
                try:
                    df = parse_log(zf, log_id, label, files)
                    if df is not None and len(df):
                        results.append(df)
                        utc_flag = "gercek-UTC" if df["timestamp_is_real_utc"].iloc[0] else "GORECELI"
                        logger.info("  [%s] %s: %d satir, label=%s, zaman=%s", folder, log_id, len(df), label, utc_flag)
                    else:
                        logger.warning("  [%s] %s: vehicle_global_position bulunamadi, atlandi", folder, log_id)
                except Exception:
                    logger.exception("  [%s] %s: HATA", folder, log_id)

        if not results:
            logger.error("Hicbir log parse edilemedi.")
            return pd.DataFrame()

        full = pd.concat(results, ignore_index=True)
        logger.info(
            "Toplam satir: %d, toplam log: %d, parse edilen: %d",
            len(full), total_logs, full["source_id"].nunique(),
        )
        return full


def _find_bronze_zip(client: ObjectStoreClient) -> str | None:
    candidates = [n for n in list_layer_objects(client, "bronze", SOURCE_TYPE) if n.lower().endswith(".zip")]
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning("Multiple UAV Attack zips found under bronze/uav_attack/, using the first: %s", candidates)
    return candidates[0]


def build_uav_attack_silver(client: ObjectStoreClient, *, bronze_object: str | None = None) -> pd.DataFrame:
    """Download the UAV Attack zip from Bronze and parse it into the UAV Attack Silver table."""
    bronze_object = bronze_object or _find_bronze_zip(client)
    if bronze_object is None:
        logger.warning("No UAV Attack zip found under bronze/uav_attack/")
        return pd.DataFrame()

    data = download_raw_bytes(client, bronze_object)
    df = parse_zip_bytes(data)
    if df.empty:
        return df
    return add_provenance(df, source_type=SOURCE_TYPE, source_file=bronze_object)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="UAV Attack Bronze zip -> Silver")
    parser.add_argument("--bronze-object", default=None, help="e.g. uav_attack/UAVAttackData.zip; auto-detected if omitted")
    parser.add_argument("--local-out", default=None, help="Optional local Parquet path")
    args = parser.parse_args()

    client = get_minio_client()
    silver = build_uav_attack_silver(client, bronze_object=args.bronze_object)
    if silver.empty:
        logger.error("Nothing to write: UAV Attack Silver is empty")
        return

    uri = write_silver(silver, SOURCE_TYPE, client=client)
    logger.info("Wrote UAV Attack Silver -> %s", uri)

    if args.local_out:
        Path(args.local_out).parent.mkdir(parents=True, exist_ok=True)
        silver.to_parquet(args.local_out, index=False)
        logger.info("Local copy written: %s", args.local_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
