"""ALFA raw rosbag (.bag) -> ALFA Silver (ML-4 Faz B).

processed.zip'e hic girmemis 8 ucus raw/'da duruyor (scripts/inventory_alfa_raw.py
envanteri, 2026-07-02): 5'i failure_status hic aktiflesmemis NORMAL ADAYI
(3'u tam nav_info'lu, toplam ~47 dk -- mevcut ~66 dk normal havuzuna buyuk ek),
2'si failure aktif (ek anomali test ucusu), 1'i konumsuz (atlanir).

Cikti semasi parse_alfa.py ile AYNI kolonlar (ts_ns, lat/lon/alt,
*_measured/_commanded, alt_error/aspd_error/xtrack_error/wp_dist, path_dev_*,
hud_airspeed_ms/ground_speed_ms/throttle/climb_rate_ms, vel_meas_*/vel_des_*,
velocity_measured/commanded, label, source_type, source_id, timestamp_utc)
-- boylece build_alfa_features degismeden calisir ve Gold/feature katmanlari
iki kaynagi tek 'alfa' source_type'i altinda gorur.

Etiket: sequence adi degil (bag adlarinda senaryo eki yok), failure_status/*
mesajlarindan: hic True gelmemisse tum ucus 'normal'; gelmisse ilk True
zamanindan itibaren '<isim>_fault' (engines->engine normalizasyonu parse_alfa
ile ayni).

Mesaj alan adlari gercek bag'de dogrulandi (2026-07-02):
NavDataPair.measured/commanded, NavErrors.alt_error/aspd_error/xtrack_error/
wp_dist, NavVector3.meas_x../des_x.., VFR_HUD.airspeed/groundspeed/throttle/
climb, NavSatFix.latitude/longitude/altitude, Vector3.x/y/z, Bool.data.

Kullanim:
    python -m src.silver.parse_alfa_rosbag [--raw-dir <dir>] [--local-out data/silver/alfa_rosbag_silver.parquet]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.minio_io import get_minio_client, write_silver
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

SOURCE_TYPE = "alfa"
DEFAULT_RAW = r"C:/Users/PC_5812_YD26/Desktop/ALFA/raw/raw"
MERGE_TOL_NS = 500_000_000  # parse_alfa ile ayni (0.5 sn)

# topic -> (mesaj alanlari -> silver kolonlari)
PAIR_TOPICS = {
    "/mavros/nav_info/roll": "roll",
    "/mavros/nav_info/pitch": "pitch",
    "/mavros/nav_info/airspeed": "airspeed",
    "/mavros/nav_info/yaw": "yaw",
}
FIXED_TOPICS: dict[str, dict[str, str]] = {
    "/mavros/nav_info/errors": {"alt_error": "alt_error", "aspd_error": "aspd_error",
                                 "xtrack_error": "xtrack_error", "wp_dist": "wp_dist"},
    "/mavctrl/path_dev": {"x": "path_dev_x", "y": "path_dev_y", "z": "path_dev_z"},
    "/mavros/vfr_hud": {"airspeed": "hud_airspeed_ms", "groundspeed": "ground_speed_ms",
                         "throttle": "throttle", "climb": "climb_rate_ms"},
    "/mavros/nav_info/velocity": {"meas_x": "vel_meas_x", "meas_y": "vel_meas_y", "meas_z": "vel_meas_z",
                                   "des_x": "vel_des_x", "des_y": "vel_des_y", "des_z": "vel_des_z"},
}
POSITION_TOPIC = "/mavros/global_position/global"
FAILURE_PREFIX = "/failure_status/"


def read_bag_topics(bag_path: Path) -> dict[str, pd.DataFrame]:
    """Bag'den gereken topic'leri {topic: DataFrame(ts_ns, <alanlar>)} olarak ceker."""
    from rosbags.highlevel import AnyReader

    wanted_fields: dict[str, list[str]] = {POSITION_TOPIC: ["latitude", "longitude", "altitude"]}
    for t in PAIR_TOPICS:
        wanted_fields[t] = ["measured", "commanded"]
    for t, mapping in FIXED_TOPICS.items():
        wanted_fields[t] = list(mapping)

    out: dict[str, list] = {}
    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections
                 if c.topic in wanted_fields or c.topic.startswith(FAILURE_PREFIX)]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic.startswith(FAILURE_PREFIX):
                out.setdefault(conn.topic, []).append((ts, bool(getattr(msg, "data", False))))
            else:
                row = tuple(getattr(msg, f, np.nan) for f in wanted_fields[conn.topic])
                out.setdefault(conn.topic, []).append((ts, *row))

    frames: dict[str, pd.DataFrame] = {}
    for topic, rows in out.items():
        if topic.startswith(FAILURE_PREFIX):
            frames[topic] = pd.DataFrame(rows, columns=["ts_ns", "data"])
        else:
            frames[topic] = pd.DataFrame(rows, columns=["ts_ns"] + wanted_fields[topic])
    return frames


def assemble_flight(topic_dfs: dict[str, pd.DataFrame], flight_id: str) -> pd.DataFrame | None:
    """Topic DataFrame'lerini parse_alfa cikti semasindaki tek tabloya birlestirir.

    Saf fonksiyon (IO yok) -- testler sahte topic_dfs ile dogrudan cagirir.
    """
    pos = topic_dfs.get(POSITION_TOPIC)
    if pos is None or pos.empty:
        logger.warning("%s: global_position yok, atlandi", flight_id)
        return None

    base = pos.rename(columns={"latitude": "lat", "longitude": "lon", "altitude": "alt"})
    base = base.sort_values("ts_ns").reset_index(drop=True)
    base["ts_ns"] = base["ts_ns"].astype("int64")
    out = base[["ts_ns", "lat", "lon", "alt"]]

    for topic, short in PAIR_TOPICS.items():
        df = topic_dfs.get(topic)
        if df is None or df.empty:
            continue
        small = df.rename(columns={"measured": f"{short}_measured", "commanded": f"{short}_commanded"})
        small = small.sort_values("ts_ns")
        small["ts_ns"] = small["ts_ns"].astype("int64")
        out = pd.merge_asof(out.sort_values("ts_ns"), small,
                             on="ts_ns", direction="nearest", tolerance=MERGE_TOL_NS)

    for topic, mapping in FIXED_TOPICS.items():
        df = topic_dfs.get(topic)
        if df is None or df.empty:
            continue
        small = df.rename(columns=mapping).sort_values("ts_ns")
        small["ts_ns"] = small["ts_ns"].astype("int64")
        out = pd.merge_asof(out.sort_values("ts_ns"), small,
                             on="ts_ns", direction="nearest", tolerance=MERGE_TOL_NS)

    if all(c in out.columns for c in ["vel_meas_x", "vel_meas_y", "vel_meas_z"]):
        out["velocity_measured"] = np.sqrt(
            out["vel_meas_x"] ** 2 + out["vel_meas_y"] ** 2 + out["vel_meas_z"] ** 2)
    if all(c in out.columns for c in ["vel_des_x", "vel_des_y", "vel_des_z"]):
        out["velocity_commanded"] = np.sqrt(
            out["vel_des_x"] ** 2 + out["vel_des_y"] ** 2 + out["vel_des_z"] ** 2)

    # Etiket: failure_status'tan. Bag adinda senaryo eki olmadigindan varsayilan
    # 'normal'; ilk True'dan itibaren '<isim>_fault' (engines->engine, parse_alfa ile ayni).
    out["label"] = "normal"
    for topic, df in topic_dfs.items():
        if not topic.startswith(FAILURE_PREFIX) or df.empty:
            continue
        active = df[df["data"].astype(bool)]
        if active.empty:
            continue
        onset = int(active["ts_ns"].min())
        raw_name = topic.removeprefix(FAILURE_PREFIX)
        if raw_name == "engines":
            raw_name = "engine"
        out.loc[out["ts_ns"] >= onset, "label"] = f"{raw_name}_fault"

    out["source_type"] = SOURCE_TYPE
    out["source_id"] = flight_id
    out["timestamp_utc"] = out["ts_ns"] / 1e9
    return out


def parse_bags(raw_dir: Path, flight_ids: list[str]) -> pd.DataFrame:
    results = []
    for i, fid in enumerate(flight_ids, 1):
        bag = raw_dir / f"{fid}.bag"
        if not bag.exists():
            logger.warning("[%d/%d] %s: bag yok, atlandi", i, len(flight_ids), fid)
            continue
        try:
            df = assemble_flight(read_bag_topics(bag), fid)
        except Exception:
            logger.exception("[%d/%d] %s: HATA", i, len(flight_ids), fid)
            continue
        if df is not None and len(df):
            results.append(df)
            logger.info("[%d/%d] %s: %d satir, label dagilimi %s",
                        i, len(flight_ids), fid, len(df), df["label"].value_counts().to_dict())
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def processed_roots(silver_path: str = "data/silver/alfa_silver.parquet") -> set[str]:
    a = pd.read_parquet(silver_path, columns=["source_id"])
    return {"_".join(s.split("_")[:2]) for s in a["source_id"].unique()}


def main() -> None:
    parser = argparse.ArgumentParser(description="ALFA raw rosbag -> Silver (processed'e girmemis ucuslar)")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW)
    parser.add_argument("--local-out", default="data/silver/alfa_rosbag_silver.parquet")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    used = processed_roots()
    all_bags = sorted(p.stem for p in raw_dir.glob("*.bag"))
    missing = [b for b in all_bags if b not in used]
    logger.info("raw'da %d bag; processed'e girmemis %d ucus parse edilecek: %s",
                len(all_bags), len(missing), missing)

    silver = parse_bags(raw_dir, missing)
    if silver.empty:
        logger.error("Hicbir bag parse edilemedi")
        return
    silver = add_provenance(silver, source_type=SOURCE_TYPE, source_file="alfa_raw_rosbag/*.bag")

    client = get_minio_client()
    uri = write_silver(silver, SOURCE_TYPE, client=client)
    logger.info("Wrote ALFA rosbag Silver -> %s (%d satir, %d ucus)",
                uri, len(silver), silver["source_id"].nunique())

    if args.local_out:
        Path(args.local_out).parent.mkdir(parents=True, exist_ok=True)
        silver.to_parquet(args.local_out, index=False)
        logger.info("Local copy written: %s", args.local_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
