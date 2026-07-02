"""
parse_alfa.py -- ALFA Silver parser.

Moved from `src/bronze2silverParsers/parse_alfa.py` per ADR-003
(docs/PIPELINE_PLAN.md, ANIL REHBERİ): the transform logic below
(`infer_fault_from_seq_name`, `find_col`, `parse_sequence`, `EXTRA_TOPICS`) is
UNCHANGED from that file -- only the IO layer changed, from a local zip path
+ local parquet output to downloading the raw zip from MinIO Bronze
(`bronze/alfa/*.zip`) and writing the result to MinIO Silver.

ALFA processed.zip icindeki her sequence klasorunu (carbonZ_...) bulur,
o klasordeki tum per-topic CSV'leri zaman ekseninde (merge_asof) birlestirir,
failure_status-*.csv dosyalarindan etiket (label) uretir.

Klasor adi = sequence_id (ALFA'da dosya adi formati: <klasor_adi>-<topic>.csv
oldugu icin ek regex'e gerek yok).

Kullanim:
    python -m src.silver.parse_alfa [--bronze-object alfa/processed.zip] [--local-out silver/alfa.parquet]
"""

from __future__ import annotations

import logging
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

SOURCE_TYPE = "alfa"


def infer_fault_from_seq_name(seq_name: str) -> str:
    s = seq_name.lower()
    if "no_failure" in s:
        return "normal"
    if "no_ground_truth" in s:
        return "unknown"
    if "aileron" in s and "rudder" in s:
        return "aileron_rudder_fault"
    if "aileron" in s:
        return "aileron_fault"
    if "rudder" in s:
        return "rudder_fault"
    if "elevator" in s:
        return "elevator_fault"
    if "engine" in s:
        return "engine_fault"
    return "unknown"


def find_col(df: pd.DataFrame, keys):
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in keys):
            return c
    return None


def read_csv_member(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_csv(f)


# measured/commanded ciftleri tasiyan topic'ler (find_col ile kolon bulunur).
# NOT: nav_info-velocity bu listede DEGIL -- gercek CSV'de kolon adlari
# "field.meas_x"/"field.des_x" oldugu icin find_col(['measured']) hic eslesmiyordu
# (velocity_mps'in bos kalmasinin kok nedeni, 2026-07-02'de gercek zip'le dogrulandi).
# Asagidaki FIXED_COL_TOPICS'te dogru kolon adlariyla ele aliniyor.
EXTRA_TOPICS = ["nav_info-roll", "nav_info-pitch", "nav_info-airspeed", "nav_info-yaw"]

# Sabit kolon adlariyla merge edilen topic'ler: {topic_suffix: {csv_col: silver_col}}.
# nav_info-errors ve path_dev otopilotun KENDI hesapladigi hata sinyalleri --
# feature engineering'de "hazir residual" olarak birinci sinif girdiler
# (docs/PIPELINE_PLAN, FableChat karari: Katman-1 residual'lar).
# vfr_hud throttle/groundspeed/climb tasir: enerji tutarliligi ve Gold
# velocity_mps/vertical_rate_mps boslugunu kapatir.
FIXED_COL_TOPICS: dict[str, dict[str, str]] = {
    "nav_info-errors": {
        "field.alt_error": "alt_error",
        "field.aspd_error": "aspd_error",
        "field.xtrack_error": "xtrack_error",
        "field.wp_dist": "wp_dist",
    },
    "mavctrl-path_dev": {
        "field.x": "path_dev_x",
        "field.y": "path_dev_y",
        "field.z": "path_dev_z",
    },
    "vfr_hud": {
        "field.airspeed": "hud_airspeed_ms",
        "field.groundspeed": "ground_speed_ms",
        "field.throttle": "throttle",
        "field.climb": "climb_rate_ms",
    },
    "nav_info-velocity": {
        "field.meas_x": "vel_meas_x",
        "field.meas_y": "vel_meas_y",
        "field.meas_z": "vel_meas_z",
        "field.des_x": "vel_des_x",
        "field.des_y": "vel_des_y",
        "field.des_z": "vel_des_z",
    },
}


def parse_sequence(zf: zipfile.ZipFile, seq_name: str, files_by_seq: dict):
    files = files_by_seq[seq_name]

    topic_files = {}
    for fname in files:
        stem = Path(fname).stem  # "<seq>-<topic>"
        prefix = seq_name + "-"
        if not stem.startswith(prefix):
            continue
        topic = stem[len(prefix):]
        topic_files[topic] = fname

    pos_topic = next((t for t in topic_files if "global_position-global" in t), None)
    if pos_topic is None:
        pos_topic = next((t for t in topic_files if t.startswith("position-global")), None)
    if pos_topic is None:
        return None

    base = read_csv_member(zf, topic_files[pos_topic])
    if "%time" not in base.columns:
        return None
    base = base.rename(columns={"%time": "ts_ns"})
    base["ts_ns"] = base["ts_ns"].astype("int64")
    base = base.sort_values("ts_ns")

    lat_col = find_col(base, ["latitude"])
    lon_col = find_col(base, ["longitude"])
    alt_col = find_col(base, ["altitude"])

    out = pd.DataFrame({
        "ts_ns": base["ts_ns"],
        "lat": base[lat_col] if lat_col else np.nan,
        "lon": base[lon_col] if lon_col else np.nan,
        "alt": base[alt_col] if alt_col else np.nan,
    })

    for topic_key in EXTRA_TOPICS:
        match = next((t for t in topic_files if t == topic_key or t.endswith(topic_key)), None)
        if match is None:
            continue
        try:
            extra = read_csv_member(zf, topic_files[match])
        except Exception:
            continue
        if "%time" not in extra.columns:
            continue
        extra = extra.rename(columns={"%time": "ts_ns"}).sort_values("ts_ns")
        meas_col = find_col(extra, ["measured"])
        cmd_col = find_col(extra, ["commanded"])

        short = topic_key.split("-")[-1]
        cols_to_take, rename = ["ts_ns"], {}
        if meas_col:
            cols_to_take.append(meas_col)
            rename[meas_col] = f"{short}_measured"
        if cmd_col:
            cols_to_take.append(cmd_col)
            rename[cmd_col] = f"{short}_commanded"
        if len(cols_to_take) == 1:
            continue

        extra_small = extra[cols_to_take].rename(columns=rename)
        out = pd.merge_asof(out.sort_values("ts_ns"), extra_small,
                             on="ts_ns", direction="nearest",
                             tolerance=500_000_000)  # 0.5 sn (ns)

    for topic_key, col_map in FIXED_COL_TOPICS.items():
        match = next((t for t in topic_files if t == topic_key or t.endswith(topic_key)), None)
        if match is None:
            continue
        try:
            extra = read_csv_member(zf, topic_files[match])
        except Exception:
            continue
        if "%time" not in extra.columns:
            continue
        extra = extra.rename(columns={"%time": "ts_ns"}).sort_values("ts_ns")
        cols_present = {src: dst for src, dst in col_map.items() if src in extra.columns}
        if not cols_present:
            continue
        extra_small = extra[["ts_ns"] + list(cols_present)].rename(columns=cols_present)
        out = pd.merge_asof(out.sort_values("ts_ns"), extra_small,
                             on="ts_ns", direction="nearest",
                             tolerance=500_000_000)  # 0.5 sn (ns)

    # nav_info-velocity bilesenlerinden hiz buyuklugu: Gold velocity_mps buradan beslenir.
    if all(c in out.columns for c in ["vel_meas_x", "vel_meas_y", "vel_meas_z"]):
        out["velocity_measured"] = np.sqrt(
            out["vel_meas_x"] ** 2 + out["vel_meas_y"] ** 2 + out["vel_meas_z"] ** 2)
    if all(c in out.columns for c in ["vel_des_x", "vel_des_y", "vel_des_z"]):
        out["velocity_commanded"] = np.sqrt(
            out["vel_des_x"] ** 2 + out["vel_des_y"] ** 2 + out["vel_des_z"] ** 2)

    fault_default = infer_fault_from_seq_name(seq_name)
    out["label"] = fault_default

    fs_topics = [t for t in topic_files if t.startswith("failure_status-")]
    for t in fs_topics:
        try:
            fs = read_csv_member(zf, topic_files[t])
        except Exception:
            continue
        if "%time" not in fs.columns or "field.data" not in fs.columns:
            continue
        active = fs[fs["field.data"].astype(float) != 0]
        if active.empty:
            continue
        fault_start = active["%time"].astype("int64").min()
        # NORMALIZE: dosya adi "engines" (cogul) veriyor ama klasor-adi-tabanli
        # varsayilan etiket "engine_fault" (tekil) uretiyordu -- iki farkli
        # kategori olarak gorunuyorlardi, ayni seyi ifade ediyorlar.
        raw_name = t.replace("failure_status-", "")
        if raw_name == "engines":
            raw_name = "engine"
        fault_label = raw_name + "_fault"
        out.loc[out["ts_ns"] >= fault_start, "label"] = fault_label

    out["source_type"] = "alfa"
    out["source_id"] = seq_name
    out["timestamp_utc"] = out["ts_ns"] / 1e9

    return out


def parse_zip_bytes(data: bytes) -> pd.DataFrame:
    """Parse an in-memory ALFA processed.zip into the flat Silver table."""
    with zipfile.ZipFile(BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        files_by_seq = defaultdict(list)
        for n in names:
            seq = Path(n).parent.name
            files_by_seq[seq].append(n)

        logger.info("%d sequence bulundu", len(files_by_seq))

        results = []
        for i, seq_name in enumerate(sorted(files_by_seq)):
            try:
                df = parse_sequence(zf, seq_name, files_by_seq)
                if df is not None and len(df):
                    results.append(df)
                    counts = df["label"].value_counts().to_dict()
                    logger.info("  [%d/%d] %s: %d satir, %s", i + 1, len(files_by_seq), seq_name, len(df), counts)
                else:
                    logger.warning("  [%d/%d] %s: konum bulunamadi, atlandi", i + 1, len(files_by_seq), seq_name)
            except Exception:
                logger.exception("  [%d/%d] %s: HATA", i + 1, len(files_by_seq), seq_name)

        if not results:
            logger.error("Hicbir sequence parse edilemedi.")
            return pd.DataFrame()

        full = pd.concat(results, ignore_index=True)
        logger.info("Toplam satir: %d, sequence sayisi: %d", len(full), full["source_id"].nunique())
        return full


def _find_bronze_zip(client: ObjectStoreClient) -> str | None:
    candidates = [n for n in list_layer_objects(client, "bronze", SOURCE_TYPE) if n.lower().endswith(".zip")]
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning("Multiple ALFA zips found under bronze/alfa/, using the first: %s", candidates)
    return candidates[0]


def build_alfa_silver(client: ObjectStoreClient, *, bronze_object: str | None = None) -> pd.DataFrame:
    """Download the ALFA zip from Bronze and parse it into the ALFA Silver table."""
    bronze_object = bronze_object or _find_bronze_zip(client)
    if bronze_object is None:
        logger.warning("No ALFA zip found under bronze/alfa/")
        return pd.DataFrame()

    data = download_raw_bytes(client, bronze_object)
    df = parse_zip_bytes(data)
    if df.empty:
        return df
    return add_provenance(df, source_type=SOURCE_TYPE, source_file=bronze_object)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ALFA Bronze zip -> Silver")
    parser.add_argument("--bronze-object", default=None, help="e.g. alfa/processed.zip; auto-detected if omitted")
    parser.add_argument("--local-out", default=None, help="Optional local Parquet path")
    args = parser.parse_args()

    client = get_minio_client()
    silver = build_alfa_silver(client, bronze_object=args.bronze_object)
    if silver.empty:
        logger.error("Nothing to write: ALFA Silver is empty")
        return

    uri = write_silver(silver, SOURCE_TYPE, client=client)
    logger.info("Wrote ALFA Silver -> %s", uri)

    if args.local_out:
        Path(args.local_out).parent.mkdir(parents=True, exist_ok=True)
        silver.to_parquet(args.local_out, index=False)
        logger.info("Local copy written: %s", args.local_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
