"""
parse_alfa.py
ALFA processed.zip icindeki her sequence klasorunu (carbonZ_...) bulur,
o klasordeki tum per-topic CSV'leri zaman ekseninde (merge_asof) birlestirir,
failure_status-*.csv dosyalarindan etiket (label) uretir.

Klasor adi = sequence_id (ALFA'da dosya adi formati: <klasor_adi>-<topic>.csv
oldugu icin ek regex'e gerek yok).

Kullanim:
    python parse_alfa.py <processed.zip yolu> [cikti.parquet]

Ornek:
    python parse_alfa.py alfa_nested_extracted/processed.zip silver/alfa_processed.parquet
"""
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


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


EXTRA_TOPICS = ["nav_info-roll", "nav_info-pitch", "nav_info-airspeed",
                "nav_info-velocity", "nav_info-yaw"]


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


def main(zip_path: str, out_path: str):
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        files_by_seq = defaultdict(list)
        for n in names:
            seq = Path(n).parent.name
            files_by_seq[seq].append(n)

        print(f"{len(files_by_seq)} sequence bulundu")

        results = []
        for i, seq_name in enumerate(sorted(files_by_seq)):
            try:
                df = parse_sequence(zf, seq_name, files_by_seq)
                if df is not None and len(df):
                    results.append(df)
                    counts = df["label"].value_counts().to_dict()
                    print(f"  [{i + 1}/{len(files_by_seq)}] {seq_name}: {len(df)} satir, {counts}")
                else:
                    print(f"  [{i + 1}/{len(files_by_seq)}] {seq_name}: konum bulunamadi, atlandi")
            except Exception as e:
                print(f"  [{i + 1}/{len(files_by_seq)}] {seq_name}: HATA {e}")

        if not results:
            print("Hicbir sequence parse edilemedi.")
            return

        full = pd.concat(results, ignore_index=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(out_path, index=False)
        print(f"\nYazildi: {out_path}")
        print(f"  Toplam satir: {len(full)}")
        print(f"  Sequence sayisi: {full['source_id'].nunique()}")
        print(full["label"].value_counts())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python parse_alfa.py <processed.zip> [cikti.parquet]")
        sys.exit(1)
    zip_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "silver/alfa_processed.parquet"
    main(zip_path, out)
