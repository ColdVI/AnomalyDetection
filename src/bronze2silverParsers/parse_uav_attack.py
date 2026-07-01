"""
parse_uav_attack.py
UAVAttackData.zip icindeki ulog2csv ciktilarini (her log icin onlarca
per-topic CSV) log bazinda gruplar; vehicle_global_position'i omurga
(zaman/lat/lon/alt) olarak alir, vehicle_attitude (quaternion->euler),
battery_status ve vehicle_gps_position (jamming_indicator, noise_per_ms,
hdop, satellites_used) ile zenginlestirir.

ONEMLI: PX4 "timestamp" kolonu gercek UTC DEGIL, acilistan beri gecen
mikrosaniye sayacidir. Gercek UTC icin vehicle_gps_position.time_utc_usec
kullanilir (varsa); yoksa timestamp_utc sadece o log icinde goreceli kalir.

Etiket dosya yolundaki anahtar kelimelerden (benign/spoofing/jamming)
cikarilir -- klasor derinligi onemli degil, tum path taranir.

Kullanim:
    python parse_uav_attack.py <UAVAttackData.zip> [cikti.parquet]
"""
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TOPIC_SUFFIX_PATTERN = re.compile(r"_([a-z0-9_]+?)_(\d+)\.csv$", re.IGNORECASE)


def split_log_and_topic(filename: str):
    name = Path(filename).name
    m = TOPIC_SUFFIX_PATTERN.search(name)
    if not m:
        return None, None
    log_id = name[: m.start()]
    return log_id, m.group(1)


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


def main(zip_path: str, out_path: str):
    with zipfile.ZipFile(zip_path) as zf:
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
                        print(f"  [{folder}] {log_id}: {len(df)} satir, label={label}, zaman={utc_flag}")
                    else:
                        print(f"  [{folder}] {log_id}: vehicle_global_position bulunamadi, atlandi")
                except Exception as e:
                    print(f"  [{folder}] {log_id}: HATA {e}")

        if not results:
            print("Hicbir log parse edilemedi.")
            return

        full = pd.concat(results, ignore_index=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(out_path, index=False)
        print(f"\nYazildi: {out_path}")
        print(f"  Toplam satir: {len(full)}")
        print(f"  Toplam log: {total_logs}  |  parse edilen: {full['source_id'].nunique()}")
        print(full["label"].value_counts())
        print(f"\n  Gercek UTC zamanli log sayisi: {full.groupby('source_id')['timestamp_is_real_utc'].first().sum()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python parse_uav_attack.py <UAVAttackData.zip> [cikti.parquet]")
        sys.exit(1)
    zip_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "silver/uav_attack.parquet"
    main(zip_path, out)
