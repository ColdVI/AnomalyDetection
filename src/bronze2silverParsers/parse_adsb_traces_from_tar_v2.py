"""
parse_adsb_traces_from_tar_v2.py
Onceki versiyonun bellek-guvenli hali.

SORUN: v1 tum dosyalari isleyip tek bir dev DataFrame'de biriktiriyordu.
47K dosya icin bu, 8 GB RAM'i kolayca doldurup sistemi kilitleyebiliyor.

COZUM: Dosyalar BATCH_SIZE'lik gruplar halinde islenir, her batch bitince
diske (parquet) yazilir ve bellekten atilir (df = None + gc.collect()).
Boylece toplam dosya sayisi ne olursa olsun, bellekte ayni anda en fazla
BATCH_SIZE dosyanin verisi tutulur.

Cikti TEK bir parquet dosyasi degil, bir KLASOR olur (icinde parca parca
part-0000.parquet, part-0001.parquet ...). Bunlari sonra tek seferde
okumak icin pandas/pyarrow zaten "dataset" olarak coklu-dosya parquet
okuyabiliyor -- ayri bir birlestirme adimina gerek yok.

Kullanim:
    python parse_adsb_traces_from_tar_v2.py <tar_yolu> [cikti_klasoru] [batch_size] [limit]

Ornek (guvenli, 8GB RAM icin):
    python parse_adsb_traces_from_tar_v2.py "v2026.06.28-planes-readsb-prod-0.tar" silver/adsb_traces 300

Test (ilk 1000 dosya):
    python parse_adsb_traces_from_tar_v2.py "v2026.06.28-planes-readsb-prod-0.tar" silver/adsb_traces 300 1000
"""
import gc
import gzip
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

TRACE_COLS = [
    "t_offset", "lat", "lon", "alt_raw", "gs", "track", "flags", "vrate",
    "ac_dict", "ads_source_type", "alt_geom", "vrate_geom", "ias", "roll",
]


def parse_trace_bytes(raw: bytes) -> pd.DataFrame:
    try:
        data = json.loads(gzip.decompress(raw))
    except OSError:
        data = json.loads(raw)

    icao = data.get("icao")
    file_ts = data.get("timestamp")
    trace = data.get("trace", [])

    rows = []
    last_ac = {}
    for row in trace:
        row = list(row) + [None] * (14 - len(row))
        rec = dict(zip(TRACE_COLS, row[:14]))

        if rec["ac_dict"]:
            last_ac.update(rec["ac_dict"])

        alt_raw = rec["alt_raw"]
        on_ground = alt_raw == "ground"
        alt_m = None if (on_ground or alt_raw is None) else round(float(alt_raw) * 0.3048, 1)
        alt_geom_m = (round(float(rec["alt_geom"]) * 0.3048, 1)
                      if rec["alt_geom"] not in (None, "ground") else None)

        rows.append({
            "source_type": "adsb",
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


def main(tar_path: str, out_dir: str, batch_size: int = 300, limit: int = None):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_files_ok = 0
    errors = 0
    part_num = 0
    seen_aircraft = set()

    with tarfile.open(tar_path, "r") as tar:
        print("Tar basliklari taraniyor (icerik degil, sadece dosya listesi)...")
        members = [m for m in tar.getmembers()
                   if "traces" in m.name
                   and (m.name.endswith(".json") or m.name.endswith(".json.gz"))]
        print(f"{len(members)} trace dosyasi bulundu")

        if limit:
            members = members[:limit]
            print(f"Test modu: ilk {limit} dosya islenecek")

        batch_dfs = []

        def flush_batch():
            nonlocal batch_dfs, part_num, total_rows
            if not batch_dfs:
                return
            batch_full = pd.concat(batch_dfs, ignore_index=True)
            part_file = out_path / f"part-{part_num:05d}.parquet"
            batch_full.to_parquet(part_file, index=False)
            total_rows += len(batch_full)
            print(f"    -> {part_file.name} yazildi ({len(batch_full)} satir, "
                  f"toplam {total_rows} satir)")
            part_num += 1
            batch_dfs = []
            del batch_full
            gc.collect()

        for i, m in enumerate(members):
            try:
                f = tar.extractfile(m)
                if f is None:
                    continue
                raw = f.read()
                df = parse_trace_bytes(raw)
                if len(df):
                    batch_dfs.append(df)
                    seen_aircraft.update(df["source_id"].dropna().unique().tolist())
                total_files_ok += 1
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  HATA {m.name}: {e}")

            if (i + 1) % batch_size == 0:
                print(f"  {i + 1}/{len(members)} dosya islendi, batch yaziliyor...")
                flush_batch()

        flush_batch()  # kalan son parca

    print(f"\nTamamlandi.")
    print(f"  Cikti klasoru: {out_path}  ({part_num} parca dosya)")
    print(f"  Toplam satir: {total_rows}")
    print(f"  Benzersiz ucak (icao24): {len(seen_aircraft)}")
    print(f"  Islenen dosya: {total_files_ok}  |  Hata: {errors}")
    print(f"\nOkumak icin:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_parquet(r'{out_path}')   # tum parcalari otomatik birlestirir")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python parse_adsb_traces_from_tar_v2.py <tar_yolu> [cikti_klasoru] [batch_size] [limit]")
        sys.exit(1)
    tar_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "silver/adsb_traces"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    main(tar_path, out_dir, batch_size, limit)
