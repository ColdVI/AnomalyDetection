"""OpenSky 2022 (Ocak-Haziran, sadece Pazartesi) verisinde AYNI rota-tekrari +
okyanus-kutu testini calistirip adsb.lol Gold ile kiyaslamak icin.

Amac uretim egitim verisi degil, dogrulama/kiyas -- bu yuzden TUM 25 hafta
degil, araliga yayilmis bir ORNEKLEM (asagida SAMPLE_DAYS) isleniyor. Ayni
zamanda arka planda suren GAT egitimini yavaslatmamak icin bu script'in
process onceligi DUSUK (BelowNormal) baslatilmali (bkz. calistirma komutu).

Kaynak veri: C:/Users/PC_6276/Desktop/opensky verileri/states/{gun}/{saat}/
states_{gun}-{saat}.avro.tar -- her .tar icinde tek bir .avro (OpenSky State
Vector semasi: icao24, lat, lon, time, velocity, ...).

Metodoloji, mevcut adsb.lol pipeline'iyla (build_training_dataset.py) AYNI:
  - Session segmentasyonu: ayni ucak (icao24), zaman sirali, >180dk bosluk =
    yeni session.
  - Gecerlilik filtreleri: >=3 nokta, sure>=0,25 saat, hiz<=1200km/h,
    mesafe>=50km.
  - Bolge testi: region_filter.in_europe_or_asia (GERCEK kita siniri, kutu
    degil) -- session'in TUM noktalari bolge icinde kalmali.
  - Havalimani eslesmesi: build_training_dataset.nearest_airport (en yakin
    gercek havalimani, <=30km, origin!=dest).
  - adsbdb FALLBACK KULLANILMIYOR (bu bir dogrulama testi, tam egitim verisi
    degil) -- sadece heuristik eslesme sayiliyor, adil kiyas icin yeterli.

Okyanus-kutu kontrolu TUM noktalar uzerinde (bolge/session filtresinden
ONCE) yapiliyor -- adsb.lol'daki "okyanus-kapsama testi"yle ayni mantik.
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"c:\Users\PC_6276\Desktop\github\AnomalyDetection")

import json
import tarfile
import time
from collections import Counter
from pathlib import Path

import fastavro
import numpy as np
import pandas as pd

from individual.metehan_varis_tahmini.region_filter import in_europe_or_asia
from individual.metehan_varis_tahmini.build_training_dataset import nearest_airport, haversine_km

STATES_DIR = Path(r"C:\Users\PC_6276\Desktop\opensky verileri\states")
OUT_PATH = Path(__file__).parent / "opensky_comparison_results.json"

# Aralaga yayilmis, TAM (24/24 saat) oldugu onceden dogrulanmis 4 gun --
# tum 25 haftayi degil bir ornegini islemek: uretim verisi degil dogrulama.
SAMPLE_DAYS = ["2022-01-03", "2022-03-07", "2022-05-02", "2022-06-27"]

# Okyanus kutulari (adsb.lol testindekiyle ayni ruhta -- kesin/resmi sinirlar
# degil, kaba orta-okyanus kontrolu; Hawaii gibi ada karasindan kacinilarak
# secildi).
OCEAN_BOXES = {
    "orta_kuzey_atlantik": (30, 55, -45, -20),   # lat_min, lat_max, lon_min, lon_max
    "guney_atlantik": (-40, -10, -40, -10),
    "orta_pasifik": (-10, 10, -170, -150),
}


def read_day_points(day: str) -> pd.DataFrame:
    icaos, lats, lons, times = [], [], [], []
    for hh in range(24):
        hh_s = f"{hh:02d}"
        tar_path = STATES_DIR / day / hh_s / f"states_{day}-{hh_s}.avro.tar"
        if not tar_path.exists():
            print(f"  [{day}] saat {hh_s} eksik, atlaniyor", flush=True)
            continue
        with tarfile.open(tar_path) as tf:
            member_name = next(n for n in tf.getnames() if n.endswith(".avro"))
            with tf.extractfile(member_name) as f:
                reader = fastavro.reader(f)
                for rec in reader:
                    lat, lon = rec["lat"], rec["lon"]
                    if lat is None or lon is None:
                        continue
                    icaos.append(rec["icao24"])
                    lats.append(lat)
                    lons.append(lon)
                    times.append(rec["time"])
        print(f"  [{day}] saat {hh_s} okundu, birikmis satir: {len(icaos)}", flush=True)

    df = pd.DataFrame({"icao24": icaos, "lat": lats, "lon": lons, "time": times})
    df["icao24"] = df["icao24"].astype("category")
    df["lat"] = df["lat"].astype("float64")
    df["lon"] = df["lon"].astype("float64")
    return df


def ocean_box_counts(df: pd.DataFrame) -> dict:
    out = {}
    for name, (lat_min, lat_max, lon_min, lon_max) in OCEAN_BOXES.items():
        mask = (df["lat"] >= lat_min) & (df["lat"] <= lat_max) & (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
        out[name] = int(mask.sum())
    return out


def process_day(day: str) -> dict:
    t0 = time.time()
    df = read_day_points(day)
    total_rows = len(df)
    ocean = ocean_box_counts(df)

    df = df.sort_values(["icao24", "time"], kind="mergesort").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
    gap_min = df.groupby("icao24", observed=True)["ts"].diff().dt.total_seconds() / 60
    new_session = gap_min.isna() | (gap_min > 180)
    df["session_id"] = new_session.astype("int64").cumsum()
    df["in_zone"] = in_europe_or_asia(df["lat"].to_numpy(), df["lon"].to_numpy())

    total_sessions = int(df["session_id"].nunique())
    zone_ok = df.groupby("session_id")["in_zone"].all()
    safe_ids = set(zone_ok[zone_ok].index)
    zone_safe = len(safe_ids)

    routes = Counter()
    heuristic_matched = 0
    d_safe = df[df["session_id"].isin(safe_ids)]
    for sid, g in d_safe.groupby("session_id"):
        if len(g) < 3:
            continue
        first, last = g.iloc[0], g.iloc[-1]
        dur_h = (last["ts"] - first["ts"]).total_seconds() / 3600
        if dur_h < 0.25:
            continue
        dist_km = haversine_km(first["lat"], first["lon"], last["lat"], last["lon"])
        speed = dist_km / dur_h if dur_h > 0 else 0
        if speed > 1200 or dist_km < 50:
            continue
        o_id, o_d = nearest_airport(first["lat"], first["lon"])
        d_id, d_dist = nearest_airport(last["lat"], last["lon"])
        if o_d <= 30 and d_dist <= 30 and o_id != d_id:
            heuristic_matched += 1
            routes[f"{o_id}||{d_id}"] += 1

    elapsed = time.time() - t0
    print(f"[{day}] bitti ({elapsed:.0f}sn): satir={total_rows}, session={total_sessions}, "
          f"zone_safe={zone_safe}, heuristic_matched={heuristic_matched}, okyanus={ocean}", flush=True)

    return {
        "day": day,
        "total_rows": total_rows,
        "total_sessions": total_sessions,
        "zone_safe_sessions": zone_safe,
        "heuristic_matched": heuristic_matched,
        "routes": dict(routes),
        "ocean_boxes": ocean,
        "elapsed_sec": elapsed,
    }


def _load_results() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {"days": {}}


def _save_results(results: dict) -> None:
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def main():
    results = _load_results()
    for day in SAMPLE_DAYS:
        if day in results["days"]:
            print(f"[{day}] zaten islenmis, atlaniyor", flush=True)
            continue
        day_result = process_day(day)
        results["days"][day] = day_result
        _save_results(results)

    # ozet
    all_routes = Counter()
    total_sessions = total_zone_safe = total_matched = 0
    for day, r in results["days"].items():
        total_sessions += r["total_sessions"]
        total_zone_safe += r["zone_safe_sessions"]
        total_matched += r["heuristic_matched"]
        for k, v in r["routes"].items():
            all_routes[k] += v

    n_routes = len(all_routes)
    single = sum(1 for v in all_routes.values() if v == 1)
    print("\n=== OZET ===", flush=True)
    print(f"Islenen gun sayisi: {len(results['days'])}", flush=True)
    print(f"Toplam session: {total_sessions}, bolge-guvenli: {total_zone_safe}, "
          f"heuristik-eslesen: {total_matched}", flush=True)
    print(f"Farkli rota: {n_routes}, tek-seferlik: {single} (%{100*single/n_routes:.1f})" if n_routes else "Rota yok", flush=True)

    results["summary"] = {
        "total_sessions": total_sessions,
        "total_zone_safe": total_zone_safe,
        "total_matched": total_matched,
        "n_routes": n_routes,
        "single_occurrence": single,
    }
    _save_results(results)


if __name__ == "__main__":
    main()
