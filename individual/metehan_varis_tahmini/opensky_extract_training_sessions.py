"""OpenSky arsivinden, nadir-rota kesisim testinde (opensky_rare_route_check.py,
bkz. opensky_rare_route_check_results.json) eslesen ~1.644 rotanin GERCEK
trajectory noktalarini cikarip training_sessions/ formatina cevirir --
session_split.parquet'e SADECE train split'i olarak eklenir (val/test HER
ZAMAN adsb.lol'un kendi orijinal split'inde kalir, bu zenginlestirmeyle
degismez -- yoksa test seti "kirlenirdi").

opensky_comparison_test.py/opensky_rare_route_check.py ile AYNI metodoloji
(session segmentasyonu + gercek bolge testi + en-yakin-havalimani heuristigi),
farkli olarak:
  1. SADECE eslesen nadir rotalar (target_routes) tutuluyor -- geri kalani
     (adsb.lol'da zaten sik/hic olmayan rotalar) atiliyor, bellek/disk tasarrufu.
  2. Nokta-seviyesi veri (lat/lon/alt/hiz/yon/vrate) da cikariliyor (onceki
     script sadece SAYIYORDU, noktalari atiyordu).
  3. build_training_dataset.py'nin ayni kesme-noktasi (5/10/15/30/full dk)
     augmentasyonu uygulaniyor -- egitim/serve tutarliligi icin ayni desen.

RAM dersi (2026-07-28/29, bkz. gunluk): bu makine paralel gun islemeyi
GUVENLE KALDIRAMIYOR (4/3/2 worker'da bile neredeyse OOM) -- bu yuzden bu
script BILINCLI olarak SERI (tek gun tek seferde, ProcessPoolExecutor YOK).
"""
from __future__ import annotations

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
from individual.metehan_varis_tahmini.opensky_comparison_test import STATES_DIR
from individual.metehan_varis_tahmini.opensky_rare_route_check import load_rare_routes, OUT_PATH as CHECK_RESULTS_PATH
from individual.metehan_varis_tahmini.train_full import SESSIONS_DIR, SPLIT_PATH

OUT_DIR = SESSIONS_DIR  # dogrudan training_sessions/ icine -- load_all_sessions() otomatik bulur
DONE_DAYS_PATH = Path(__file__).parent / "opensky_extract_done_days.txt"

TRUNCATE_MINUTES = [5, 10, 15, 30, None]  # build_training_dataset.py ile AYNI -- egitim/serve tutarliligi
MIN_POINTS_FOR_TRUNCATION = 3


def target_routes_from_check_results() -> set[str]:
    """opensky_rare_route_check_results.json'daki (tum-arsiv taramasi zaten
    tamamlanmis) route sayimlarini nadir-rota kumesiyle kesistirip SADECE
    eslesen rota anahtarlarini (origin||dest) dondurur -- bu script o
    kesisimdeki rotalarin GERCEK noktalarini cikaracak."""
    rare_routes = load_rare_routes()
    results = json.loads(CHECK_RESULTS_PATH.read_text())
    all_routes: dict[str, int] = {}
    for day, r in results["days"].items():
        for k, v in r["routes"].items():
            all_routes[k] = all_routes.get(k, 0) + v
    matched = {k for k in all_routes if k in rare_routes}
    return matched


def read_day_points_full(day: str) -> pd.DataFrame:
    """opensky_comparison_test.read_day_points ile AYNI ama TUM ucus alanlarini
    (alt/hiz/yon/vrate) da okur -- egitim ornegi icin gerekli.

    BELLEK DUZELTMESI (2026-07-29): ilk versiyon TUM gunu (24 saat, ~50-60M
    satir) TEK Python listesinde biriktirip en sonda DataFrame'e ceviriyordu
    -- bu, 9,4M satirda bile 9,2GB RAM'e cikip makineyi cokertme sinirina
    getirdi (Python float/obj listesi cok pahali, numpy'nin ~8 kati).
    Duzeltme: HER SAAT kendi kucuk listesini (~2-2,5M satir) numpy'ye
    cevirip DataFrame'e ekliyor, bellekte asla TUM gunun ham Python listesi
    birikmez -- ayrica float64 yerine float32 (lat/lon haric) kullanilarak
    ek bir kazanc daha saglaniyor."""
    hour_frames = []
    total = 0
    for hh in range(24):
        hh_s = f"{hh:02d}"
        tar_path = STATES_DIR / day / hh_s / f"states_{day}-{hh_s}.avro.tar"
        if not tar_path.exists():
            print(f"  [{day}] saat {hh_s} eksik, atlaniyor", flush=True)
            continue
        icaos, lats, lons, times, alts, vels, heads, vrates = [], [], [], [], [], [], [], []
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
                    alts.append(rec["baroaltitude"] if rec["baroaltitude"] is not None else 0.0)
                    vels.append(rec["velocity"] if rec["velocity"] is not None else 0.0)
                    heads.append(rec["heading"] if rec["heading"] is not None else 0.0)
                    vrates.append(rec["vertrate"] if rec["vertrate"] is not None else 0.0)
        hour_df = pd.DataFrame({
            "icao24": pd.Categorical(icaos),
            "lat": np.asarray(lats, dtype=np.float32),
            "lon": np.asarray(lons, dtype=np.float32),
            "time": np.asarray(times, dtype=np.int64),
            "alt": np.asarray(alts, dtype=np.float32),
            "vel": np.asarray(vels, dtype=np.float32),
            "head": np.asarray(heads, dtype=np.float32),
            "vrate": np.asarray(vrates, dtype=np.float32),
        })
        hour_frames.append(hour_df)
        total += len(hour_df)
        print(f"  [{day}] saat {hh_s} okundu, birikmis satir: {total}", flush=True)

    df = pd.concat(hour_frames, ignore_index=True)
    return df


def _build_truncated_examples(day: str, sid: int, g: pd.DataFrame, o_id: str, d_id: str) -> list[dict]:
    first = g.iloc[0]
    elapsed_min = (g["ts"] - first["ts"]).dt.total_seconds() / 60
    total_min = elapsed_min.iloc[-1]
    out = []
    for trunc in TRUNCATE_MINUTES:
        if trunc is None:
            sub = g
        else:
            if total_min < trunc:
                continue
            sub = g[elapsed_min <= trunc]
        if len(sub) < MIN_POINTS_FOR_TRUNCATION:
            continue
        sub_elapsed = elapsed_min[sub.index]
        dt_norm = (sub_elapsed / max(sub_elapsed.max(), 1e-6)).to_numpy(dtype=np.float32)
        out.append({
            "session_id": f"opensky_2022::{day}::{sid}",
            "label_source": "opensky_heuristic",
            "origin_airport": o_id,
            "dest_airport": d_id,
            "truncate_min": "full" if trunc is None else str(trunc),
            "n_points": len(sub),
            "lats": sub["lat"].to_numpy(dtype=np.float32).tolist(),
            "lons": sub["lon"].to_numpy(dtype=np.float32).tolist(),
            "alts": sub["alt"].to_numpy(dtype=np.float32).tolist(),
            "velocities": sub["vel"].to_numpy(dtype=np.float32).tolist(),
            "headings": sub["head"].to_numpy(dtype=np.float32).tolist(),
            "vrates": sub["vrate"].to_numpy(dtype=np.float32).tolist(),
            "dt_norms": dt_norm.tolist(),
        })
    return out


def process_day_extract(day: str, target_routes: set[str]) -> list[dict]:
    t0 = time.time()
    df = read_day_points_full(day)
    if df.empty:
        return []

    df = df.sort_values(["icao24", "time"], kind="mergesort").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
    gap_min = df.groupby("icao24", observed=True)["ts"].diff().dt.total_seconds() / 60
    new_session = gap_min.isna() | (gap_min > 180)
    df["session_id"] = new_session.astype("int64").cumsum()
    df["in_zone"] = in_europe_or_asia(df["lat"].to_numpy(), df["lon"].to_numpy())

    zone_ok = df.groupby("session_id")["in_zone"].all()
    safe_ids = set(zone_ok[zone_ok].index)
    d_safe = df[df["session_id"].isin(safe_ids)]

    examples = []
    matched_sessions = 0
    for sid, g in d_safe.groupby("session_id"):
        if len(g) < MIN_POINTS_FOR_TRUNCATION:
            continue
        g = g.reset_index(drop=True)
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
        if o_d > 30 or d_dist > 30 or o_id == d_id:
            continue
        route_key = f"{o_id}||{d_id}"
        if route_key not in target_routes:
            continue
        matched_sessions += 1
        examples.extend(_build_truncated_examples(day, sid, g, o_id, d_id))

    print(f"[{day}] bitti ({time.time()-t0:.0f}sn): {matched_sessions} hedef-rota session'i, "
          f"{len(examples)} ornek (kesme-noktalari dahil)", flush=True)
    return examples


def main(n_workers: int = 1):
    target_routes = target_routes_from_check_results()
    print(f"Hedef (eslesen nadir) rota sayisi: {len(target_routes)}", flush=True)

    days = sorted(p.name for p in STATES_DIR.iterdir() if p.is_dir())
    done = set(DONE_DAYS_PATH.read_text().splitlines()) if DONE_DAYS_PATH.exists() else set()
    todo = [d for d in days if d not in done]
    mode = "SERI (1 gun/seferde)" if n_workers == 1 else f"{n_workers} PARALEL worker"
    print(f"Islenecek gun: {len(todo)} (zaten islenmis: {len(done)}/{len(days)}) -- {mode}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_new_session_ids: set[str] = set()
    total_examples = 0

    def _handle_result(day, examples):
        nonlocal total_examples
        if examples:
            df = pd.DataFrame(examples)
            df.to_parquet(OUT_DIR / f"opensky_enrich_{day}.parquet", index=False)
            total_examples += len(examples)
            all_new_session_ids.update(df["session_id"].unique().tolist())
        with open(DONE_DAYS_PATH, "a") as f:
            f.write(day + "\n")

    if n_workers == 1:
        for day in todo:
            examples = process_day_extract(day, target_routes)
            _handle_result(day, examples)
    else:
        # Bellek duzeltmesinden (read_day_points_full, saat-basi numpy'ye
        # cevirme) sonra guvenli -- gun basina tepe RAM ~2-3GB'a dustu.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(process_day_extract, day, target_routes): day for day in todo}
            for fut in as_completed(futures):
                day = futures[fut]
                _handle_result(day, fut.result())
                print(f"[{day}] kaydedildi", flush=True)

    print(f"\nToplam yeni ornek (kesme-noktalari dahil): {total_examples}, "
          f"benzersiz yeni session: {len(all_new_session_ids)}", flush=True)

    if all_new_session_ids:
        split_df = pd.read_parquet(SPLIT_PATH)
        existing = set(split_df["session_id"])
        # origin/dest'i taze uretilen parquet'lerden topla (session basina 1 satir yeter)
        new_rows = []
        seen = set()
        for day in todo:
            fp = OUT_DIR / f"opensky_enrich_{day}.parquet"
            if not fp.exists():
                continue
            df = pd.read_parquet(fp, columns=["session_id", "origin_airport", "dest_airport"])
            for sid, o, d in df.drop_duplicates("session_id").itertuples(index=False):
                if sid in existing or sid in seen:
                    continue
                seen.add(sid)
                new_rows.append({"session_id": sid, "origin_airport": o, "dest_airport": d, "split": "train"})
        if new_rows:
            split_df = pd.concat([split_df, pd.DataFrame(new_rows)], ignore_index=True)
            split_df.to_parquet(SPLIT_PATH, index=False)
            print(f"session_split.parquet guncellendi: +{len(new_rows)} yeni TRAIN session'i "
                  f"(toplam {len(split_df)})", flush=True)
    print("\nBITTI.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    main(n_workers=args.workers)
