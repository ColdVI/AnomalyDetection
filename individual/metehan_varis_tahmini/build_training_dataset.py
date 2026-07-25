"""adsb.lol arsivinden GERCEK egitim ornekleri (X, y) uretir -- v2.

v1'e gore FARKLAR (kullanici: "bastan yap, eksik olmasin"):
  1. Gold yerine SILVER'dan okunuyor -- Gold'un 7+3+1 semasi flight_callsign
     TASIMIYOR (COLUMN_MAPS'te yok), adsbdb icin callsign sart. Silver zaten
     ayni konum verisini + callsign'i birlikte tasiyor, ek bir sey kaybetmiyoruz.
  2. Bolge testi artik KUTU degil, GERCEK kita/ulke siniri (region_filter.py,
     Natural Earth + shapely) -- v1'deki EUROPE_BBOX/ASIA_BBOX arasinda Dogu
     Turkiye/Iran/Irak/Suriye/Suudi Arabistan'i kapsayan lon 40-60 bosluğu
     vardi, gercek ucuslar yanlislikla eleniyordu.
  3. Heading artik TUREMIYOR -- Silver'in gercek `track_deg` kolonu kullaniliyor.
  4. Heuristik (en-yakin-nokta, 30km) BASARISIZ olan session'lar icin adsbdb
     callsign-lookup FALLBACK'i eklendi (bkz. adsbdb_client.py) -- GPS'e
     bagimli olmadigi icin heuristigin kaybettigi session'lari kurtarabiliyor.
     adsbdb'nin bulduğu varis da Avrupa+Asya disindaysa (ör. transatlantik)
     yine ELENIYOR -- kapsam karari degismedi, sadece etiketleme yontemi
     zenginlesti.

Bellek guvenligi: onceki gibi parca-parca (Silver'in ~400K satirlik 6852
parcasi), sonuc hemen diske yaziliyor, ham veri belekte birikmiyor.
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"c:\Users\PC_6276\Desktop\github\AnomalyDetection")
import time
from math import radians as m_radians
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.minio_io import get_minio_client, read_parquet_object
from team_dashboard.layer_index import load_index
from individual.metehan_varis_tahmini.region_filter import in_europe_or_asia
from individual.metehan_varis_tahmini import adsbdb_client

OUT_DIR = Path(__file__).parent / "training_sessions"
DONE_PARTS_PATH = Path(__file__).parent / "training_sessions_done_parts.txt"
STATS_PATH = Path(__file__).parent / "training_sessions_stats.json"

TRUNCATE_MINUTES = [5, 10, 15, 30, None]  # None = tam iz (100%)
MIN_POINTS_FOR_TRUNCATION = 3

AIRPORTS_CSV = Path(r"c:\Users\PC_6276\Desktop\github\AnomalyDetection\individual\metehan_geo_country\data\airports.csv")
_airports = pd.read_csv(AIRPORTS_CSV, low_memory=False)
_real_airports = _airports[(_airports["type"].isin(["large_airport", "medium_airport"])) & (_airports["scheduled_service"] == "yes")]
_real_airports = _real_airports[["ident", "latitude_deg", "longitude_deg"]].reset_index(drop=True)
_lat_r = np.radians(_real_airports["latitude_deg"].values)
_lon_r = np.radians(_real_airports["longitude_deg"].values)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_airport(lat, lon):
    lat1, lon1 = m_radians(lat), m_radians(lon)
    dlat = _lat_r - lat1
    dlon = _lon_r - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(_lat_r) * np.sin(dlon / 2) ** 2
    d = 2 * 6371 * np.arcsin(np.sqrt(a))
    i = np.argmin(d)
    return _real_airports.iloc[i]["ident"], d[i]


def _representative_callsign(callsigns: pd.Series) -> str | None:
    cs_series = callsigns.dropna()
    if cs_series.empty:
        return None
    callsign = cs_series.mode().iat[0].strip()
    return callsign or None


def _route_from_adsbdb(route: dict | None):
    """adsbdb sonucunu (origin_icao, dest_icao) ciftine cevirir -- farkli
    havalimani VE Avrupa/Asya icinde kalma sartlarini kontrol eder."""
    if route is None:
        return None
    if route["origin_icao"] == route["dest_icao"]:
        return None
    o_lat, o_lon = route["origin_lat"], route["origin_lon"]
    d_lat, d_lon = route["dest_lat"], route["dest_lon"]
    if o_lat is None or d_lat is None:
        return None
    in_zone = in_europe_or_asia(np.array([o_lat, d_lat]), np.array([o_lon, d_lon]))
    if not in_zone.all():
        return None
    return route["origin_icao"], route["dest_icao"]


def _build_truncated_examples(object_name, sid, g, first, o_id, d_id, source):
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
            "session_id": f"{object_name}::{sid}",
            "label_source": source,
            "origin_airport": o_id,
            "dest_airport": d_id,
            "truncate_min": "full" if trunc is None else str(trunc),
            "n_points": len(sub),
            "lats": sub["lat"].to_numpy(dtype=np.float32).tolist(),
            "lons": sub["lon"].to_numpy(dtype=np.float32).tolist(),
            "alts": sub["alt"].fillna(0).to_numpy(dtype=np.float32).tolist(),
            "velocities": sub["ground_speed_ms"].fillna(0).to_numpy(dtype=np.float32).tolist(),
            "headings": sub["track_deg"].fillna(0).to_numpy(dtype=np.float32).tolist(),
            "vrates": sub["vertical_rate_ms"].fillna(0).to_numpy(dtype=np.float32).tolist(),
            "dt_norms": dt_norm.tolist(),
        })
    return out


def extract_examples_from_part(object_name: str, client) -> tuple[list[dict], dict]:
    cols = ["source_id", "timestamp_utc", "lat", "lon", "alt", "ground_speed_ms",
            "track_deg", "vertical_rate_ms", "flight_callsign"]
    stats = {"sessions_total": 0, "zone_safe": 0, "heuristic_matched": 0, "adsbdb_recovered": 0, "discarded": 0}

    d = read_parquet_object(client, "silver", object_name, columns=cols)
    d = d.dropna(subset=["source_id", "timestamp_utc", "lat", "lon"])
    if d.empty:
        return [], stats
    d["ts"] = pd.to_datetime(d["timestamp_utc"], unit="s", utc=True)
    d = d.sort_values(["source_id", "ts"]).reset_index(drop=True)

    gap_min = d.groupby("source_id")["ts"].diff().dt.total_seconds() / 60
    new_session = gap_min.isna() | (gap_min > 180)
    d["session_id"] = new_session.astype("int64").cumsum()
    d["in_zone"] = in_europe_or_asia(d["lat"].to_numpy(), d["lon"].to_numpy())

    zone_ok = d.groupby("session_id")["in_zone"].all()
    safe_ids = zone_ok[zone_ok].index
    stats["sessions_total"] = d["session_id"].nunique()
    stats["zone_safe"] = len(safe_ids)
    if len(safe_ids) == 0:
        return [], stats
    d = d[d["session_id"].isin(safe_ids)]

    examples = []
    pending = []  # heuristik basarisiz olan session'lar -- adsbdb'ye toplu sorulacak

    # --- GECIS 1: temel filtreler + heuristik (aglara hic gitmeden, hizli) ---
    for sid, g in d.groupby("session_id"):
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
        if o_d <= 30 and d_dist <= 30 and o_id != d_id:
            stats["heuristic_matched"] += 1
            examples.extend(_build_truncated_examples(object_name, sid, g, first, o_id, d_id, "heuristic"))
        else:
            callsign = _representative_callsign(g["flight_callsign"])
            pending.append((sid, g, first, callsign))

    # --- GECIS 2: heuristigin kacirdigi session'larin callsign'lerini TOPLU sorgula ---
    if pending:
        callsigns = [cs for *_, cs in pending if cs]
        routes = adsbdb_client.lookup_routes_bulk(callsigns)
        for sid, g, first, callsign in pending:
            route = routes.get(callsign) if callsign else None
            recovered = _route_from_adsbdb(route)
            if recovered is None:
                stats["discarded"] += 1
                continue
            o_id, d_id = recovered
            stats["adsbdb_recovered"] += 1
            examples.extend(_build_truncated_examples(object_name, sid, g, first, o_id, d_id, "adsbdb"))

    return examples, stats


def _load_done_parts() -> set[str]:
    if DONE_PARTS_PATH.exists():
        return set(DONE_PARTS_PATH.read_text().splitlines())
    return set()


def _mark_done(part: str) -> None:
    with open(DONE_PARTS_PATH, "a") as f:
        f.write(part + "\n")


def main(n_parts: int | None = None, shard_id: int = 0, num_shards: int = 1):
    idx = load_index()
    subset = idx[(idx["layer"] == "silver") & (idx["dataset"] == "adsblol_historical")].copy()
    all_parts = sorted(subset["object_name"].tolist())
    if n_parts is not None:
        all_parts = all_parts[:n_parts]

    done = _load_done_parts()
    remaining_all = [p for p in all_parts if p not in done]
    # paralel shard'lar arasinda AYRIK (round-robin) bolusturme -- her shard
    # sadece kendi payini isler, ayni parcayi iki kez islemezler.
    remaining = remaining_all[shard_id::num_shards] if num_shards > 1 else remaining_all
    print(f"Toplam parca: {len(all_parts)}, zaten bitmis: {len(done)}, kalan (tumu): {len(remaining_all)}, "
          f"bu shard'a dusen: {len(remaining)} (shard {shard_id}/{num_shards})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = get_minio_client()
    start = time.time()
    total_examples = 0
    agg = {"sessions_total": 0, "zone_safe": 0, "heuristic_matched": 0, "adsbdb_recovered": 0, "discarded": 0}

    for i, part in enumerate(remaining):
        examples, stats = extract_examples_from_part(part, client)
        for k in agg:
            agg[k] += stats[k]
        if examples:
            df = pd.DataFrame(examples)
            safe_name = part.replace("/", "_").replace(".parquet", "")
            df.to_parquet(OUT_DIR / f"{safe_name}.parquet", index=False)
            total_examples += len(examples)
        _mark_done(part)

        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            adsbdb_client.save_cache()

        if (i + 1) % 20 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta_min = (len(remaining) - (i + 1)) / rate / 60 if rate > 0 else float("nan")
            print(f"[shard {shard_id}] [{i + 1}/{len(remaining)}] ornek={total_examples} "
                  f"heuristic={agg['heuristic_matched']} adsbdb={agg['adsbdb_recovered']} "
                  f"discarded={agg['discarded']} | {rate:.3f} parca/sn, ETA {eta_min:.1f} dk", flush=True)

    adsbdb_client.save_cache()
    print(f"\n[shard {shard_id}] Bitti. {total_examples} yeni ornek. Session istatistigi: {agg}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-parts", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    main(n_parts=args.n_parts, shard_id=args.shard_id, num_shards=args.num_shards)
