"""step3_build_country_routes.py -- Adim 3, ulke bazli en sik rotalar.

Ham veride (aircraft_dump_*.csv) rota/havalimani bilgisi YOK -- sadece ham
lat/lon konum kayitlari (bkz. step2 docstring). Bu script:
  1. Her ucak (hex) icin ham izleri ZAMAN BOSLUGU kuraliyla (>45dk) tekil
     UCUS BACAKLARINA ayirir (docs/route_deviation_prompt.md'deki -- artik
     repo'da olmayan -- route_clustering.py'nin ayni yontemi, sifirdan
     yeniden yazildi, script kaybolmustu).
  2. Her bacagin ILK ve SON noktasini OurAirports (acik veri seti, ilk
     calistirmada indirilip data/airports.csv'ye onbelleklenir) ile ~15km
     icindeki en yakin havalimanina esler (snap) -- eslesmeyen bacaklar
     (havalimanindan uzak baslama/bitis, orn. hala havadayken kesilen trace)
     ATILIR.
  3. Bacagin UCAGININ (step2'deki gibi hex_country.py TESCIL ulkesi --
     dominant_country DEGIL) ulkesini atar -- "bu ulkeye tescilli ucaklarin
     EN SIK gittigi rotalar" anlaminda.
  4. Ulke+pair (orn. "LTBA->LTAE") basina BENZERSIZ ucus bacagi sayar, ULKE
     ICINDE rank eder (rank==1 = en sik rota, "is_top": true).
  5. GERCEK KORIDOR uretir (route_corridors.geojson): o pair'e ait TUM ucus
     bacaklarinin gectigi HAM noktalari H3 res5 hucrelere ayirip, hucre
     basina "ratio" (o pair'in ucuslarinin ne kadari o hucreden gecti)
     hesaplar -- polygon olarak TUM gecilen hucreler yazilir (sert esik
     YOK, bkz. build_corridor_geojson notu), ratio frontend'de opaklik
     gradyani icin kullanilir. 2026-07-09: kullanici DUZ CIZGI (origin/dest
     havalimani arasi LineString) secenegini degerlendirdikten sonra
     KALDIRILMASINI istedi -- ucagin GERCEKTE izledigi yolu degil sadece
     baslangic-bitis noktasini gosterdigi icin yaniltici bulundu, artik
     SADECE koridor uretiliyor.

Kullanim:
    python -m individual.metehan_geo_country.step3_build_country_routes
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from individual.metehan_geo.geo import assign_h3_cell, h3_cell_to_polygon
from individual.metehan_geo_country.step2_build_country_layers import (
    DATA_DIR,
    VIZ_DATA_DIR,
    load_enriched_aircraft_df,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AIRPORTS_CSV = DATA_DIR / "airports.csv"
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

FLIGHT_GAP_MINUTES = 45          # ayni ucaktan >45dk sinyal gelmezse yeni bacak basliyor
AIRPORT_SNAP_KM = 15             # baslangic/bitis bu yaricap disindaysa bacak atilir
MIN_LEG_POINTS = 3               # gurultu (tek-iki nokta) bacaklari eleme
H3_RESOLUTION = 5                # step2/main proje ile tutarli
MIN_FLIGHTS_PER_PAIR = 2         # tek ucusluk "rota" gurultu sayilir, atilir
TOP_N_PER_COUNTRY = 15           # dosya boyutu/render yukunu sinirlamak icin

# 2026-07-09 (kullanici geri bildirimi -- ilk versiyon "bu ya ne biçim rota,
# çizgiler ufacık" diye reddedildi): bacaklarin %67'si (534'un 360'i) 50km'nin
# ALTINDAYDI -- bu dataset egitim/askeri ucuslardan olustugu icin COK sayida
# GERCEK ama YEREL egitim rotasi (touch-and-go, yerel sahada tur) var, bunlar
# "en sik rota" sirlamasini domine edip anlamli/uzun rotalari gizliyordu.
MIN_ROUTE_DISTANCE_KM = 50


def _ensure_airports_csv() -> Path:
    if AIRPORTS_CSV.exists():
        return AIRPORTS_CSV
    logger.info("OurAirports veri seti indiriliyor: %s", AIRPORTS_URL)
    resp = requests.get(AIRPORTS_URL, timeout=60)
    resp.raise_for_status()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AIRPORTS_CSV.write_bytes(resp.content)
    logger.info("Yazildi: %s (%d bayt)", AIRPORTS_CSV, len(resp.content))
    return AIRPORTS_CSV


def load_airports() -> pd.DataFrame:
    path = _ensure_airports_csv()
    df = pd.read_csv(path, low_memory=False)
    df = df[df["type"].isin(["large_airport", "medium_airport", "small_airport"])]
    df = df.dropna(subset=["latitude_deg", "longitude_deg", "ident"])
    df = df[["ident", "name", "latitude_deg", "longitude_deg", "iso_country"]].reset_index(drop=True)
    logger.info("Havalimani referansi: %d havalimani (large/medium/small)", len(df))
    return df


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def snap_to_airport(lats: np.ndarray, lons: np.ndarray, airports: pd.DataFrame, max_km: float) -> list[str | None]:
    """Her (lat,lon) icin en yakin havalimanini bulur (brute-force haversine,
    her nokta icin TUM havalimanlarina karsi vektorlestirilmis -- binlerce
    bacak ucbirimi x ~9000 havalimani icin KDTree'ye gerek kalmadan yeterince
    hizli)."""
    air_lat = airports["latitude_deg"].to_numpy()
    air_lon = airports["longitude_deg"].to_numpy()
    idents = airports["ident"].to_numpy()
    out: list[str | None] = []
    for lat, lon in zip(lats, lons):
        d = _haversine_km(lat, lon, air_lat, air_lon)
        i = np.argmin(d)
        out.append(idents[i] if d[i] <= max_km else None)
    return out


def build_flight_legs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Her ucak (hex) icin ham izleri zaman-boslugu kuraliyla ucus bacaklarina
    ayirir. Doner: (bacak-basi ozet DataFrame, flight_id kolonlu ham nokta
    DataFrame -- ikincisi koridor hesaplamasinda lazim)."""
    df = df.sort_values(["hex", "seen_at"]).reset_index(drop=True)
    gap = df.groupby("hex")["seen_at"].diff()
    new_leg = gap.isna() | (gap > pd.Timedelta(minutes=FLIGHT_GAP_MINUTES))
    leg_num = new_leg.groupby(df["hex"]).cumsum()
    df = df.assign(leg_num=leg_num, flight_id=df["hex"] + "_" + leg_num.astype(str))

    legs = df.groupby("flight_id").agg(
        hex=("hex", "first"),
        n_points=("seen_at", "size"),
        start_ts=("seen_at", "first"),
        end_ts=("seen_at", "last"),
        start_lat=("latitude", "first"),
        start_lon=("longitude", "first"),
        end_lat=("latitude", "last"),
        end_lon=("longitude", "last"),
        # 2026-07-10: is_military ucak-bazli SABIT bir ozellik (step2'nin
        # load_enriched_aircraft_df'i zaten her noktaya isliyor) -- "first"
        # yeterli, ayni ucagin tum noktalarinda ayni deger.
        is_military=("is_military", "first"),
    ).reset_index()
    logger.info("Ham bacak sayisi (filtre oncesi): %d", len(legs))
    legs = legs[legs["n_points"] >= MIN_LEG_POINTS]
    logger.info("MIN_LEG_POINTS=%d sonrasi: %d bacak", MIN_LEG_POINTS, len(legs))
    return legs, df


def enrich_legs_with_airports(legs: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    legs = legs.copy()
    legs["origin_icao"] = snap_to_airport(legs["start_lat"].to_numpy(), legs["start_lon"].to_numpy(), airports, AIRPORT_SNAP_KM)
    legs["dest_icao"] = snap_to_airport(legs["end_lat"].to_numpy(), legs["end_lon"].to_numpy(), airports, AIRPORT_SNAP_KM)
    before = len(legs)
    legs = legs.dropna(subset=["origin_icao", "dest_icao"])
    legs = legs[legs["origin_icao"] != legs["dest_icao"]]
    logger.info("Havalimani esleme (<=%dkm) sonrasi: %d/%d bacak", AIRPORT_SNAP_KM, len(legs), before)

    # 2026-07-09 (kullanici geri bildirimi): havalimani-arasi DUZ mesafe
    # MIN_ROUTE_DISTANCE_KM'nin altindaysa bacak atilir -- yerel egitim
    # turlarinin (touch-and-go vb.) "en sik rota" olarak on plana cikmasini
    # engeller (bkz. modul basi not).
    before_dist = len(legs)
    dist_km = _haversine_km(
        legs["start_lat"].to_numpy(), legs["start_lon"].to_numpy(),
        legs["end_lat"].to_numpy(), legs["end_lon"].to_numpy(),
    )
    legs = legs[dist_km >= MIN_ROUTE_DISTANCE_KM]
    logger.info("MIN_ROUTE_DISTANCE_KM=%dkm sonrasi: %d/%d bacak", MIN_ROUTE_DISTANCE_KM, len(legs), before_dist)

    legs["pair"] = legs["origin_icao"] + "->" + legs["dest_icao"]
    return legs


def rank_routes(legs: pd.DataFrame) -> pd.DataFrame:
    counts = legs.groupby(["country", "pair"]).agg(
        flight_count=("flight_id", "nunique"),
        origin_icao=("origin_icao", "first"),
        dest_icao=("dest_icao", "first"),
    ).reset_index()
    before = len(counts)
    counts = counts[counts["flight_count"] >= MIN_FLIGHTS_PER_PAIR]
    counts["rank"] = counts.groupby("country")["flight_count"].rank(method="first", ascending=False).astype(int)
    counts = counts[counts["rank"] <= TOP_N_PER_COUNTRY]
    counts["is_top"] = counts["rank"] == 1
    # 2026-07-09 (kullanici geri bildirimi): rank==1 HER ZAMAN bir "kazanan"
    # secer, alttaki ornek buyuklugune BAKMAKSIZIN -- az veri olan bir ulkede
    # (ör. Turkiye, 4 ucusla rank=1) bu, 4 ile 2 ucus arasindaki ISTATISTIKSEL
    # OLARAK ANLAMSIZ farki "guclu bir bulgu" gibi gosterebilir. Bireysel
    # projedeki (route_deviation, artik kaldirilmis) ayni AYNI esik (>=15
    # ucus) burada da "dusuk guven" bayragi olarak kullaniliyor -- rota
    # ELENMIYOR (kullanici kendi karar versin diye), sadece isaretleniyor.
    MIN_CONFIDENT_FLIGHTS = 15
    counts["low_confidence"] = counts["flight_count"] < MIN_CONFIDENT_FLIGHTS
    logger.info(
        "Rota siralama: %d (ulke,pair) -> MIN_FLIGHTS_PER_PAIR=%d ve TOP_N=%d sonrasi %d "
        "(bunlarin %d'i dusuk guven, <%d ucus)",
        before, MIN_FLIGHTS_PER_PAIR, TOP_N_PER_COUNTRY, len(counts),
        int(counts["low_confidence"].sum()), MIN_CONFIDENT_FLIGHTS,
    )
    return counts.sort_values(["country", "rank"]).reset_index(drop=True)


def build_corridor_geojson(points_df: pd.DataFrame, legs: pd.DataFrame, ranked: pd.DataFrame) -> dict:
    """points_df: build_flight_legs()'in dondurdugu, HER ham noktayi flight_id
    ile tasiyan DataFrame. legs: enrich_legs_with_airports() sonrasi (country
    atanmis) bacak-basi ozet. ranked: rank_routes() ciktisi (topN + min-flights
    filtresini gecmis (ulke,pair) listesi)."""
    kept = legs.merge(ranked[["country", "pair"]], on=["country", "pair"], how="inner")
    kept_ids = set(kept["flight_id"])
    pts = points_df[points_df["flight_id"].isin(kept_ids)].copy()
    # ONEMLI: points_df (load_enriched_aircraft_df'ten miras) zaten "country"
    # tasiyor -- burada SADECE "pair"i getiriyoruz, aksi halde merge iki ayri
    # "country" kolonunu country_x/country_y'ye cevirip asagidaki groupby'i
    # KeyError'a dusuruyordu (ilk calistirmada yakalanan hata).
    pts = pts.merge(kept[["flight_id", "pair"]], on="flight_id", how="left")
    pts = pts.rename(columns={"latitude": "lat", "longitude": "lon"})
    pts = assign_h3_cell(pts, H3_RESOLUTION)

    # 2026-07-09 (kullanici geri bildirimi: "koridor tam degil, garip"):
    # ilk versiyon ratio>=0.5 hucreleri SERT filtreliyordu -- coz umlu ucus
    # sayisi arttikca (GPS/rota sapmasi nedeniyle) hicbir hucre %50 esigini
    # gecemeyebiliyordu, bu da SEYREK/KOPUK noktalar halinde gorunen bir
    # "koridor" uretiyordu (medyan sadece 3 hucre/rota). SIMDI: o pair'in
    # ucuslarinin GECTIGI TUM hucreler tutuluyor (filtre YOK) -- surekli,
    # bosluksuz bir yol gorunumu. "ratio" yine de hesaplanip saklaniyor,
    # frontend'de opaklik/renk yogunlugu icin kullanilabilir (dusuk ratio =
    # sadece birkac ucagin saptigi kenar, yuksek ratio = gercek "cekirdek").
    per_hex = pts.groupby(["country", "pair", "h3_cell"])["flight_id"].nunique().reset_index(name="flight_count_hex")
    totals = ranked.set_index(["country", "pair"])["flight_count"]
    per_hex["total_flights"] = [totals.get((c, p), np.nan) for c, p in zip(per_hex["country"], per_hex["pair"])]
    per_hex["ratio"] = per_hex["flight_count_hex"] / per_hex["total_flights"]
    logger.info("Koridor: %d (ulke,pair,hex) toplam hucre (filtresiz, tam yol)", len(per_hex))

    rank_lookup = ranked.set_index(["country", "pair"])[["rank", "is_top", "flight_count", "low_confidence"]]
    features = []
    for row in per_hex.itertuples(index=False):
        ring = h3_cell_to_polygon(row.h3_cell)
        if ring is None:
            continue
        meta = rank_lookup.loc[(row.country, row.pair)]
        features.append({
            "type": "Feature",
            "properties": {
                "country": row.country, "pair": row.pair, "h3_cell": row.h3_cell,
                "ratio": round(float(row.ratio), 3),
                "flight_count": int(meta["flight_count"]), "rank": int(meta["rank"]), "is_top": bool(meta["is_top"]),
                "low_confidence": bool(meta["low_confidence"]),
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    df = load_enriched_aircraft_df()
    df = df.dropna(subset=["country"])
    logger.info("Ulke atanmis ham nokta: %d", len(df))

    airports = load_airports()
    legs, points_df = build_flight_legs(df)

    hex_to_country = df.drop_duplicates("hex").set_index("hex")["country"]
    legs["country"] = legs["hex"].map(hex_to_country)
    legs = legs.dropna(subset=["country"])

    legs = enrich_legs_with_airports(legs, airports)

    VIZ_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 2026-07-10 (kullanici istegi): "Tumu" (mevcut davranis, dosya adlari
    # DEGISMEDI -- geriye donuk uyumlu) + Sivil-only + Askeri-only rota
    # siralama/koridor. Ucak zaten is_military ile flaglenmis (bkz.
    # step2.load_enriched_aircraft_df -- ana projenin lookup'undan JOIN
    # edildi). Askeri ucus sayisi cok daha az olabilir (kullanici: "ek
    # projede asgari uçak olmayabilir o kalabilir") -- bu durumda ilgili
    # ulkeler o varyantta BASITCE gorunmez, hata degil.
    variants = {
        "": legs,
        "_civil": legs[~legs["is_military"]],
        "_military": legs[legs["is_military"]],
    }
    for suffix, legs_subset in variants.items():
        label = suffix.lstrip("_") or "all"
        if legs_subset.empty:
            logger.warning("Varyant '%s': hic bacak yok, bos dosya yazilacak", label)
            ranked = pd.DataFrame(columns=[
                "country", "pair", "flight_count", "origin_icao", "dest_icao",
                "rank", "is_top", "low_confidence",
            ])
            corridors = {"type": "FeatureCollection", "features": []}
        else:
            ranked = rank_routes(legs_subset)
            corridors = build_corridor_geojson(points_df, legs_subset, ranked)

        corridors_path = VIZ_DATA_DIR / f"route_corridors{suffix}.geojson"
        with corridors_path.open("w", encoding="utf-8") as f:
            json.dump(corridors, f)
        logger.info("Yazildi (%s): %s (%d hucre)", label, corridors_path, len(corridors["features"]))

        routes_csv_path = DATA_DIR / f"country_routes{suffix}.csv"
        ranked.to_csv(routes_csv_path, index=False)
        logger.info("Yazildi (%s): %s (%d ulke+rota)", label, routes_csv_path, len(ranked))


if __name__ == "__main__":
    main()
