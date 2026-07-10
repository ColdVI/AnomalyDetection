"""build_traffic_clusters_realtime.py -- dbscan_geo_hotspot_prompt.md, 7d
realtime icin genisletme (2026-07-09, kullanici istegi).

Tarihsel (build_traffic_clusters.py) density_flights_res5.parquet'e
dayaniyordu -- burada AYNI bolgesel-DBSCAN pipeline'i (geo_clustering.py),
ama InfluxDB'nin "-7d" penceresinden HESAPLANAN hex yogunlugu uzerinde
calisiyor. SADECE 7d desteklenir -- live (~3dk) ve 24h pencereleri
DBSCAN icin yeterli veri biriktirmiyor (bkz. build_traffic_clusters
docstring, "hub" kavraminin uzun-vadeli yapisal bir orunutu olmasi
gerektigi notu), bu yuzden frontend'de sadece 7d'de "Trafik Kumeleri"
secenegi acik birakiliyor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h3

from individual.metehan_geo.data import clean_coordinates
from individual.metehan_geo.geo import assign_h3_cell, h3_cell_to_polygon
from individual.metehan_geo.geo_clustering import compute_knn_local_mask, run_dbscan_two_pass, summarize_clusters
from individual.metehan_geo.influx_client import load_realtime_window

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "viz" / "data"
H3_RESOLUTION = 5

# 7d realtime pencere olcegi tarihselden (243.835 hex, p95~1527) COK farkli
# (147.024 hex, p95~84) -- ayni oranlarda ama kucuk mutlak sayilarla
# calisiyoruz. min_absolute tarihseldeki 50 yerine 5 (p50=7 civarinda).
# 2026-07-10: grid yerine KNN-tabanli yerel esik (bkz. build_traffic_clusters.py
# ve geo_clustering.compute_knn_local_mask docstring'i) -- K_NEIGHBORS
# tarihseldeki DOGRULANMIS deger (1000) ile AYNI tutuldu (kucuk bir k'nin
# gercek hub'lari kaybettirdigi zaten ampirik olarak gosterildi, burada
# yeni/test edilmemis bir deger denemek yerine ayni guvenli secim korundu).
K_NEIGHBORS = 1000
PERCENTILE = 0.95
MIN_ABSOLUTE = 5
EPS_KM = 50
MIN_SAMPLES = 15           # Pass 1 (siki) -- tarihseldeki 30'un yarisi, kucuk veri hacmi
MIN_SAMPLES_RELAXED = 7    # Pass 2 (gevsek, SADECE Pass-1 gurultusu uzerinde) -- ayni oran (15'in yarisi)
# 2026-07-09 (kullanici bulgusu, tarihsel veride Istanbul/Bukres ornegiyle
# tespit edildi): tek-gecisli DBSCAN, yayilmis-ama-gercek hub'lari kaybediyordu
# -- bkz. build_traffic_clusters.py ve geo_clustering.run_dbscan_two_pass
# docstring'i. Ayni iki-gecisli duzeltme burada da uygulaniyor.


# 2026-07-09 (kullanici karari + olcumle dogrulandi): sunucu-tarafi
# aggregateWindow ile 7d sorgusunu hafiflet. ONCE 2dk denendi ama -6h'lik bir
# pencerede ham veriyle KARSILASTIRINCA kuresel flight_count farki %25.6
# cikti (HER buyuklukteki hex'te tutarli ~%25-27 kayip -- producer ~60sn'de
# bir yazdigi icin 2dk pencere ham noktalarin YARISINI atiyor) VE regional
# mask esigini gecen hex sayisi 6391 -> 6006'ya dustu (1043 hex kayboldu,
# 658 yeni/gurultu hex girdi -- DBSCAN girdisini gercekten degistirecek
# kadar buyuk bir fark). 1dk'ya dusurulunce kuresel fark -%0.25'e (ihmal
# edilebilir) indi, mask esigini gecen hex kumesi de neredeyse sabit kaldi
# (6408 -> 6383, sadece 47 kayip + 22 yeni, ortak 6361). SONUC: 1dk kullan.
AGG_WINDOW = "1m"


def build_hex_density_from_realtime(range_start: str = "-7d") -> "pd.DataFrame":
    import pandas as pd  # noqa: F401 (tip ipucu icin, gercek import zaten pandas'ta)

    df = load_realtime_window(range_start, agg_every=AGG_WINDOW)
    cleaned = clean_coordinates(df)
    chunk = assign_h3_cell(cleaned, H3_RESOLUTION)
    grouped = chunk.groupby("h3_cell").agg(flight_count=("source_id", "nunique")).reset_index()
    lat, lon = zip(*(h3.cell_to_latlng(h) for h in grouped["h3_cell"]))
    grouped = grouped.assign(lat=lat, lon=lon)
    logger.info("build_hex_density_from_realtime: %s -> %d hex", range_start, len(grouped))
    return grouped


def build_and_save() -> None:
    density_df = build_hex_density_from_realtime("-7d")
    mask = compute_knn_local_mask(
        density_df, k_neighbors=K_NEIGHBORS, percentile=PERCENTILE, min_absolute=MIN_ABSOLUTE,
    )
    clustered = run_dbscan_two_pass(
        density_df[mask], eps_km=EPS_KM, min_samples_strict=MIN_SAMPLES, min_samples_relaxed=MIN_SAMPLES_RELAXED,
    )
    summary = summarize_clusters(clustered)

    features = []
    for row in clustered[clustered["cluster"] != -1].itertuples(index=False):
        ring = h3_cell_to_polygon(row.h3_cell)
        if ring is None:
            continue
        features.append({
            "type": "Feature",
            "properties": {"h3_cell": row.h3_cell, "cluster": int(row.cluster), "flight_count": int(row.flight_count)},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "clusters": summary.to_dict("records"),
        "params": {
            "k_neighbors": K_NEIGHBORS, "percentile": PERCENTILE, "min_absolute": MIN_ABSOLUTE,
            "eps_km": EPS_KM, "min_samples": MIN_SAMPLES, "min_samples_relaxed": MIN_SAMPLES_RELAXED,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "traffic_clusters_7d.geojson"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f)
    logger.info("Yazildi: %s (%d hex, %d kume)", out_path, len(features), len(summary))


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="7 gunluk realtime trafik kumeleri (DBSCAN)")
    parser.add_argument(
        "--loop-seconds", type=int, default=0,
        help="0 = tek seferlik. >0 ise bu araliklarla surekli yeniden hesaplar. "
             "Her calisma InfluxDB'den ~170sn'lik bir sorgu iceriyor (4.4M+ satir, "
             "2026-07-09 olcumu) -- hub tespiti icin saniye-hassasiyetinde tazelik "
             "gerekmiyor, 1800sn (30dk) gibi rahat bir aralik oneriliyor.",
    )
    args = parser.parse_args()

    if args.loop_seconds <= 0:
        build_and_save()
        return

    logger.info("Loop modu: 7d kumeleri her %ds bir yeniden hesaplanacak (Ctrl+C durdurur)", args.loop_seconds)
    while True:
        try:
            build_and_save()
        except Exception:
            logger.exception("7d kume hesaplamasi basarisiz -- bir sonraki turda tekrar denenecek")
        time.sleep(args.loop_seconds)


if __name__ == "__main__":
    main()
