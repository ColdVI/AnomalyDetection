"""build_traffic_clusters.py -- dbscan_geo_hotspot_prompt.md Adim 3.

geo_clustering.py'nin DBSCAN sonucunu (Avrupa+Ortadogu prototipinde
dogrulandi, bkz. docs) TUM DUNYA veresine olcekleyip viz/index.html'in
"Trafik Kumeleri" modu icin GeoJSON uretir.

KNN-tabanli yerel esik kullanir (bkz. geo_clustering.compute_knn_local_mask
docstring'i, 2026-07-10 duzeltmesi) -- ne TEK global esik, ne de sert
sinirli bir grid.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from individual.metehan_geo.geo import h3_cell_to_polygon
from individual.metehan_geo.geo_clustering import (
    compute_knn_local_mask,
    load_hex_density,
    run_dbscan_two_pass,
    summarize_clusters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "viz" / "data"

# 2026-07-09: TEK global esik (min_flight_count=1723, dunya p95) yerine
# BOLGESEL goreceli esik (25 derecelik grid) -- ABD/Avrupa'nin ADS-B alici-
# istasyon yogunlugu kaynakli mutlak-sayi ustunlugu Korfez/Guney Asya/Guney
# Amerika/Afrika/Avustralya hub'larini gizliyordu.
# 2026-07-10: gridin KENDISI de sorunlu cikti (Paris/Frankfurt/Londra grid
# sinirinda FARKLI esiklerle degerlendiriliyordu, bkz. compute_knn_local_mask
# docstring'i) -- KNN-tabanli yerel esige gecildi, sert sinir yok.
K_NEIGHBORS = 1000
PERCENTILE = 0.95
MIN_ABSOLUTE = 50
EPS_KM = 50
MIN_SAMPLES = 30            # Pass 1 (siki) -- kompakt hub'lar (bkz. Bukres)
MIN_SAMPLES_RELAXED = 15    # Pass 2 (gevsek, SADECE Pass-1 gurultusu uzerinde)
# 2026-07-09 (kullanici bulgusu): tek-gecisli DBSCAN, Istanbul gibi COK
# BUYUK/YAYILMIS ama gercek hub'lari (Bogaz'in iki yakasi) kaybediyordu --
# hicbir hex kendi 50km cevresinde MIN_SAMPLES=30 komsu bulamiyordu, oysa
# bolgede toplam 38 esik-gecen hex vardi. TEK cozum min_samples'i GLOBAL
# gevsetmekti ama bu Bati Avrupa'nin ayri kumelerini (Londra/Paris/Frankfurt)
# TEK bir 600km+ mega-blob'a geri birlestiriyordu. Iki gecisli yaklasim
# (bkz. geo_clustering.run_dbscan_two_pass docstring) ikisini de cozuyor.


def main() -> None:
    df = load_hex_density(5)
    mask = compute_knn_local_mask(
        df, k_neighbors=K_NEIGHBORS, percentile=PERCENTILE, min_absolute=MIN_ABSOLUTE,
    )
    clustered = run_dbscan_two_pass(
        df[mask], eps_km=EPS_KM, min_samples_strict=MIN_SAMPLES, min_samples_relaxed=MIN_SAMPLES_RELAXED,
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
    out_path = OUT_DIR / "traffic_clusters.geojson"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f)
    logger.info("Yazildi: %s (%d hex, %d kume)", out_path, len(features), len(summary))


if __name__ == "__main__":
    main()
