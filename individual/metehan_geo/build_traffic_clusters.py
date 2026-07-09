"""build_traffic_clusters.py -- dbscan_geo_hotspot_prompt.md Adim 3.

geo_clustering.py'nin DBSCAN sonucunu (Avrupa+Ortadogu prototipinde
dogrulandi, bkz. docs) TUM DUNYA veresine olcekleyip viz/index.html'in
"Trafik Kumeleri" modu icin GeoJSON uretir.

Bolgesel goreceli esik kullanir (bkz. geo_clustering.compute_regional_mask
docstring'i, 2026-07-09 duzeltmesi) -- TEK global esik degil.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from individual.metehan_geo.geo import h3_cell_to_polygon
from individual.metehan_geo.geo_clustering import (
    compute_regional_mask,
    load_hex_density,
    run_dbscan_two_pass,
    summarize_clusters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "viz" / "data"

# 2026-07-09 duzeltmesi: TEK global esik (min_flight_count=1723, dunya p95)
# yerine BOLGESEL goreceli esik -- ABD/Avrupa'nin ADS-B alici-istasyon
# yogunlugu kaynakli mutlak-sayi ustunlugu Korfez/Guney Asya/Guney Amerika/
# Afrika/Avustralya hub'larini tamamen gizliyordu (kanit: bu bolgelerin
# KENDI p95'leri global esigin altinda). 25 derecelik kaba grid + hucre-ici
# p95 ile artik TUM kitalarda hub bulunuyor (Dubai, Sao Paulo, Hong Kong,
# Sidney, Johannesburg, Hyderabad vb. -- bkz. docs).
GRID_DEG = 25.0
PERCENTILE = 0.95
MIN_ABSOLUTE = 50
MIN_CELL_HEXES = 20
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
    mask = compute_regional_mask(
        df, grid_deg=GRID_DEG, percentile=PERCENTILE,
        min_absolute=MIN_ABSOLUTE, min_cell_hexes=MIN_CELL_HEXES,
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
            "grid_deg": GRID_DEG, "percentile": PERCENTILE,
            "min_absolute": MIN_ABSOLUTE, "min_cell_hexes": MIN_CELL_HEXES,
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
