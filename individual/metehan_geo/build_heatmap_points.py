"""build_heatmap_points.py -- hexagon poligonlarindan MapLibre `heatmap`
layer'i icin nokta+weight GeoJSON uretir (2026-07-07, kullanicinin
"heatmap glow" (magma/inferno estetigi) istegi).

MapLibre'nin native heatmap layer'i Point geometry + sayisal bir
`weight` property bekliyor (Polygon degil). Zaten hesaplanmis hex
yogunluk tablolarindan (density.parquet / density_flights_res5.parquet)
h3.cell_to_latlng() ile her hex'in MERKEZ noktasini alip weight=flight_count
(veya point_count) tasiyan bir Point FeatureCollection uretiyoruz --
1 milyar satiri tekrar taramaya GEREK YOK, zaten kucuk (res5: ~244K satir)
agregat tablo uzerinde calisiyoruz.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h3
import pandas as pd

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "viz" / "data"


def build_points_geojson(parquet_path: Path, weight_col: str, extra_cols: list[str] | None = None) -> dict:
    df = pd.read_parquet(parquet_path)
    extra_cols = extra_cols or []

    features = []
    for row in df.itertuples(index=False):
        h3_cell = row.h3_cell
        lat, lon = h3.cell_to_latlng(h3_cell)
        weight = getattr(row, weight_col)
        properties = {"h3_cell": h3_cell, "weight": float(weight)}
        for col in extra_cols:
            properties[col] = getattr(row, col)
        features.append({
            "type": "Feature",
            "properties": properties,
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        })

    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # flight_count metrigi (duzeltilmis, varsayilan) -- day_count'u da tasi,
    # heatmap modunda da gun-sayisi filtresi calisabilsin diye.
    flight_geojson = build_points_geojson(
        OUT_DIR / "density_flights_res5.parquet", "flight_count", extra_cols=["day_count"]
    )
    with open(OUT_DIR / "heatmap_points_flight.geojson", "w", encoding="utf-8") as f:
        json.dump(flight_geojson, f)
    logger.info("Yazildi: heatmap_points_flight.geojson (%d nokta)", len(flight_geojson["features"]))

    # point_count metrigi (eski/ham, karsilastirma icin)
    point_geojson = build_points_geojson(OUT_DIR / "density.parquet", "point_count")
    with open(OUT_DIR / "heatmap_points_point.geojson", "w", encoding="utf-8") as f:
        json.dump(point_geojson, f)
    logger.info("Yazildi: heatmap_points_point.geojson (%d nokta)", len(point_geojson["features"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
