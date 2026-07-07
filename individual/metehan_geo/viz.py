"""viz.py -- Bolum 5, Adim 7-8 (docs/BIREYSEL_PROJE_MASTER (1).md).

Folium DEGIL -- GeoJSON export (Bolum 5.1 karari). Ciktilar
individual/metehan_geo/viz/data/ altina yazilir, ayri bir MapLibre GL JS
(vanilla, CDN) sayfasi (individual/metehan_geo/viz/index.html) bunlari
fetch() ile okuyup cizer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from individual.metehan_geo.geo import h3_cell_to_polygon

logger = logging.getLogger(__name__)


def build_density_geojson(density_df: pd.DataFrame) -> dict:
    """Her hex'i point_count'lu bir GeoJSON Feature yapar.

    Renklendirme (fill-color data-driven expression) MapLibre tarafinda
    (index.html) yapilir -- burada sadece geometri + sayi tasinir.
    """
    features = []
    for row in density_df.itertuples(index=False):
        features.append({
            "type": "Feature",
            "properties": {"h3_cell": row.h3_cell, "point_count": int(row.point_count)},
            "geometry": {"type": "Polygon", "coordinates": [h3_cell_to_polygon(row.h3_cell)]},
        })
    return {"type": "FeatureCollection", "features": features}


def save_geojson(geojson_obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(geojson_obj, f)
    logger.info("Yazildi: %s (%d feature)", path, len(geojson_obj.get("features", [])))
