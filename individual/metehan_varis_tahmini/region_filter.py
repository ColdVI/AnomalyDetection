"""Avrupa+Asya icin GERCEK kita/ulke sinirlarina dayali (kutu DEGIL) hizli
vektorel nokta-icinde-mi testi.

Onceki versiyon (EUROPE_BBOX/ASIA_BBOX, basit lat/lon araligi) Dogu Turkiye,
Iran, Irak, Suriye, Suudi Arabistan'i kapsayan lon 40-60 arasinda GERCEK bir
bosluk birakiyordu -- bu bolgeden gecen/bu bolge icindeki gercek ucuslar
yanlislikla eleniyordu. Bu modul, Natural Earth ulke poligonlarinin birlesimini
(CONTINENT=Europe veya Asia) kullanarak dogru sinir testi yapar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely
from shapely.ops import unary_union

NE_GEOJSON = Path(r"c:\Users\PC_6276\Desktop\github\AnomalyDetection\individual\metehan_geo_country\data\ne_110m_admin_0_countries.geojson")

_EUROPE_ASIA_GEOM = None


def _load_geometry():
    global _EUROPE_ASIA_GEOM
    if _EUROPE_ASIA_GEOM is not None:
        return _EUROPE_ASIA_GEOM
    with open(NE_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    shapes = []
    for feat in gj["features"]:
        if feat["geometry"] is None:
            continue
        if feat["properties"].get("CONTINENT") in ("Europe", "Asia"):
            shapes.append(shapely.geometry.shape(feat["geometry"]))
    _EUROPE_ASIA_GEOM = unary_union(shapes)
    return _EUROPE_ASIA_GEOM


def in_europe_or_asia(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """lats/lons: aligned numpy arrays. Donus: ayni boyutta boolean array."""
    geom = _load_geometry()
    pts = shapely.points(lons, lats)
    return shapely.contains(geom, pts)
