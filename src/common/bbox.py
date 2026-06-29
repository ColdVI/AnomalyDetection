"""Geographic bounds used by adsb.lol Bronze ingestion."""

from __future__ import annotations

import math
from typing import Any

TURKEY_BBOX = {"lat": (36.0, 42.0), "lon": (26.0, 45.0)}


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def in_turkey(lat: Any, lon: Any) -> bool:
    """Return whether a coordinate is inside the inclusive Turkey bbox."""
    latitude = _finite_float(lat)
    longitude = _finite_float(lon)
    if latitude is None or longitude is None:
        return False
    min_lat, max_lat = TURKEY_BBOX["lat"]
    min_lon, max_lon = TURKEY_BBOX["lon"]
    return min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon
