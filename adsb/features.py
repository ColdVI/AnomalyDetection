"""Faz 0.5 kararinin uygulanmasi: fiziksel-tutarlilik residual'lari + birlesik
feature tablosu. Bkz. adsb/reports/measurability_table.md -- hangi iliskinin
olculebilir/yalniz-gecişte-olculebilir/olculemez oldugu kararinin gerekcesi orada.

Residual'lar OGRENILMIS DEGIL, aritmetik ozdeslik -- SEAD'i vuran genlik-
baskinligi artefakti (archive/.../decisions.md ADR-016) burada yapisal olarak
olusamaz: residual sifira yakinsa kanal tutarlidir, buyukluk tek basina hicbir
sey ifade etmez. `flight_id` bazinda gruplanmis diff/shift kullanilir -- ucus
sinirinin disina hicbir hesap tasmaz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0
G_MPS2 = 9.80665

PRIMARY_FEATURES = [
    "alt", "ground_speed_ms", "track_deg", "vertical_rate_ms",
    "vertical_rate_residual", "speed_residual", "heading_residual",
]
SECONDARY_FEATURES = ["roll_deg", "turn_bank_residual"]


def _group_diff(df: pd.DataFrame, col: str, group_col: str) -> pd.Series:
    return df.groupby(group_col, sort=False)[col].diff()


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def _circular_diff_deg(a, b) -> np.ndarray:
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def vertical_rate_residual(df: pd.DataFrame, *, flight_id_col="flight_id",
                            alt_col="alt", vrate_col="vertical_rate_ms",
                            time_col="timestamp_utc") -> pd.Series:
    dt = _group_diff(df, time_col, flight_id_col)
    dalt = _group_diff(df, alt_col, flight_id_col)
    return df[vrate_col] - dalt / dt.replace(0, np.nan)


def speed_residual(df: pd.DataFrame, *, flight_id_col="flight_id", lat_col="lat",
                    lon_col="lon", speed_col="ground_speed_ms", time_col="timestamp_utc") -> pd.Series:
    g = df.groupby(flight_id_col, sort=False)
    lat_prev, lon_prev = g[lat_col].shift(1), g[lon_col].shift(1)
    dt = _group_diff(df, time_col, flight_id_col)
    dist_m = _haversine_m(lat_prev, lon_prev, df[lat_col], df[lon_col])
    return df[speed_col] - dist_m / dt.replace(0, np.nan)


def heading_residual(df: pd.DataFrame, *, flight_id_col="flight_id", lat_col="lat",
                      lon_col="lon", track_col="track_deg") -> pd.Series:
    g = df.groupby(flight_id_col, sort=False)
    lat_prev, lon_prev = g[lat_col].shift(1), g[lon_col].shift(1)
    measured_bearing = _bearing_deg(lat_prev, lon_prev, df[lat_col], df[lon_col])
    return pd.Series(_circular_diff_deg(df[track_col], measured_bearing), index=df.index)


def turn_bank_residual(df: pd.DataFrame, *, flight_id_col="flight_id", track_col="track_deg",
                        roll_col="roll_deg", speed_col="ground_speed_ms", time_col="timestamp_utc") -> pd.Series:
    dt = _group_diff(df, time_col, flight_id_col)
    dtrack = _circular_diff_deg(df[track_col], df.groupby(flight_id_col, sort=False)[track_col].shift(1))
    observed_rate = dtrack / dt.replace(0, np.nan)
    speed = df[speed_col].replace(0, np.nan)
    predicted_rate = np.degrees(G_MPS2 * np.tan(np.radians(df[roll_col])) / speed)
    return observed_rate - predicted_rate


def build_feature_table(df: pd.DataFrame, *, flight_id_col: str = "flight_id") -> pd.DataFrame:
    """PRIMARY_FEATURES + SECONDARY_FEATURES kolonlarini ekler (yeni kopya doner)."""
    out = df.copy()
    out["vertical_rate_residual"] = vertical_rate_residual(df, flight_id_col=flight_id_col)
    out["speed_residual"] = speed_residual(df, flight_id_col=flight_id_col)
    out["heading_residual"] = heading_residual(df, flight_id_col=flight_id_col)
    if "roll_deg" in df.columns:
        out["turn_bank_residual"] = turn_bank_residual(df, flight_id_col=flight_id_col)
    else:
        out["turn_bank_residual"] = np.nan
    return out
