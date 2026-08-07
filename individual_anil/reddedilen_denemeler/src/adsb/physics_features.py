"""ADSB-0: kolonlar-arasi fiziksel-tutarlilik residual'lari.

Bunlar OGRENILMIS ozellikler DEGIL -- bildirilen bir kanalla, aynı fiziksel
niceligin ham lat/lon/alt'tan turetilen (finite-difference) haliyle arasindaki
aritmetik fark. SEAD/RFLY'deki LSTM/Dense-AE/USAD skorlarini surukleyen "genlik-
baskinligi" artefakti (ADR-016/017/018/019) burada yapisal olarak olusamaz: residual
sifira yakinsa kanal tutarli demektir, buyukluk (magnitude) tek basina hicbir sey
ifade etmez.

Her fonksiyon `segmentation.segment_flights()` cikisi gibi bir df bekler (bir
`flight_id_col` kolonu olmali, ucus siniri disina tasan diff hesaplanmaz -- her
grubun ilk satirinda NaN doner). `compute_physics_residuals()` hepsini birden
ekler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0
G_MPS2 = 9.80665


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
    """a-b, [-180, 180] araligina sarilmis (dairesel fark)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def vertical_rate_residual(
    df: pd.DataFrame,
    *,
    flight_id_col: str = "flight_id",
    alt_col: str = "alt",
    vrate_col: str = "vertical_rate_ms",
    time_col: str = "timestamp_utc",
) -> pd.Series:
    """Bildirilen dikey hiz - olculen d(alt)/dt (m/s). Alcalma/tirmanma yonu isaretiyle."""
    dt = _group_diff(df, time_col, flight_id_col)
    dalt = _group_diff(df, alt_col, flight_id_col)
    measured = dalt / dt.replace(0, np.nan)
    return df[vrate_col] - measured


def speed_residual(
    df: pd.DataFrame,
    *,
    flight_id_col: str = "flight_id",
    lat_col: str = "lat",
    lon_col: str = "lon",
    speed_col: str = "ground_speed_ms",
    time_col: str = "timestamp_utc",
) -> pd.Series:
    """Bildirilen yer hizi - konum farkindan (haversine) turetilen hiz (m/s)."""
    g = df.groupby(flight_id_col, sort=False)
    lat_prev, lon_prev = g[lat_col].shift(1), g[lon_col].shift(1)
    dt = _group_diff(df, time_col, flight_id_col)
    dist_m = _haversine_m(lat_prev, lon_prev, df[lat_col], df[lon_col])
    measured = dist_m / dt.replace(0, np.nan)
    return df[speed_col] - measured


def heading_residual(
    df: pd.DataFrame,
    *,
    flight_id_col: str = "flight_id",
    lat_col: str = "lat",
    lon_col: str = "lon",
    track_col: str = "track_deg",
) -> pd.Series:
    """Bildirilen track - ardisik konumlardan turetilen kerteriz (derece, dairesel fark)."""
    g = df.groupby(flight_id_col, sort=False)
    lat_prev, lon_prev = g[lat_col].shift(1), g[lon_col].shift(1)
    measured_bearing = _bearing_deg(lat_prev, lon_prev, df[lat_col], df[lon_col])
    # ilk satir (lat_prev NaN) -> _bearing_deg NaN uretir, dogal olarak yayilir
    return pd.Series(_circular_diff_deg(df[track_col], measured_bearing), index=df.index)


def turn_bank_residual(
    df: pd.DataFrame,
    *,
    flight_id_col: str = "flight_id",
    track_col: str = "track_deg",
    roll_col: str = "roll_deg",
    speed_col: str = "ground_speed_ms",
    time_col: str = "timestamp_utc",
) -> pd.Series:
    """Bildirilen donus hizi (d(track)/dt) - koordineli-donus fizigi g*tan(roll)/v (derece/s).

    `roll_col` cogu kayitta yok (ampirik kapsama ~%8.5, format referansi §5.1) --
    eksik oldugu yerlerde NaN doner, sonuc ikincil/seyrek sinyal olarak kullanilmali.
    """
    dt = _group_diff(df, time_col, flight_id_col)
    dtrack = _circular_diff_deg(df[track_col], df.groupby(flight_id_col, sort=False)[track_col].shift(1))
    observed_rate = dtrack / dt.replace(0, np.nan)
    speed = df[speed_col].replace(0, np.nan)
    predicted_rate = np.degrees(G_MPS2 * np.tan(np.radians(df[roll_col])) / speed)
    return observed_rate - predicted_rate


def compute_physics_residuals(
    df: pd.DataFrame,
    *,
    flight_id_col: str = "flight_id",
    time_col: str = "timestamp_utc",
    lat_col: str = "lat",
    lon_col: str = "lon",
    alt_col: str = "alt",
    speed_col: str = "ground_speed_ms",
    track_col: str = "track_deg",
    vrate_col: str = "vertical_rate_ms",
    roll_col: str = "roll_deg",
) -> pd.DataFrame:
    """4 residual'i birden hesaplayip df'e ekler (yeni kopya dondurur, girdi degismez)."""
    out = df.copy()
    out["vertical_rate_residual"] = vertical_rate_residual(
        df, flight_id_col=flight_id_col, alt_col=alt_col, vrate_col=vrate_col, time_col=time_col
    )
    out["speed_residual"] = speed_residual(
        df, flight_id_col=flight_id_col, lat_col=lat_col, lon_col=lon_col,
        speed_col=speed_col, time_col=time_col,
    )
    out["heading_residual"] = heading_residual(
        df, flight_id_col=flight_id_col, lat_col=lat_col, lon_col=lon_col, track_col=track_col
    )
    if roll_col in df.columns:
        out["turn_bank_residual"] = turn_bank_residual(
            df, flight_id_col=flight_id_col, track_col=track_col, roll_col=roll_col,
            speed_col=speed_col, time_col=time_col,
        )
    else:
        out["turn_bank_residual"] = np.nan
    return out
