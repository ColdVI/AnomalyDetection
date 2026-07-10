"""ADSB-0: sürekli ICAO24 trace akışını ayrık uçuşlara bölme.

adsb.lol/readsb bir uçak için günlük TEK sürekli trace verir (uçuş yok, iniş/kalkış
yok) -- ALFA/SEAD gibi önceden uçuş-uçuş kesilmiş değil. Burada iki bağımsız sinyal
kullanılır: (1) zaman-boşluğu kuralı (windowing.py'deki max_gap_s diff-kesme deseninin
segmentasyon versiyonu) ve (2) readsb'nin kendi `flags_new_leg` bayrağı (format
referansı, adsblo_data_format_reference.md §4). İkisi birbirini kör biçimde ezmez;
`new_leg_agreement()` uyuşma oranını raporlar, ADSB-0 kapısı bu oranın makul (plan
dokümanında önceden yazılan bir eşiğin üstünde) olmasını ister.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_GAP_S = 1800.0  # 30 dakika


def assign_flight_ids(
    df: pd.DataFrame,
    *,
    id_col: str = "source_id",
    time_col: str = "timestamp_utc",
    gap_s: float = DEFAULT_GAP_S,
) -> pd.Series:
    """Her `id_col` grubu içinde `time_col` boşluğu > gap_s ise yeni uçuş başlat.

    Returns bir pd.Series (df.index hizalı), degerler "{id}_{seq:03d}" formatinda.
    Girdi df sirali olmak ZORUNDA degil -- fonksiyon kendi icinde id+zaman sirasina
    gore siralar, sonucu orijinal index'e geri hizalar.
    """
    if df.empty:
        return pd.Series([], dtype=object, index=df.index)

    order = df.sort_values([id_col, time_col]).index
    sorted_df = df.loc[order]

    gap = sorted_df.groupby(id_col, sort=False)[time_col].diff()
    new_flight = (gap.isna()) | (gap > gap_s)
    seq = new_flight.groupby(sorted_df[id_col]).cumsum().astype(int) - 1

    flight_id = sorted_df[id_col].astype(str) + "_" + seq.map(lambda s: f"{s:03d}")
    flight_id.index = order
    return flight_id.reindex(df.index)


def segment_flights(
    df: pd.DataFrame,
    *,
    id_col: str = "source_id",
    time_col: str = "timestamp_utc",
    gap_s: float = DEFAULT_GAP_S,
    flight_id_col: str = "flight_id",
) -> pd.DataFrame:
    """`assign_flight_ids` sonucunu df'e ekleyip id+zaman sirasina gore dondurur."""
    out = df.copy()
    out[flight_id_col] = assign_flight_ids(df, id_col=id_col, time_col=time_col, gap_s=gap_s)
    return out.sort_values([id_col, time_col]).reset_index(drop=True)


def new_leg_agreement(
    df: pd.DataFrame,
    *,
    flight_id_col: str = "flight_id",
    id_col: str = "source_id",
    time_col: str = "timestamp_utc",
    new_leg_col: str = "flags_new_leg",
) -> float:
    """Boşluk-tabanlı uçuş sınırlarının `flags_new_leg` ile uyuşma oranı.

    df, `segment_flights` çıktısı gibi id+zaman sıralı olmalı. Her uçuşun ilk satırı
    (o icao24'ün genel ilk satırı hariç) bir "sınır"dır; o satırda `new_leg_col`
    True ise uyuşma sayılır. NaN/eksik `new_leg_col` -> uyuşmuyor sayılır (iyimser
    yuvarlama yapılmaz).
    """
    if df.empty:
        return float("nan")

    ordered = df.sort_values([id_col, time_col])
    is_first_of_id = ordered[id_col] != ordered[id_col].shift(1)
    is_first_of_flight = ordered[flight_id_col] != ordered[flight_id_col].shift(1)
    boundary = is_first_of_flight & ~is_first_of_id

    n_boundaries = int(boundary.sum())
    if n_boundaries == 0:
        return float("nan")

    agreed_count = int(ordered.loc[boundary, new_leg_col].fillna(False).sum())
    return agreed_count / n_boundaries
