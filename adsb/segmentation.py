"""Faz 0.3: surekli ICAO24 trace akisini ayrik ucuslara bolme.

adsb.lol/readsb bir ucak icin gunluk TEK surekli trace verir -- ucus/inis/kalkis
onceden kesilmemis. Iki bagimsiz sinyal kullanilir: (1) zaman-boslugu kurali
(varsayilan 1800s = 30dk) ve (2) readsb'nin kendi `flags_new_leg` bayragi
(format referansi SS4). Ikisi birbirini kor bicimde ezmez; `new_leg_agreement()`
gercek uyusma oranini olcer.

0.1'deki envanterde (adsb/reports/inventory_profile.json) gozlenen ornekleme-araligi
dagilimi bu esigi destekliyor: cogu ardisik kayit 0-20s araliginda (normal ADS-B ping
hizi), ama uzun-kuyruklu binlerce saniyelik boslukar da var (ucak menzil disi/yerde
sessiz) -- 1800s bu ikisini ayirmak icin makul bir baslangic noktasi, kesin degil.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_GAP_S = 1800.0


def assign_flight_ids(
    df: pd.DataFrame,
    *,
    id_col: str = "source_id",
    time_col: str = "timestamp_utc",
    gap_s: float = DEFAULT_GAP_S,
) -> pd.Series:
    """Her `id_col` grubu icinde `time_col` boslugu > gap_s ise yeni ucus baslat.

    df.index hizali bir pd.Series doner, degerler "{id}_{seq:03d}". Girdi onceden
    siralanmis olmak zorunda degil.
    """
    if df.empty:
        return pd.Series([], dtype=object, index=df.index)

    order = df.sort_values([id_col, time_col]).index
    ordered = df.loc[order]

    gap = ordered.groupby(id_col, sort=False)[time_col].diff()
    is_new_flight = gap.isna() | (gap > gap_s)
    seq = is_new_flight.groupby(ordered[id_col]).cumsum().astype(int) - 1

    flight_id = ordered[id_col].astype(str) + "_" + seq.map(lambda s: f"{s:03d}")
    return flight_id.reindex(df.index)


def segment_flights(
    df: pd.DataFrame,
    *,
    id_col: str = "source_id",
    time_col: str = "timestamp_utc",
    gap_s: float = DEFAULT_GAP_S,
    flight_id_col: str = "flight_id",
) -> pd.DataFrame:
    """`assign_flight_ids` sonucunu ekleyip id+zaman sirasina gore dondurur."""
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
    """Bosluk-tabanli ucus sinirlarinin `flags_new_leg` ile uyusma orani.

    df, `segment_flights` cikisi gibi id+zaman sirali olmali. Her ucusun ilk satiri
    (o id'nin genel ilk satiri haric) bir "sinir"dir; o satirda `new_leg_col` True
    ise uyusma sayilir. NaN -> uyusmuyor sayilir. Sinir yoksa (tum id'ler tek ucus)
    NaN doner.
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

    agreed = int(ordered.loc[boundary, new_leg_col].fillna(False).sum())
    return agreed / n_boundaries


def flight_summary(df: pd.DataFrame, *, flight_id_col: str = "flight_id",
                    time_col: str = "timestamp_utc") -> pd.DataFrame:
    """Her ucus icin satir sayisi, sure (s) ve baslangic zamani -- galeri secimi icin."""
    g = df.groupby(flight_id_col, sort=False)
    return pd.DataFrame({
        "n_rows": g.size(),
        "duration_s": g[time_col].apply(lambda s: s.max() - s.min()),
        "start_time": g[time_col].min(),
    }).reset_index()
