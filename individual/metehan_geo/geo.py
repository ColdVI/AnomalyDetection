"""geo.py -- Bolum 5, Adim 3-6.1 (docs/BIREYSEL_PROJE_MASTER (1).md)."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterator

import h3
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_bbox(df: pd.DataFrame, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> pd.DataFrame:
    """Istege bagli bolgesel filtre (analiz karari, pipeline'da yok)."""
    return df[
        df["lat"].between(min_lat, max_lat) & df["lon"].between(min_lon, max_lon)
    ]


def assign_h3_cell(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Her satira h3_cell kolonu ekler (h3.latlng_to_cell, h3-py v4 API)."""
    out = df.copy()
    out["h3_cell"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(out["lat"], out["lon"])
    ]
    return out


def h3_cell_to_polygon(hex_id: str) -> list[list[float]]:
    """Hex sinirini GeoJSON Polygon `coordinates` formatina cevirir.

    GeoJSON [lon, lat] sirasi bekler (h3.cell_to_boundary [lat, lon] donuyor),
    ve halka (ring) kapali olmali (ilk nokta = son nokta).
    """
    boundary = h3.cell_to_boundary(hex_id)  # [(lat, lon), ...]
    ring = [[lon, lat] for lat, lon in boundary]
    ring.append(ring[0])
    return ring


def compute_hex_density(chunks: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """`load_adsb_gold_data()`'den gelen chunk'lari tuketip hex basina nokta
    sayisini biriktirir. Toplam SATIR sayisi degil toplam BENZERSIZ HEX sayisi
    bellekte tutuluyor (resolution'a gore binler-milyonlar arasi) -- bu yuzden
    1 milyar satirlik girdide bile guvenle calisir.
    """
    counts: Counter[str] = Counter()
    total_rows = 0
    for chunk in chunks:
        if "h3_cell" not in chunk.columns:
            raise ValueError("compute_hex_density: chunk'ta h3_cell kolonu yok -- once assign_h3_cell uygula")
        vc = chunk["h3_cell"].value_counts()
        counts.update(vc.to_dict())
        total_rows += len(chunk)

    logger.info("compute_hex_density: %d satir, %d benzersiz hex", total_rows, len(counts))
    return pd.DataFrame(
        {"h3_cell": list(counts.keys()), "point_count": list(counts.values())}
    )


def reservoir_sample_chunks(chunks: Iterator[pd.DataFrame], max_points: int, *, seed: int = 42) -> pd.DataFrame:
    """Streaming rezervuar orneklemesi -- kumeleme (Bolum 6.1) icin sabit
    boyutlu bir alt kume biriktirir, chunk sayisi/boyutu onceden bilinmeden.

    Klasik tek-satirlik rezervuar yerine (cok yavas olurdu), her chunk kendi
    icinde rastgele alt-orneklenir ve global bir rezervuara eklenir; rezervuar
    max_points'i asarsa rastgele kirpilir. Kesin istatistiksel garantisi olan
    saf rezervuar orneklemesi kadar hassas degil ama pratik amac icin
    (kumeleme girdisi kucultme) yeterli ve cok daha hizli.
    """
    rng = np.random.default_rng(seed)
    reservoir: list[pd.DataFrame] = []
    reservoir_size = 0

    for chunk in chunks:
        n = len(chunk)
        if n == 0:
            continue
        # Her chunk'tan en fazla max_points kadar al (chunk zaten kucukse hepsini al).
        take = min(n, max_points)
        idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        reservoir.append(chunk.iloc[idx])
        reservoir_size += take

        if reservoir_size > max_points * 2:
            # Rezervuar cok buyudugunde birlestirip max_points'e kirp -- bellek
            # sinirsizca artmasin.
            combined = pd.concat(reservoir, ignore_index=True)
            if len(combined) > max_points:
                keep_idx = rng.choice(len(combined), size=max_points, replace=False)
                combined = combined.iloc[keep_idx]
            reservoir = [combined]
            reservoir_size = len(combined)

    if not reservoir:
        return pd.DataFrame()

    result = pd.concat(reservoir, ignore_index=True)
    if len(result) > max_points:
        keep_idx = rng.choice(len(result), size=max_points, replace=False)
        result = result.iloc[keep_idx].reset_index(drop=True)
    logger.info("reservoir_sample_chunks: %d nokta orneklendi (max_points=%d)", len(result), max_points)
    return result
