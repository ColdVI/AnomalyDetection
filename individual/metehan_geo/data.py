"""data.py -- Bolum 5, Adim 1-2 (docs/BIREYSEL_PROJE_MASTER (1).md).

Salt okunur: sadece src/common/minio_io.py'yi import eder, ortak pipeline'a
(src/) dokunmaz.
"""

from __future__ import annotations

import logging
from typing import Iterator

import pandas as pd

from src.common.minio_io import (
    ObjectStoreClient,
    get_minio_client,
    list_layer_objects,
    read_parquet_object,
)

logger = logging.getLogger(__name__)

ADSB_SOURCE_TYPES = frozenset({
    "adsblol_historical",
    "adsblol_hist",
    "adsblol_realtime",
    "adsblol_rt",
})


def load_adsb_gold_data(client: ObjectStoreClient | None = None, *, gold_bucket: str | None = None) -> Iterator[pd.DataFrame]:
    """Stream MinIO Gold's adsb rows one part at a time.

    Gerçek veri 1.006.744.756 satır / 2.500 Gold parçası çıktı (2026-07-07) --
    16GB RAM'e tek DataFrame olarak sığmaz. `read_layer()` (hepsini pd.concat
    eden) yerine burada her parça TEK TEK okunup adsb source_type'a filtrelenip
    yield edilir; hiçbir zaman tüm veri aynı anda bellekte olmaz.
    """
    import os

    client = client or get_minio_client()
    bucket = gold_bucket or os.getenv("MINIO_GOLD_BUCKET", "gold")

    object_names = list_layer_objects(client, bucket, "unified")
    logger.info("Gold'da %d parca bulundu, adsb satirlari icin taraniyor", len(object_names))

    total_rows = 0
    for i, name in enumerate(object_names):
        df = read_parquet_object(client, bucket, name)
        adsb_rows = df[df["source_type"].isin(ADSB_SOURCE_TYPES)]
        if adsb_rows.empty:
            continue
        total_rows += len(adsb_rows)
        if (i + 1) % 500 == 0:
            logger.info("  %d/%d parca tarandi, su ana kadar %d adsb satiri", i + 1, len(object_names), total_rows)
        yield adsb_rows

    logger.info("Gold tarama bitti: %d adsb satiri (toplam %d parcadan)", total_rows, len(object_names))


def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Null lat/lon'u at, gecersiz aralik disini (-90/90, -180/180) filtrele.

    Her chunk'a ayri ayri uygulanir (load_adsb_gold_data()'nin generator'ini
    tuketen dongude).
    """
    before = len(df)
    out = df.dropna(subset=["lat", "lon"])
    out = out[out["lat"].between(-90, 90) & out["lon"].between(-180, 180)]
    dropped = before - len(out)
    if dropped:
        logger.info("clean_coordinates: %d/%d satir silindi (null veya gecersiz aralik)", dropped, before)
    return out
