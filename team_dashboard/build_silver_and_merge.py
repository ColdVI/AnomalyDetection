"""Tek seferlik yardimci: Gold icin ONCEDEN olusturulmus indeksi (gold_index.py'nin
uretimi) yeniden tarama YAPMADAN yeni layer_index semasina tasir, Silver'i
YENIDEN (ilk kez) tarar, ikisini birlestirip export_layer_index.parquet'e yazar.
Ardindan kolon katalogunu (Silver icin gercek okuma, Gold icin GOLD_COLUMNS) uretir.
"""
import logging

import pandas as pd

from team_dashboard.gold_index import INDEX_PATH as OLD_GOLD_INDEX_PATH
from team_dashboard.layer_index import (
    INDEX_PATH,
    SILVER_DATASETS,
    _iter_part_stats,
    build_columns_catalog,
)
from src.common.minio_io import get_minio_client, list_layer_objects
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

client = get_minio_client()
silver_bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")

# 1) Gold: onceden uretilmis indeksi YENIDEN TARAMADAN donustur
old_gold = pd.read_parquet(OLD_GOLD_INDEX_PATH)
gold_rows = old_gold.rename(columns={"source_type": "dataset"})
gold_rows["layer"] = "gold"
gold_rows = gold_rows[["layer", "dataset", "object_name", "min_ts", "max_ts", "row_count"]]
logger.info("Gold (yeniden tarama YOK): %d parca tasindi", len(gold_rows))

# 2) Silver: ilk kez tara
silver_rows = []
for dataset in SILVER_DATASETS:
    object_names = list_layer_objects(client, silver_bucket, dataset)
    if not object_names:
        logger.info("Silver/%s: hic parca yok", dataset)
        continue
    logger.info("Silver/%s: %d parca taranacak", dataset, len(object_names))
    for i, (name, part) in enumerate(_iter_part_stats(client, silver_bucket, object_names)):
        ts = part["timestamp_utc"].dropna()
        if ts.empty:
            continue
        silver_rows.append({
            "layer": "silver", "dataset": dataset, "object_name": name,
            "min_ts": float(ts.min()), "max_ts": float(ts.max()), "row_count": int(len(ts)),
        })
        if (i + 1) % 500 == 0:
            logger.info("  Silver/%s: %d/%d parca indekslendi", dataset, i + 1, len(object_names))

silver_df = pd.DataFrame(silver_rows)
logger.info("Silver: %d parca indekslendi", len(silver_df))

combined = pd.concat([gold_rows, silver_df], ignore_index=True)
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(INDEX_PATH, index=False)
summary = combined.groupby(["layer", "dataset"])["row_count"].agg(["sum", "count"])
logger.info("Birlesik indeks yazildi: %s\n%s", INDEX_PATH, summary.to_string())

build_columns_catalog(client, index_df=combined)
logger.info("TAMAMLANDI")
