"""layer_index.py -- export dashboard'un Silver VE Gold katmanlarini,
katman icindeki her "dataset" (source_type) icin AYRI AYRI, tarih-bazli
hizli sorgulanabilir hale getiren indeks.

NEDEN GEREKLI: Kullanici artik once KATMAN (Silver/Gold), sonra o katmandaki
DATASET'i (adsblol_historical, alfa, uav_attack, ...) seciyor -- her ikisinin
KENDI kolon semasi ve KENDI tarih araligi var. Bu modul, gold_index.py'nin
(tek katman/tek dataset varsayan ilk versiyon) yerine gecti; ayni "sadece
timestamp_utc oku, parca-parca tara" optimizasyonunu Silver'in 5 ayri
source_type prefix'ine de uyguluyor.

Salt okunur: src/common/minio_io.py disinda ortak pipeline'a dokunmaz.
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path

import pandas as pd

from src.common.minio_io import (
    ObjectStoreClient,
    get_minio_client,
    list_layer_objects,
    read_parquet_object,
)
from src.gold.unify import GOLD_COLUMNS

logger = logging.getLogger(__name__)

INDEX_PATH = Path("data/state/export_layer_index.parquet")
COLUMNS_CATALOG_PATH = Path("data/state/export_columns_catalog.json")

# Silver'da her biri KENDI MinIO prefix'i olan, bilinen kaynak turleri
# (src/gold/unify.py::COLUMN_MAPS'in gercek/aktif anahtarlari -- "_hist"/"_rt"
# gibi aliaslar haric, cunku onlar ayri veri tasimiyor, sadece test-uyumlulugu
# icin COLUMN_MAPS'te var).
SILVER_DATASETS = ["adsblol_historical", "adsblol_realtime", "alfa", "uav_attack", "uav_sead"]

# adsb.lol disi kaynaklarin timestamp_utc'si GERCEK takvim tarihi degil
# (senaryo/oturum-ici goreli sayac -- ilk Gold indekslemesinde 1970..2033
# gibi anlamsiz bir aralik cikmasiyla kesif edildi). Tarih filtresi bu
# kaynaklarda da UYGULANABILIR (kod calisir) ama sonuc gercek bir takvim
# araligi degildir -- arayuzde her dataset'in KENDI (dogru ya da yaniltici)
# min/max'i ayri ayri gosterilir, boylece kullanici gorup karar verebilir.
REAL_CALENDAR_DATASETS = frozenset({"adsblol_historical", "adsblol_realtime"})


def _iter_part_stats(client: ObjectStoreClient, bucket: str, object_names: list[str]):
    for name in object_names:
        response = client.get_object(bucket, name)
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()
        try:
            part = pd.read_parquet(io.BytesIO(raw), columns=["timestamp_utc", "source_type"])
        except Exception:
            logger.warning("Parca okunamadi, atlaniyor: %s", name, exc_info=True)
            continue
        yield name, part


def build_index(client: ObjectStoreClient | None = None, *, index_path: Path = INDEX_PATH) -> pd.DataFrame:
    """Silver'in 5 dataset'i + Gold'un 'unified' prefix'i (dataset basina
    source_type ile ayristirilir) icin (layer, dataset, object_name, min_ts,
    max_ts, row_count) satirlari uretir."""
    client = client or get_minio_client()
    silver_bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")
    gold_bucket = os.getenv("MINIO_GOLD_BUCKET", "gold")

    rows: list[dict] = []

    for dataset in SILVER_DATASETS:
        object_names = list_layer_objects(client, silver_bucket, dataset)
        if not object_names:
            logger.info("Silver/%s: hic parca yok, atlaniyor", dataset)
            continue
        logger.info("Silver/%s: %d parca taranacak", dataset, len(object_names))
        for i, (name, part) in enumerate(_iter_part_stats(client, silver_bucket, object_names)):
            ts = part["timestamp_utc"].dropna()
            if ts.empty:
                continue
            rows.append({
                "layer": "silver", "dataset": dataset, "object_name": name,
                "min_ts": float(ts.min()), "max_ts": float(ts.max()), "row_count": int(len(ts)),
            })
            if (i + 1) % 500 == 0:
                logger.info("  Silver/%s: %d/%d parca indekslendi", dataset, i + 1, len(object_names))

    gold_object_names = list_layer_objects(client, gold_bucket, "unified")
    logger.info("Gold/unified: %d parca taranacak (dataset = source_type)", len(gold_object_names))
    for i, (name, part) in enumerate(_iter_part_stats(client, gold_bucket, gold_object_names)):
        source_type = part["source_type"].mode().iat[0] if not part["source_type"].empty else None
        ts = part["timestamp_utc"].dropna()
        if ts.empty or source_type is None:
            continue
        rows.append({
            "layer": "gold", "dataset": source_type, "object_name": name,
            "min_ts": float(ts.min()), "max_ts": float(ts.max()), "row_count": int(len(ts)),
        })
        if (i + 1) % 500 == 0:
            logger.info("  Gold: %d/%d parca indekslendi", i + 1, len(gold_object_names))

    index_df = pd.DataFrame(rows)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(index_path, index=False)

    summary = index_df.groupby(["layer", "dataset"])["row_count"].agg(["sum", "count"])
    logger.info("Indeks yazildi: %s\n%s", index_path, summary.to_string())
    return index_df


def build_columns_catalog(
    client: ObjectStoreClient | None = None,
    *,
    index_df: pd.DataFrame | None = None,
    catalog_path: Path = COLUMNS_CATALOG_PATH,
) -> dict:
    """Her (layer, dataset) icin GERCEKTEN mevcut kolon listesini cikarir.

    Gold zaten ortak 7+3(+is_military) semaya hizalanmis oldugundan tum
    Gold dataset'leri AYNI kolon listesini paylasir (GOLD_COLUMNS) -- ayrica
    okumaya gerek yok. Silver'da ise HER dataset FARKLI, kaynak-ozgu kolonlar
    tasir (ornek: alfa'da yok ama adsblol_historical'da olan `squawk`,
    `roll_deg`, `is_military`...) -- bunlar icin bir orneklem parca okunup
    kolon adlari cikarilir.
    """
    client = client or get_minio_client()
    silver_bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")
    index_df = index_df if index_df is not None else load_index()

    catalog: dict[str, dict[str, list[str]]] = {"silver": {}, "gold": {}}

    for dataset in SILVER_DATASETS:
        subset = index_df[(index_df["layer"] == "silver") & (index_df["dataset"] == dataset)]
        if subset.empty:
            continue
        sample_name = subset.iloc[0]["object_name"]
        sample_df = read_parquet_object(client, silver_bucket, sample_name)
        catalog["silver"][dataset] = list(sample_df.columns)

    for dataset in sorted(index_df.loc[index_df["layer"] == "gold", "dataset"].unique()):
        catalog["gold"][dataset] = list(GOLD_COLUMNS)

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    logger.info("Kolon katalogu yazildi: %s", catalog_path)
    return catalog


def load_index(index_path: Path = INDEX_PATH) -> pd.DataFrame:
    if not index_path.exists():
        raise FileNotFoundError(f"Export indeksi bulunamadi: {index_path} -- once build_index() calistirilmali.")
    return pd.read_parquet(index_path)


def load_columns_catalog(catalog_path: Path = COLUMNS_CATALOG_PATH) -> dict:
    if not catalog_path.exists():
        raise FileNotFoundError(f"Kolon katalogu bulunamadi: {catalog_path} -- once build_columns_catalog() calistirilmali.")
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def parts_overlapping(index_df: pd.DataFrame, layer: str, dataset: str, start_ts: float, end_ts: float) -> pd.DataFrame:
    """Istenen (layer, dataset) ikilisinde, [start_ts, end_ts) araligiyla
    KESISEN parcalari dondurur -- ilgisiz her seyi (baska katman/dataset,
    veya tarih araligi disi parcalar) eleyerek sadece gercekten okunmasi
    gereken parcalara iner."""
    subset = index_df[(index_df["layer"] == layer) & (index_df["dataset"] == dataset)]
    subset = subset.dropna(subset=["min_ts", "max_ts"])
    return subset[(subset["max_ts"] >= start_ts) & (subset["min_ts"] < end_ts)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    idx = build_index()
    build_columns_catalog(index_df=idx)
