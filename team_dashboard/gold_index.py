"""gold_index.py -- Gold export dashboard'un tarih-bazli sorgusunu HIZLI
yapabilmesi icin, her Gold parcasinin (part-*.parquet) timestamp_utc
min/max degerlerini onceden cikarip yerel bir indekste tutar.

NEDEN GEREKLI: Gold `unified/` altinda 6800+ parca / 2.76 milyar satir var,
parca adlari (part-<zaman damgasi>-<uuid>.parquet) VERI icerigine gore degil
YAZILMA sirasina gore verilmis -- yani "hangi parcada 2026-01-05 verisi var"
sorusu, indeks olmadan SADECE tum parcalari okuyup bakarak cevaplanabilir
(saatler surer). Bu modul TEK SEFERLIK bir tarama yapip her parcanin
timestamp_utc araligini kucuk bir yerel dosyada saklar; sonraki her export
istegi, ilgisiz parcalari (istenen tarih araligiyla KESISMEYENLERI) bu
indekse bakarak ATLAR -- sadece gercekten gerekli parcalar indirilip okunur.

Salt okunur: src/common/minio_io.py disinda ortak pipeline'a dokunmaz
(individual/metehan_geo/data.py'deki ayni "read-only disiplini" ile tutarli).
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import pandas as pd

from src.common.minio_io import ObjectStoreClient, get_minio_client, list_layer_objects

logger = logging.getLogger(__name__)

INDEX_PATH = Path("data/state/gold_export_index.parquet")

# ALFA/UAV Attack'in timestamp_utc'si GERCEK takvim tarihi DEGIL (senaryo/
# oturum-ici goreli sayac, epoch 0'a yakin veya anlamsiz uzak-gelecek
# degerler uretebiliyor -- ilk indekslemede 1970-01-01..2033-08-19 gibi
# saçma bir genel aralik cikmasiyla fark edildi). Bu export araci SADECE
# gercek takvim tarihi olan ADS-B kaynaklarini indeksler/sunar -- ayni
# ayrim individual/metehan_geo/data.py::ADSB_SOURCE_TYPES'ta da var.
ADSB_SOURCE_TYPES = frozenset({"adsblol_historical", "adsblol_hist", "adsblol_realtime", "adsblol_rt"})


def build_index(
    client: ObjectStoreClient | None = None,
    *,
    gold_bucket: str | None = None,
    index_path: Path = INDEX_PATH,
) -> pd.DataFrame:
    """Her Gold parcasi icin (object_name, source_type, min_ts, max_ts,
    row_count) cikarir -- SADECE ADSB_SOURCE_TYPES'a ait parcalar (gercek
    takvim tarihi tasiyanlar) tutulur, ALFA/UAV Attack gibi sentetik-zamanli
    kaynaklar indekslenmez (bkz. modul basi not).

    Her parcadan SADECE `timestamp_utc` + `source_type` kolonlari okunur
    (pyarrow'un kolon-secici okumasi ile) -- 10+ kolonun tamamini her parca
    icin belleğe almaktan cok daha hafif.
    """
    client = client or get_minio_client()
    bucket = gold_bucket or os.getenv("MINIO_GOLD_BUCKET", "gold")

    object_names = list_layer_objects(client, bucket, "unified")
    logger.info("Gold export indeksi: %d parca taranacak", len(object_names))

    rows = []
    for i, name in enumerate(object_names):
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

        # Gold parcalari stream_unify() tarafindan HER ZAMAN tek bir
        # source_type icin uretiliyor (bkz. teknik rapor) -- ama yine de
        # guvenli tarafta kalmak icin cogunluk degerine bakiyoruz.
        source_type = part["source_type"].mode().iat[0] if not part["source_type"].empty else None
        if source_type not in ADSB_SOURCE_TYPES:
            continue   # ALFA/UAV Attack/vb. -- gercek tarih tasimiyor, indekslenmez

        ts = part["timestamp_utc"].dropna()
        if ts.empty:
            continue
        rows.append({
            "object_name": name,
            "source_type": source_type,
            "min_ts": float(ts.min()),
            "max_ts": float(ts.max()),
            "row_count": int(len(ts)),
        })

        if (i + 1) % 500 == 0:
            logger.info("  %d/%d parca indekslendi", i + 1, len(object_names))

    index_df = pd.DataFrame(rows)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(index_path, index=False)
    logger.info("Indeks yazildi: %s (%d parca, toplam %d satir)",
                index_path, len(index_df), int(index_df["row_count"].sum()))
    return index_df


def load_index(index_path: Path = INDEX_PATH) -> pd.DataFrame:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Export indeksi bulunamadi: {index_path} -- once "
            "`python -m team_dashboard.gold_index` calistirilmali."
        )
    return pd.read_parquet(index_path)


def parts_overlapping(index_df: pd.DataFrame, start_ts: float, end_ts: float) -> pd.DataFrame:
    """Istenen [start_ts, end_ts) araligiyla KESISEN parcalari dondurur.

    Bir parca, KENDI [min_ts, max_ts] araligi istenen aralikla hic
    kesismiyorsa (tamamen once ya da tamamen sonra) elenir -- bu, her
    export isteginin sadece ilgili parcalari indirmesini saglayan asil
    optimizasyon.
    """
    valid = index_df.dropna(subset=["min_ts", "max_ts"])
    overlap = valid[(valid["max_ts"] >= start_ts) & (valid["min_ts"] < end_ts)]
    return overlap


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_index()
