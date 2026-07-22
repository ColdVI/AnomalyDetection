"""bronze_watcher.py -- MinIO Bronze'daki yeni ham veriyi otomatik Silver'a cevirir.

adsb_watcher.py'den FARKLI: o YEREL dosya sistemini (yeni .tar dosyasi) izliyor;
bu script dogrudan MinIO Bronze'u izliyor -- "make bronze-alfa" / "make bronze-attack"
/ "make bronze-uav-sead" ile bir kez ya da ara sira yuklenen kaynaklar icin.

Bronze'daki nesne listesi (isim + boyut) bir onceki kontrolden BERI degistiyse
(yeni dosya eklendi VEYA ayni isimli dosya farkli icerikle degistirildi), o
kaynagin Silver parser'ini calistirir (parse_alfa.py/parse_uav_attack.py/
parse_uav_sead.py -- her biri artik kendi ESKI Silver ciktisini yazmadan once
temizliyor, bkz. o dosyalardaki "coklanma" yorumu), ardindan Gold unify'i
calistirir. Bronze'a ASLA yazmiyor/silmiyor -- sadece OKUYUP tetikliyor (ham
veri Bronze'da kalici kalir, bu katmanin butun amaci bu).

Ilk calistirmada: Bronze'da veri olup Silver'da hic karsiligi yoksa (daha once
hic islenmemis kaynak) hemen bir kerelik isleme baslatir -- boylece watcher'i
gec baslatsan bile birikmis veri kacmaz.

Kullanim:
    python -m src.ingestion.bronze_watcher [--interval 900]

Ortam degiskeni (Docker/servis olarak calistirirken):
    BRONZE_WATCH_INTERVAL  (varsayilan: 900sn / 15dk -- bu kaynaklar realtime
    adsb.lol kadar sik degismiyor, elle/ara sira yukleniyor)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

from src.common.minio_io import get_minio_client, list_layer_objects

logger = logging.getLogger(__name__)

# source_type: (bronze_prefix, silver_module)
WATCHED_SOURCES: dict[str, tuple[str, str]] = {
    "alfa": ("alfa/", "src.silver.parse_alfa"),
    "uav_attack": ("uav_attack/", "src.silver.parse_uav_attack"),
    "uav_sead": ("uav_sead/", "src.silver.parse_uav_sead"),
}

DEFAULT_INTERVAL_S = 900


def _bronze_snapshot(client, bucket: str, prefix: str) -> frozenset:
    """(object_name, size) ciftlerinin degismez kumesi -- sadece yeni dosya
    eklenmesini degil, AYNI isimli dosyanin farkli icerikle degistirilmesini
    (boyut degisir) de yakalar; salt isim karsilastirmak bunu kacirirdi."""
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    return frozenset((obj.object_name, getattr(obj, "size", None)) for obj in objects)


def _run(args: list[str]) -> bool:
    result = subprocess.run(args)
    return result.returncode == 0


def process_source(source_type: str, silver_module: str) -> None:
    logger.info("%s: Bronze degisikligi algilandi -- Silver parse basliyor", source_type)
    ok = _run([sys.executable, "-m", silver_module])
    if not ok:
        logger.error("%s: Silver parse BASARISIZ", source_type)
        return
    logger.info("%s: Silver tamamlandi -- Gold unify basliyor", source_type)
    ok = _run([sys.executable, "-m", "src.gold.unify"])
    if not ok:
        logger.error("Gold unify BASARISIZ (Silver yazildi ama Gold guncellenmedi)")
        return
    logger.info("%s: Gold guncellendi.", source_type)


def main() -> None:
    interval = int(os.environ.get("BRONZE_WATCH_INTERVAL", str(DEFAULT_INTERVAL_S)))
    client = get_minio_client()
    bronze_bucket = os.getenv("MINIO_BRONZE_BUCKET", "bronze")
    silver_bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")

    logger.info("Izleme basliyor: %s (her %ds kontrol)", list(WATCHED_SOURCES), interval)

    known: dict[str, frozenset] = {}
    for source_type, (prefix, silver_module) in WATCHED_SOURCES.items():
        current = _bronze_snapshot(client, bronze_bucket, prefix)
        has_silver = bool(list_layer_objects(client, silver_bucket, source_type))
        if current and not has_silver:
            logger.info("%s: Bronze'da veri var ama Silver bos -- ilk isleme baslatiliyor",
                        source_type)
            process_source(source_type, silver_module)
        known[source_type] = current

    try:
        while True:
            time.sleep(interval)
            for source_type, (prefix, silver_module) in WATCHED_SOURCES.items():
                current = _bronze_snapshot(client, bronze_bucket, prefix)
                if current != known[source_type]:
                    process_source(source_type, silver_module)
                    known[source_type] = current
    except KeyboardInterrupt:
        logger.info("Izleyici durduruldu.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
