"""adsb_watcher.py — yeni tarihsel .tar dosyalarini izle ve otomatik isle.

data/bronze/adsblol_historical/_input/ klasorune yeni bir .tar dosyasi geldiginde
otomatik olarak Silver parse + Gold unify adimlarini calistirir.

Kullanim:
    python -m src.ingestion.adsb_watcher

Calisirken yeni .tar dosyasi yerlestirilirse (FTP, kopyala-yapistir, vb.) en fazla
POLL_INTERVAL_S saniye icinde otomatik olarak islenir.

MinIO yon: Silver ve Gold dogrudan MinIO'ya yazilir (STORAGE_BACKEND=minio) ya da
yerel mock'a (STORAGE_BACKEND=local). Ortam degiskenleri icin .env dosyasina bakiniz.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

WATCH_DIR = Path("data/bronze/adsblol_historical/_input")
POLL_INTERVAL_S = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run(args: list[str]) -> bool:
    result = subprocess.run(args)
    return result.returncode == 0


def process_tar(tar_path: Path) -> None:
    logger.info("Yeni tar algilandi: %s -- Silver parse basliyor", tar_path)
    ok = _run([
        sys.executable, "-m", "src.silver.parse_adsblol_historical",
        "--local-tar", str(tar_path),
    ])
    if not ok:
        logger.error("Silver parse BASARISIZ: %s", tar_path)
        return
    logger.info("Silver tamamlandi: %s -- Gold unify basliyor", tar_path)
    ok = _run([sys.executable, "-m", "src.gold.unify"])
    if not ok:
        logger.error("Gold unify BASARISIZ (Silver yazildi ama Gold guncellenmedi)")
        return
    logger.info("Gold guncellendi. %s islendi.", tar_path)


def main() -> None:
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    known: set[Path] = set(WATCH_DIR.glob("*.tar"))
    logger.info(
        "Izleme basliyor: %s (%d mevcut tar, her %ds kontrol)",
        WATCH_DIR, len(known), POLL_INTERVAL_S,
    )

    try:
        while True:
            current = set(WATCH_DIR.glob("*.tar"))
            new_files = current - known
            for tar_path in sorted(new_files):
                process_tar(tar_path)
            known = current
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        logger.info("Izleyici durduruldu.")


if __name__ == "__main__":
    main()
