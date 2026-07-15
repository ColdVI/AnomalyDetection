"""Dashboard servislerinin (uav_producer, dashboard_consumer, minio_archiver,
app) print() ciktisini, mevcut terminal/`docker logs` akisini BOZMADAN, ayni
zamanda logs/ altindaki kendi dosyasina da yazan kucuk yardimci.

Servisler artik Docker-only calisiyor (native Windows kurulumu kaldirildi),
bu yuzden yol her zaman calisma dizinine (docker-compose.yml'de WORKDIR
/app) gore cozuluyor -- "./logs:/app/logs" bind-mount'u sayesinde bu,
host'taki repo kokundeki logs/ klasorune karsilik gelir."""

from __future__ import annotations

import sys
from pathlib import Path

_MAX_BYTES = 20 * 1024 * 1024  # 20MB -- asilirsa servis yeniden baslarken
                                 # eski dosya .log.1'e tasinir (gercek zamanli
                                 # dondurme degil, ama sinirsiz buyumeyi engeller)


class _Tee:
    """Yazilan her seyi birden fazla akisa (orn. gercek stdout + log dosyasi) aynı anda gonderir."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def enable_file_logging(service_name: str, logs_dir: str = "logs") -> None:
    """stdout ve stderr'i, mevcut davranisi (terminal/`docker logs`) koruyarak
    logs_dir/<service_name>.log dosyasina da yazar. Her servis kendi
    __main__ blogunun EN BASINDA, ilk print()'ten once cagirmali."""
    directory = Path(logs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"{service_name}.log"

    if log_path.exists() and log_path.stat().st_size > _MAX_BYTES:
        backup = directory / f"{service_name}.log.1"
        log_path.replace(backup)

    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
