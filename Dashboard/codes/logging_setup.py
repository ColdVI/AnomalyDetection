"""Dashboard servislerinin (uav_producer, dashboard_consumer, minio_archiver,
app) print() ciktisini, mevcut terminal/`docker logs` akisini BOZMADAN, ayni
zamanda Dashboard/logs/ altindaki kendi dosyasina da yazan kucuk yardimci.

ONEMLI: varsayilan konum CALISMA DIZININE (cwd) GORE DEGIL, bu dosyanin
KENDI konumuna gore hesaplaniyor (dirname(dirname(__file__))/logs -- yani
Dashboard/codes/'in bir ustu, Dashboard/logs/). Once cwd'ye gore bagli
"logs" (relative) kullanilmisti; script'in NEREDEN calistirildigina gore
(Docker WORKDIR /app, yoksa baska bir dizin) farkli/beklenmeyen bir yerde
logs/ olusuyordu. Bu artik cagrildigi yerden BAGIMSIZ, hep Dashboard/logs/'e
(container icinde /app/logs/) yazar -- docker-compose.yml'deki
"./Dashboard/logs:/app/logs" bind-mount'u bunu host'taki Dashboard/logs/'e
esler."""

from __future__ import annotations

import sys
from pathlib import Path

_DEFAULT_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

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


def enable_file_logging(service_name: str, logs_dir: str | Path | None = None) -> None:
    """stdout ve stderr'i, mevcut davranisi (terminal/`docker logs`) koruyarak
    logs_dir/<service_name>.log dosyasina da yazar. logs_dir verilmezse
    Dashboard/logs/ kullanilir (bkz. modul docstring'i). Her servis kendi
    __main__ blogunun EN BASINDA, ilk print()'ten once cagirmali."""
    directory = Path(logs_dir) if logs_dir is not None else _DEFAULT_LOGS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"{service_name}.log"

    if log_path.exists() and log_path.stat().st_size > _MAX_BYTES:
        backup = directory / f"{service_name}.log.1"
        log_path.replace(backup)

    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
