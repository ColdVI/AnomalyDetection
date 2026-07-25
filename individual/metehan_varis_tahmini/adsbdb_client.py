"""adsbdb.com callsign -> rota (origin/destination havalimani) istemcisi.

Statik bir veritabani (gercek-zamanli DEGIL) -- gecmis (2017-2026) callsign'ler
icin de calisir. Toplu/batch endpoint yok, callsign basina tek GET istegi --
bu yuzden KALICI CACHE sart (ayni callsign'i iki kez sorgulamamak icin).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# Paralel shard'lar AYNI dosyaya ayni anda yazip birbirini ezmesin diye her
# shard kendi cache dosyasini kullanir (ADSBDB_CACHE_PATH env degiskeni ile).
CACHE_PATH = Path(os.environ.get("ADSBDB_CACHE_PATH", str(Path(__file__).parent / "adsbdb_cache.json")))
API_URL = "https://api.adsbdb.com/v0/callsign/{}"
TIMEOUT_SEC = 6
REQUEST_DELAY_SEC = 0.02  # adsbdb'yi yormamak icin istekler arasi kucuk bekleme

_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if CACHE_PATH.exists():
        _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    else:
        _cache = {}
    return _cache


def save_cache() -> None:
    if _cache is not None:
        CACHE_PATH.write_text(json.dumps(_cache), encoding="utf-8")


def lookup_route(callsign: str) -> dict | None:
    """Callsign icin {origin_icao, origin_lat, origin_lon, dest_icao,
    dest_lat, dest_lon} dondurur, bulunamazsa None. Sonuc (bulunsun/bulunmasin)
    cache'e yazilir."""
    cache = _load_cache()
    if callsign in cache:
        return cache[callsign]

    result = None
    status_code = None
    try:
        r = requests.get(API_URL.format(callsign), timeout=TIMEOUT_SEC)
        status_code = r.status_code
        if status_code == 200:
            data = r.json().get("response", {}).get("flightroute")
            if data and data.get("origin") and data.get("destination"):
                o, d = data["origin"], data["destination"]
                result = {
                    "origin_icao": o.get("icao_code"),
                    "origin_lat": o.get("latitude"),
                    "origin_lon": o.get("longitude"),
                    "dest_icao": d.get("icao_code"),
                    "dest_lat": d.get("latitude"),
                    "dest_lon": d.get("longitude"),
                }
    except requests.RequestException:
        status_code = None  # ag hatasi -- cache'e YAZMA, tekrar denensin (gecici olabilir)

    if result is not None or status_code == 404:
        cache[callsign] = result
    time.sleep(REQUEST_DELAY_SEC)
    return result


DEFAULT_MAX_WORKERS = int(os.environ.get("ADSBDB_MAX_WORKERS", "40"))


def lookup_routes_bulk(callsigns: list[str], *, max_workers: int = DEFAULT_MAX_WORKERS) -> dict[str, dict | None]:
    """Birden fazla FARKLI callsign'i es zamanli sorgular (I/O-bound aglar
    icin thread havuzu -- adsbdb'yi asiri yormamak icin makul (8) bir
    concurrency sinirinda tutulur). Zaten cache'de olanlar hic ag istegi
    yapmadan dondurulur."""
    cache = _load_cache()
    unique = sorted(set(c for c in callsigns if c))
    to_fetch = [c for c in unique if c not in cache]
    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(lookup_route, to_fetch))
    return {c: cache.get(c) for c in unique}
