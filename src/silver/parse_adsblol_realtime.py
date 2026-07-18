"""parse_adsblol_realtime.py -- adsb.lol realtime Silver parser.

Reads raw JSONL files from MinIO Bronze (bronze/adsblol_realtime/_landing/)
and writes Silver Parquet to MinIO.

2026-07-18 DUZELTME (onemli): Bu modul eskiden, artik var olmayan
src/ingestion/adsblol_producer.py'nin yayinladigi VARSAYILAN ham adsb.lol
`ac` semasini (hex, alt_baro, gs, baro_rate, dbFlags, flight, r, t...)
bekliyordu. Gercek uretici Dashboard/codes/uav_producer.py ise Kafka'ya
yazmadan ONCE _normalize_common() ile alanlari sadelestirip yeniden
adlandiriyor (icao24, alt, velocity, vertical_rate, track, is_ground,
is_military, callsign, ...) VE birim donusumunu KENDISI yapiyor (alt zaten
metre, velocity/vertical_rate zaten m/s). Eski kod hicbir alani
BULAMIYORDU -- source_id/alt/ground_speed_ms/vertical_rate_ms/
flight_callsign SESSIZCE hep None, is_military/on_ground SESSIZCE hep
False kaliyordu (bkz. proje sohbet gecmisi -- bu, 3 "isim degisikligi
kalintisi" hatasiyla (InfluxDB bucket, Kafka topic, Dashboard image) AYNI
turden 4. bir ornek). Bu tarihten ONCE islenmis realtime Silver/Gold
satirlari (ham Bronze JSONL "yaz-sonra-sil" deseniyle zaten silindigi
icin) GERI KAZANILAMAZ -- duzeltme sadece BUNDAN SONRA toplanan veriyi
etkiler.

Artik gercek JSONL semasi (Dashboard/codes/uav_producer.py + minio_archiver.py):
  icao24, lat, lon, alt (metre), velocity (m/s), track (derece),
  vertical_rate (m/s), is_ground (bool), is_military (bool), callsign,
  category, squawk, emergency, signal_age_sec, source, cycle_id,
  ts (ISO 8601, per-kayit hassas zaman damgasi -- dosya adindaki toplu
  batch zaman damgasindan daha dogru, oncelik ona verilir).

Usage:
    python -m src.silver.parse_adsblol_realtime
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.minio_io import (
    ObjectStoreClient,
    download_raw_bytes,
    get_minio_client,
    write_silver,
)
from src.common.provenance import add_provenance

logger = logging.getLogger(__name__)

SOURCE_TYPE = "adsblol_realtime"
_LANDING_PREFIX = "adsblol_realtime/_landing/"
_TS_RE = re.compile(r"states-(\d{8}T\d{6})")

# 2026-07-09 (kullanici istegi): --loop GUNLUK araliga cekilince ("son basarili
# calisma zamani"ni gormeden gunlerce fark etmeden veri kaybi) riski sorulunca
# eklendi -- her BASARILI (exception firlatmayan) run() cagrisindan SONRA bu
# dosyaya UTC zaman damgasi + kisa sonuc yazilir. Dosyanin GUNCELLENMEMESI
# (mtime bir gunden eski kalmasi) loop'un sessizce durdugunun/crash oldugunun
# veya surekli exception yiyip HICBIR basariya ulasamadiginin isaretidir --
# basarisiz bir run() bu dosyaya YAZMAZ (bkz. asagidaki except bloguı).
_HEARTBEAT_PATH = Path("data/state/silver_realtime_heartbeat.txt")


def _write_heartbeat(detail: str) -> None:
    try:
        _HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _HEARTBEAT_PATH.write_text(f"{ts} {detail}\n", encoding="utf-8")
    except OSError:
        logger.exception("Heartbeat dosyasi yazilamadi (%s) -- bu run'in KENDISI basarili, "
                          "sadece izleme kaydi basarisiz", _HEARTBEAT_PATH)


def _batch_timestamp(object_name: str) -> float | None:
    """Extract Unix-ish epoch from the JSONL file name (states-YYYYMMDDTHHMMSS...).

    Returns None if the name doesn't match the expected pattern.
    """
    m = _TS_RE.search(object_name)
    if not m:
        return None
    from datetime import datetime, timezone
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _record_timestamp(record: dict, fallback: float | None) -> float | None:
    """Kayit-bazinda hassas 'ts' (ISO 8601) alanini tercih eder -- yoksa/
    parse edilemezse dosya-adindan cikarilan kaba batch zaman damgasina
    duser."""
    ts_raw = record.get("ts")
    if ts_raw:
        try:
            return datetime.fromisoformat(ts_raw).timestamp()
        except (TypeError, ValueError):
            pass
    return fallback


def _parse_ac_record(record: dict, batch_ts: float | None) -> dict:
    return {
        "source_type": SOURCE_TYPE,
        "source_id": record.get("icao24"),
        "timestamp_utc": _record_timestamp(record, batch_ts),
        "lat": record.get("lat"),
        "lon": record.get("lon"),
        "alt": record.get("alt"),                       # zaten metre (uav_producer.py SI'ye ceviriyor)
        "on_ground": bool(record.get("is_ground", False)),
        "label": None,
        "ground_speed_ms": record.get("velocity"),       # zaten m/s
        "track_deg": record.get("track"),
        "vertical_rate_ms": record.get("vertical_rate"), # zaten m/s
        "flight_callsign": (record.get("callsign") or "").strip() or None,
        "category": record.get("category"),
        "squawk": record.get("squawk"),
        "emergency": record.get("emergency"),
        "is_military": bool(record.get("is_military", False)),
        "signal_age_sec": record.get("signal_age_sec"),
        "source": record.get("source"),
        "cycle_id": record.get("cycle_id"),
    }


def parse_jsonl_bytes(raw: bytes, object_name: str) -> pd.DataFrame:
    """Parse one JSONL blob (bytes) into a Silver DataFrame."""
    batch_ts = _batch_timestamp(object_name)
    rows = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            rows.append(_parse_ac_record(record, batch_ts))
        except (json.JSONDecodeError, Exception):
            logger.warning("Skipping malformed line in %s", object_name)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _delete_processed(client: ObjectStoreClient, bucket: str, object_names: list[str]) -> None:
    """Bronze'daki islenmis JSONL'leri sil -- bu, iki seyi birden saglar: (1) 7 gunluk
    lifecycle kuralinin TEK guvenlik agi olmasini onler (islenen veri hemen silinir,
    henuz islenmemis veri hala 7 gun payina sahiptir), (2) run() tekrar cagrildiginda
    AYNI dosyalarin ikinci kez islenip Silver'da kopya satir uretmesini engeller (bkz.
    proje sohbet gecmisi -- eskiden bu silme YOKTU, run() periyodik calistirilirsa
    veri cogalirdi)."""
    for name in object_names:
        try:
            client.remove_object(bucket, name)
        except Exception:
            logger.warning("Islenen Bronze objesi silinemedi (bir sonraki calismada "
                            "tekrar denenecek, kopya riski var): %s", name)


def run(
    bronze_prefix: str = _LANDING_PREFIX,
    *,
    client: ObjectStoreClient | None = None,
    bronze_bucket: str | None = None,
) -> list[str]:
    """Bronze'daki TUM JSONL landing dosyalarini okuyup TEK bir Silver Parquet
    objesinde birlestirir (Returns: bos liste veya tek elemanli s3:// URI listesi).

    ONEMLI (kucuk-dosya sorunu, bkz. proje sohbet gecmisi): eskiden HER JSONL dosyasi
    icin AYRI bir write_silver() cagrisi yapiliyordu -- minio_archiver.py dakikada
    bir kucuk JSONL yazdigi icin bu, Silver'da da ayni "binlerce kucuk dosya" sorununu
    yaratiyordu. Artik tum dosyalarin satirlari TEK DataFrame'de birlestirilip TEK
    parquet olarak yaziliyor -- provenance (_source_file) yine SATIR BAZINDA dogru
    kalir, cunku add_provenance() her dosyanin DataFrame'ine BIRLESTIRMEDEN ONCE
    uygulaniyor."""
    client = client or get_minio_client()
    bronze_bucket = bronze_bucket or os.getenv("MINIO_BRONZE_BUCKET", "bronze")

    jsonl_objects = [
        obj.object_name
        for obj in client.list_objects(bronze_bucket, prefix=bronze_prefix, recursive=True)
        if obj.object_name.endswith(".jsonl")
    ]

    if not jsonl_objects:
        logger.warning("No .jsonl objects found under %s/%s", bronze_bucket, bronze_prefix)
        return []

    logger.info("Found %d JSONL file(s) to parse", len(jsonl_objects))
    frames: list[pd.DataFrame] = []
    processed_objects: list[str] = []

    for obj_name in sorted(jsonl_objects):
        raw = download_raw_bytes(client, obj_name, bucket=bronze_bucket)
        df = parse_jsonl_bytes(raw, obj_name)
        if df.empty:
            logger.warning("No rows parsed from %s", obj_name)
            processed_objects.append(obj_name)  # gecerli satir yok, tekrar denemenin faydasi yok
            continue
        df = add_provenance(
            df, source_type=SOURCE_TYPE, source_file=obj_name, schema_version="silver_v1"
        )
        frames.append(df)
        processed_objects.append(obj_name)

    if not frames:
        logger.warning("All JSONL files were empty; nothing to write")
        _delete_processed(client, bronze_bucket, processed_objects)
        return []

    combined = pd.concat(frames, ignore_index=True)
    # ONEMLI: silme, write_silver basariyla BITTIKTEN SONRA yapiliyor -- yazma
    # sirasinda bir hata olursa (exception) asagidaki satirlara hic ulasilmaz,
    # Bronze dosyalari YERINDE kalir (veri kaybi degil, bir sonraki calismada
    # tekrar denenir).
    uri = write_silver(combined, SOURCE_TYPE, client=client)
    logger.info("Parsed %d JSONL file(s) -> %d rows -> %s",
                len(jsonl_objects), len(combined), uri)

    _delete_processed(client, bronze_bucket, processed_objects)
    logger.info("Deleted %d processed Bronze JSONL file(s)", len(processed_objects))

    return [uri]


if __name__ == "__main__":
    import argparse
    import time

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="adsb.lol realtime JSONL → Silver Parquet")
    parser.add_argument(
        "--bronze-prefix", default=_LANDING_PREFIX, help="MinIO Bronze prefix for JSONL files"
    )
    # ONEMLI: MinIO'da artik bir silme/lifecycle kurali YOK (2026-07-09 karari,
    # bkz. Dashboard/codes/minio_archiver.py modul docstring'i) -- Bronze'daki realtime
    # landing verisi silinmiyor, sadece bu script calisip Silver'a "islemedigi"
    # surece BIRIKIYOR (zararsiz, sadece disk kullanir). --loop bu script'i
    # minio_archiver.py'nin kendi dongusuyle AYNI desende (sonsuz dongu + sabit
    # bekleme) surekli calisir hale getiriyor -- elle hatirlamaya bagimli
    # olmadan Bronze duzenli araliklarla Silver'a bosaltiliyor.
    parser.add_argument(
        "--loop", action="store_true",
        help="Tek seferlik calismak yerine --interval saniyede bir surekli calis",
    )
    parser.add_argument(
        "--interval", type=int, default=int(os.environ.get("SILVER_INTERVAL", "86400")),
        help="--loop ile birlikte iki calisma arasi SABIT bekleme (sn). --daily-at "
             "verilmisse YOK SAYILIR. Varsayilan: 86400 (1 gun) -- ortam degiskeni: "
             "SILVER_INTERVAL",
    )
    # 2026-07-09 (kullanici durumu): PC sadece mesai saatlerinde (orn. 08-18)
    # ACIK -- sabit --interval (86400sn), PC HER GUN KAPANIP ACILDIGI icin
    # ISE YARAMAZ: sayac PC kapaliyken donuyor, ertesi gun giriste kaldigi
    # yerden DEGIL sifirdan baslar, "her gun saat 17:00'de calis" gibi bir
    # SAAT-BAZLI garanti VEREMEZ. --daily-at bunun yerine HER GUN belirli bir
    # SAATTE (yerel saat) calisir -- giris/baslangicta HEMEN bir kez (birikmis
    # varsa yakalamak icin), sonra o gunun (gecmisse ertesi gunun) hedef
    # saatine kadar uyur. PC 18:00'de kapanip ertesi 08:00'de acilirsa, o gun
    # 17:00'deki calisma zaten TAMAMLANMIS olur (17:00 < 18:00 kapanma), yeni
    # gun basinda bir catch-up + yeni 17:00 hedefi kurulur.
    parser.add_argument(
        "--daily-at", default=None, metavar="HH:MM",
        help="Her gun bu YEREL saatte calis (orn. 17:00) -- --interval'i gecersiz "
             "kilar, --loop'u zimnen acar. Baslangicta HEMEN bir catch-up calismasi "
             "da yapilir.",
    )
    args = parser.parse_args()

    def _seconds_until(hhmm: str) -> float:
        from datetime import datetime, timedelta
        hour, minute = map(int, hhmm.split(":"))
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    def _run_once_with_heartbeat() -> None:
        try:
            uris = run(args.bronze_prefix)
            _write_heartbeat(f"basarili -- {len(uris)} parca yazildi")
        except Exception:
            # ONEMLI: uav_producer.py/minio_archiver.py ile AYNI ilke -- gecici
            # bir hata (orn. MinIO o an erisilemez) TUM dongunun crash olmasina
            # sebep olmamali, bir sonraki denemede devam edilmeli. Heartbeat
            # BILEREK yazilmiyor -- dosyanin "son basarili" anlami korunuyor,
            # sessiz basarisizlik dosyanin ESKI kalmasiyla fark edilir.
            logger.exception("Calisma sirasinda hata -- bir sonraki denemede devam edilecek")

    if args.daily_at:
        logger.info("Gunluk-saat modu: hemen bir catch-up, sonra her gun %s'de (durdurmak icin Ctrl+C)", args.daily_at)
        _run_once_with_heartbeat()
        while True:
            wait_s = _seconds_until(args.daily_at)
            logger.info("Bir sonraki calisma: %.0f saniye sonra (hedef %s)", wait_s, args.daily_at)
            time.sleep(wait_s)
            _run_once_with_heartbeat()
    elif not args.loop:
        uris = run(args.bronze_prefix)
        _write_heartbeat(f"tek seferlik -- {len(uris)} parca yazildi")
    else:
        logger.info("Loop modu: her %ds bir calisacak (durdurmak icin Ctrl+C)", args.interval)
        while True:
            _run_once_with_heartbeat()
            time.sleep(args.interval)
