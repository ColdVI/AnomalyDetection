"""realtime_density.py -- --live/--24h/--7d yogunluk haritasi.

docs/realtime_pipeline_prompt.md'de tanimlanan 3 yeni mod. Historical'in
(build_flight_density.py) tersine burada TEK sorgu + TEK DataFrame yeterli --
InfluxDB penceresi 7 gunle sinirli (binler-milyonlar mertebesi, milyar
degil), chunking/streaming gerekmiyor.

KRITIK: flight_count AYNI TANIMLA hesaplaniyor (benzersiz source_id per hex) --
historical ve realtime arasinda tutarli bir "yogunluk" metrigi (bkz.
docs/realtime_pipeline_prompt.md "Historical taraftan tasinmasi gereken
kritik ogrenimler", madde 1).

day_count buraya TASINMADI (plan madde: "opsiyonel, dusuk oncelik") --
--live/--24h/--7d pencereleri zaten kisa, "kac farkli gun" ayrimi bu
olcekte anlamli degil.

MLAT/ads_source_type katmani de burada YOK -- Kafka semasinda (KAFKA_SCHEMA.md)
per-nokta kaynak tipi tasinmiyor (sadece "source"="adsblol"/"opensky", hangi
SISTEM, historical'daki mlat/adsb_icao/tisb ayrimi degil). Bu bilinen bir
kapsam siniri (2026-07-07 karari).
"""

from __future__ import annotations

import logging
from pathlib import Path

from individual.metehan_geo.data import clean_coordinates
from individual.metehan_geo.geo import assign_h3_cell
from individual.metehan_geo.influx_client import load_realtime_window
from individual.metehan_geo.viz import build_density_geojson, save_geojson

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "viz" / "data"

RANGE_PRESETS = {
    # "-2m" degil "-3m": uav-producer'in GERCEK dongu suresi hedeflenen 15sn
    # degil, adsb.lol fetch'i yuzunden ~130-143sn (bkz. uav-producer
    # loglari, 2026-07-07 dogrulamasi). Bir onceki burst 120sn'den once
    # "-2m" penceresinden dusuyor, yeni burst henuz gelmemis oluyordu --
    # sonuc: "live" modu periyodik olarak SIFIR veri donduruyordu (gercek bir
    # kesinti degil, pencere/dongu suresi uyumsuzlugu). "-3m" (180sn),
    # gozlemlenen en kotu durumu (143sn) rahat kapsiyor.
    "live": "-3m",
    "24h": "-24h",
    "7d": "-7d",
}


def build_realtime_density(mode: str, h3_resolution: int = 5) -> dict:
    if mode not in RANGE_PRESETS:
        raise ValueError(f"Bilinmeyen mod: {mode!r} (secenekler: {list(RANGE_PRESETS)})")

    df = load_realtime_window(RANGE_PRESETS[mode])
    if df.empty:
        logger.warning("realtime/%s: veri yok, bos GeoJSON donduruluyor", mode)
        return {"type": "FeatureCollection", "features": []}

    cleaned = clean_coordinates(df)
    if cleaned.empty:
        logger.warning("realtime/%s: clean_coordinates sonrasi veri kalmadi", mode)
        return {"type": "FeatureCollection", "features": []}

    chunk = assign_h3_cell(cleaned, h3_resolution)
    if "is_military" not in chunk.columns:
        chunk = chunk.assign(is_military=False)

    grouped = chunk.groupby("h3_cell").agg(
        point_count=("h3_cell", "size"),
        flight_count=("source_id", "nunique"),
    ).reset_index()
    # 2026-07-10 (kullanici istegi): historical (build_flight_density.py) ile
    # AYNI ayrim -- sivil/askeri benzersiz ucus sayisi ayrica hesaplanip
    # frontend'e tasiniyor, flight_count (toplam) DEGISMIYOR.
    civil_counts = (
        chunk[~chunk["is_military"]].groupby("h3_cell")["source_id"].nunique()
    )
    military_counts = (
        chunk[chunk["is_military"]].groupby("h3_cell")["source_id"].nunique()
    )
    grouped["flight_count_civil"] = grouped["h3_cell"].map(civil_counts).fillna(0).astype(int)
    grouped["flight_count_military"] = grouped["h3_cell"].map(military_counts).fillna(0).astype(int)

    geojson_input = grouped[["h3_cell", "flight_count"]].rename(columns={"flight_count": "point_count"})
    geojson = build_density_geojson(geojson_input)
    point_count_by_hex = dict(zip(grouped["h3_cell"], grouped["point_count"]))
    civil_by_hex = dict(zip(grouped["h3_cell"], grouped["flight_count_civil"]))
    military_by_hex = dict(zip(grouped["h3_cell"], grouped["flight_count_military"]))
    for feature in geojson["features"]:
        h = feature["properties"]["h3_cell"]
        feature["properties"]["point_count_raw"] = int(point_count_by_hex[h])
        feature["properties"]["flight_count_civil"] = int(civil_by_hex[h])
        feature["properties"]["flight_count_military"] = int(military_by_hex[h])

    logger.info(
        "realtime/%s: %d satir, %d benzersiz ucak, %d hex",
        mode, len(chunk), chunk["source_id"].nunique(), len(geojson["features"]),
    )
    return geojson


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Realtime yogunluk haritasi (--live/--24h/--7d)")
    parser.add_argument("--mode", choices=list(RANGE_PRESETS), required=True)
    parser.add_argument("--h3-resolution", type=int, default=5)
    parser.add_argument(
        "--loop-seconds", type=int, default=0,
        help="0 = tek seferlik (varsayilan). >0 ise bu araliklarla SUREKLI yeniden "
             "hesaplayip ayni dosyayi uzerine yazar (Ctrl+C durdurur). Frontend'in "
             "'Canli' sekmesinin gercekten tazelenmesi icin bu ARKA PLANDA calisiyor "
             "olmali -- tek seferlik calistirma dosyayi bir kere yazar, sonra kimse "
             "guncellemez (2026-07-07, kullanicinin 'gercekten taze mi' sorusu).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"density_realtime_{args.mode}.geojson"

    if args.loop_seconds <= 0:
        geojson = build_realtime_density(args.mode, args.h3_resolution)
        save_geojson(geojson, out_path)
        return

    logger.info("Loop modu: '%s' her %ds bir yeniden hesaplanacak (Ctrl+C durdurur)", args.mode, args.loop_seconds)
    while True:
        try:
            geojson = build_realtime_density(args.mode, args.h3_resolution)
            save_geojson(geojson, out_path)
        except Exception:
            logger.exception("realtime/%s hesaplanirken hata -- bir sonraki turda tekrar denenecek", args.mode)
        time.sleep(args.loop_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
