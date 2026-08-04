"""build_daily_snapshots.py -- Gunluk yogunluk anlik goruntuleri.

2026-08-01 (kullanici istegi): "bizim ek geo projesi gibi gunluk shift
yogunluk gorme gelsin" -- individual/metehan_geo_country/step4_build_daily_snapshots.py
ile AYNI cikti sekli (her hex icin gun-basina flight_count dizisi, hangi
gunun gosterilecegi TAMAMEN istemci tarafinda secilir) ama FARKLI veri
kaynagi: o proje sabit CSV'den (62 gunluk statik arsiv) besleniyordu, bu
script InfluxDB'nin canli rolling penceresinden besleniyor (bkz.
influx_client.py, docker-compose "streaming" profili).

WINDOW_DAYS=7: InfluxDB'nin kendi mimarisi (7 gunluk rolling pencere) ile
sinirli zaten -- bkz. proje hafizasi "influxdb-anomaly-hookup" (metehan_geo_country
tarafinda da ayni sinir icin ayni sebep). Docker bu oturumda yeni acildigi
icin ilk calistirmalarda sadece 1 gun gorunecek, InfluxDB'ye veri biriktikce
sonraki calistirmalarda otomatik artacak.

Kullanim (manuel, kullanicinin tercih ettigi calistirma deseni -- bkz.
gunluk_kayit.md "realtime-silver-scheduler-manual"):
    python -m individual.metehan_geo.build_daily_snapshots

--loop-seconds ile arka planda periyodik yeniden hesaplama da mumkun
(realtime_density.py ile ayni desen), ama varsayilan tek seferlik.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from individual.metehan_geo.data import clean_coordinates
from individual.metehan_geo.geo import assign_h3_cell, h3_cell_to_polygon
from individual.metehan_geo.influx_client import load_realtime_window

logger = logging.getLogger(__name__)

H3_RESOLUTION = 5
WINDOW_DAYS = 7
OUT_DIR = Path(__file__).parent / "viz" / "data"


def build_daily_hex_counts(df: pd.DataFrame, h3_resolution: int = H3_RESOLUTION) -> pd.DataFrame:
    """Ham noktalari (temizlenmis, H3 hucrelenmis) TAKVIM GUNU basina
    benzersiz ucak sayisina indirger."""
    if df.empty:
        return pd.DataFrame(columns=["h3_cell", "day", "flight_count"])

    cleaned = clean_coordinates(df)
    if cleaned.empty:
        return pd.DataFrame(columns=["h3_cell", "day", "flight_count"])

    cleaned = cleaned.copy()
    cleaned["day"] = pd.to_datetime(cleaned["_time"]).dt.date.astype(str)
    chunk = assign_h3_cell(cleaned, h3_resolution)

    daily = (
        chunk.groupby(["h3_cell", "day"])["source_id"].nunique().reset_index(name="flight_count")
    )
    return daily


def build_snapshot_payload(daily: pd.DataFrame) -> dict:
    """Her hex icin gun-sirali flight_count dizisi -- son WINDOW_DAYS takvim
    gunuyle sinirli (verinin KENDI en son gunune gore)."""
    if daily.empty:
        return {"type": "FeatureCollection", "days": [], "features": []}

    days = sorted(daily["day"].unique())[-WINDOW_DAYS:]
    daily = daily[daily["day"].isin(days)]
    pivot = daily.pivot_table(index="h3_cell", columns="day", values="flight_count", fill_value=0)
    pivot = pivot.reindex(columns=days, fill_value=0)

    features = []
    for h3_cell, row in pivot.iterrows():
        ring = h3_cell_to_polygon(h3_cell)
        if ring is None:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "h3_cell": h3_cell,
                "counts": [int(v) for v in row.to_numpy()],
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    logger.info("Toplam %d hex, %d gun", len(features), len(days))
    return {"type": "FeatureCollection", "days": days, "features": features}


def build_daily_snapshots(h3_resolution: int = H3_RESOLUTION) -> dict:
    df = load_realtime_window("-7d")
    daily = build_daily_hex_counts(df, h3_resolution)
    return build_snapshot_payload(daily)


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Gunluk yogunluk anlik goruntuleri (InfluxDB'den)")
    parser.add_argument("--h3-resolution", type=int, default=H3_RESOLUTION)
    parser.add_argument(
        "--loop-seconds", type=int, default=0,
        help="0 = tek seferlik (varsayilan). >0 ise bu araliklarla SUREKLI yeniden "
             "hesaplayip ayni dosyayi uzerine yazar (Ctrl+C durdurur).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "daily_hex_snapshots.geojson"

    def run_once() -> None:
        payload = build_daily_snapshots(args.h3_resolution)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        logger.info(
            "Yazildi: %s (%d gun, %d hex)", out_path, len(payload["days"]), len(payload["features"]),
        )

    if args.loop_seconds <= 0:
        run_once()
        return

    logger.info("Loop modu: her %ds bir yeniden hesaplanacak (Ctrl+C durdurur)", args.loop_seconds)
    while True:
        try:
            run_once()
        except Exception:
            logger.exception("gunluk anlik goruntu hesaplanirken hata -- bir sonraki turda tekrar denenecek")
        time.sleep(args.loop_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
