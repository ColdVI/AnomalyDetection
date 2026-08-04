"""build_daily_snapshots_backup.py -- Sunum icin YEDEK gunluk yogunluk anlik
goruntuleri (2026-08-01, kullanici istegi: "persembe sunumda video icin en az
5-6 gunluk canli veri lazim, en kotu temmuzdan 7 gun cekip hazirda tutalim").

build_daily_snapshots.py (InfluxDB'den canli, ama Docker bugun acildigi icin
su an sadece 1 gun birikmis) ile AYNI cikti sekli (days + h3_cell basina
counts dizisi), ama FARKLI veri kaynagi: burada individual/metehan_geo_country
projesinin ZATEN VAR OLAN, global kapsamli aircraft dump CSV'sinden
(data/aircraft_dump_20260707_181141.csv, 2026-05-06 -> 2026-07-07, 1.3M satir,
124MB) besleniyor -- Gold arsivinin milyar-satirlik tam taramasi (saatler
surer) yerine, zaten kucuk/hazir bu CSV'yi res5 H3 hucrelerine gruplayarak
saniyeler icinde 7 GERCEK takvim gunu uretiyoruz.

Secilen 7 gun: CSV'deki EN SON 7 gercek takvim gunu (2026-07-02 CSV'de yok,
atlanir): 06-30, 07-01, 07-03, 07-04, 07-05, 07-06, 07-07.

Kullanim (tek seferlik, sunum oncesi):
    python -m individual.metehan_geo.build_daily_snapshots_backup

Sunum gunu InfluxDB'nin kendi birikimi yetersiz kalirsa, bu betigin uretttigi
`daily_hex_snapshots_backup.geojson`, `daily_hex_snapshots.geojson` olarak
kopyalanip/yeniden adlandirilip canli dosyanin yerine konabilir (ya da
index.html'deki fetch yolunu gecici olarak buna cevirebiliriz).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from individual.metehan_geo.geo import assign_h3_cell, h3_cell_to_polygon

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

H3_RESOLUTION = 5
N_DAYS = 7
AIRCRAFT_CSV = (
    Path(__file__).resolve().parents[1] / "metehan_geo_country" / "data" / "aircraft_dump_20260707_181141.csv"
)
OUT_PATH = Path(__file__).parent / "viz" / "data" / "daily_hex_snapshots_backup.geojson"


def load_source_df() -> pd.DataFrame:
    df = pd.read_csv(AIRCRAFT_CSV, usecols=["hex", "seen_at", "latitude", "longitude"])
    df["seen_at"] = pd.to_datetime(df["seen_at"], utc=True, format="ISO8601")
    df = df.rename(columns={"hex": "source_id", "latitude": "lat", "longitude": "lon"})
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    return df


def build_daily_hex_counts(
    df: pd.DataFrame, n_days: int = N_DAYS, end_date: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """end_date verilirse (ör. "2026-06-15"), CSV'nin KENDI son gunu yerine o
    tarihte veya ONCESINDE biten n_days'lik pencere secilir -- CSV 2026-05-06
    -> 2026-07-07 arasi 63 gunu kapsiyor, test icin ARDISIK herhangi 7 gun
    (bazi gunler CSV'de eksik olabilir, o zaman pencere o kadar gun geriye
    uzar) bu sekilde secilebilir."""
    df = df.copy()
    df["day"] = df["seen_at"].dt.date.astype(str)
    available = sorted(df["day"].unique())
    if end_date is not None:
        available = [d for d in available if d <= end_date]
    days = available[-n_days:]
    windowed = df[df["day"].isin(days)]
    logger.info("Secilen %d gun: %s (%d satir)", len(days), days, len(windowed))

    chunk = assign_h3_cell(windowed, H3_RESOLUTION)
    daily = chunk.groupby(["h3_cell", "day"])["source_id"].nunique().reset_index(name="flight_count")
    return daily, days


def build_snapshot_payload(daily: pd.DataFrame, days: list[str]) -> dict:
    pivot = daily.pivot_table(index="h3_cell", columns="day", values="flight_count", fill_value=0)
    pivot = pivot.reindex(columns=days, fill_value=0)

    features = []
    for h3_cell, row in pivot.iterrows():
        ring = h3_cell_to_polygon(h3_cell)
        if ring is None:
            continue
        features.append({
            "type": "Feature",
            "properties": {"h3_cell": h3_cell, "counts": [int(v) for v in row.to_numpy()]},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    logger.info("Toplam %d hex, %d gun", len(features), len(days))
    return {"type": "FeatureCollection", "days": days, "features": features}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Yedek gunluk yogunluk anlik goruntusu (aircraft_dump CSV'den)")
    parser.add_argument(
        "--end-date", default=None,
        help='Pencerenin biteceği tarih (ör. "2026-06-15") -- verilmezse CSV\'nin kendi son gunu kullanilir. '
             "Test icin CSV'nin 2026-05-06 -> 2026-07-07 araligindan istenen herhangi 7-gunluk pencere secilebilir.",
    )
    parser.add_argument("--n-days", type=int, default=N_DAYS)
    parser.add_argument(
        "--out", default=None,
        help="Cikti dosyasi (varsayilan: viz/data/daily_hex_snapshots_backup.geojson)",
    )
    args = parser.parse_args()

    df = load_source_df()
    daily, days = build_daily_hex_counts(df, n_days=args.n_days, end_date=args.end_date)
    payload = build_snapshot_payload(daily, days)

    out_path = Path(args.out) if args.out else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    logger.info("Yazildi: %s (%d gun, %d hex)", out_path, len(payload["days"]), len(payload["features"]))


if __name__ == "__main__":
    main()
