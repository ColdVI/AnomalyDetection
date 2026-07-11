"""step4_build_daily_snapshots.py -- Adim 4, gun-bazli hex anomali tespiti.

2026-07-10 (kullanici istegi): "son 7 gun isi haritasi degisimi" -- herhangi
iki gun arasinda yogunlugu (benzersiz ucak sayisi) ANORMAL degisen hex'leri
isaretleyip kullaniciyi arayuzde uyarmak. Kapsam GLOBAL (ulke secimine
bagli degil, TUM hex'ler taranir).

ONEMLI -- neden sadece son 7 gun (62 gunluk veri varken)?: bu ozellik
ILERIDE (bu is degil, ayri bir takip -- bkz. proje hafizasi
"project-influxdb-anomaly-hookup") canli InfluxDB'ye baglanacak, ve
InfluxDB'nin kendi mimarisi geregi SADECE 7 gunluk rolling pencere tutuluyor
(bkz. individual/metehan_geo/influx_client.py). Su an statik CSV 62 gun
tasisa da BILEREK sadece son 7 takvim gunu kullaniliyor -- veri kaynagi
CSV'den InfluxDB'ye gecince frontend/anomali mantiginda HICBIR SEY
degismesin diye (sadece load_daily_counts()'un veri kaynagi degisecek).

Cikti: tek bir GeoJSON (`daily_hex_snapshots.geojson`) -- her hex feature'i
7 gunun HER BIRI icin flight_count tasir (`counts` dizisi, days ile ayni
sirada) + o hex'in kendi 7-gunluk ortalama/std'si (z-skor hesaplamasi icin).
Anomali skoru (z-skor) VE hangi iki gun karsilastirilacagi TAMAMEN
istemci tarafinda (index.html) hesaplanir -- boylece kullanici hangi gun
ciftini secerse secsin yeniden fetch/hesaplama gerekmez (civil/military
filtresiyle ayni "tek dosya, istemci karsilastirir" deseni).

Kullanim:
    python -m individual.metehan_geo_country.step4_build_daily_snapshots
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from individual.metehan_geo.geo import assign_h3_cell, h3_cell_to_polygon
from individual.metehan_geo_country.step2_build_country_layers import (
    VIZ_DATA_DIR,
    load_enriched_aircraft_df,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

H3_RESOLUTION = 5
# 2026-07-10: InfluxDB'nin 7-gunluk rolling penceresiyle BIREBIR eslessin diye
# kasitli olarak 7 -- bkz. modul docstring'i.
WINDOW_DAYS = 7
# Kucuk mutlak sayilarda (ör. 1 ucaktan 3 ucaga cikmak) z-skor/oransal degisim
# yapay derecede carpici cikabilir -- iki gunden EN AZ biri bu esigi
# gecmedikce hex anomali adayi bile sayilmiyor (frontend'de uygulanir,
# burada sadece parametre olarak tasiniyor).
MIN_ABSOLUTE = 3


def build_daily_hex_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Ham noktalari (ulke atanmis, H3 hucrelenmis) TAKVIM GUNU basina
    benzersiz ucak sayisina indirger -- son WINDOW_DAYS takvim gunuyle
    sinirli (verinin KENDI max(seen_at) tarihine gore, step2'deki ayni
    "gercek simdi degil, verinin kendi takvimi" ilkesi)."""
    max_date = df["seen_at"].max().date()
    window_start = max_date - pd.Timedelta(days=WINDOW_DAYS - 1)
    windowed = df[df["seen_at"].dt.date >= window_start].copy()
    windowed["day"] = windowed["seen_at"].dt.date.astype(str)
    logger.info(
        "Pencere: %s -> %s (%d takvim gunu), %d satir",
        window_start, max_date, WINDOW_DAYS, len(windowed),
    )

    renamed = windowed.rename(columns={"latitude": "lat", "longitude": "lon"})
    renamed = assign_h3_cell(renamed, H3_RESOLUTION)

    daily = (
        renamed.groupby(["h3_cell", "day"])["hex"].nunique().reset_index(name="flight_count")
    )
    return daily


def build_snapshot_payload(daily: pd.DataFrame) -> dict:
    """Her hex icin gun-sirali flight_count dizisi + o hex'in KENDI 7-gunluk
    ortalama/std'si (mutlak bir esik yerine goreceli anomali icin -- bu
    projede defalarca ogrenildi: tek global esik farkli olcekteki hex'ler
    icin adil degil, bkz. geo_clustering.compute_knn_local_mask)."""
    days = sorted(daily["day"].unique())
    pivot = daily.pivot_table(index="h3_cell", columns="day", values="flight_count", fill_value=0)
    pivot = pivot.reindex(columns=days, fill_value=0)

    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1)

    features = []
    for h3_cell, row in pivot.iterrows():
        ring = h3_cell_to_polygon(h3_cell)
        if ring is None:
            continue
        hex_std = std[h3_cell]
        features.append({
            "type": "Feature",
            "properties": {
                "h3_cell": h3_cell,
                "counts": [int(v) for v in row.to_numpy()],
                "mean": round(float(mean[h3_cell]), 2),
                # sabit-degerli (7 gun boyunca hep ayni sayi) hex'te std=0 --
                # z-skor tanimsiz/sonsuz olur, None birakiyoruz (frontend
                # bunu "yeni/kayip" disindaki normal degisim analizinden
                # haric tutar).
                "std": None if hex_std == 0 else round(float(hex_std), 3),
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    logger.info("Toplam %d hex, %d gun", len(features), len(days))
    return {
        "type": "FeatureCollection",
        "days": days,
        "min_absolute": MIN_ABSOLUTE,
        "features": features,
    }


def main() -> None:
    df = load_enriched_aircraft_df()
    daily = build_daily_hex_counts(df)
    payload = build_snapshot_payload(daily)

    VIZ_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIZ_DATA_DIR / "daily_hex_snapshots.geojson"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    logger.info("Yazildi: %s (%d gun, %d hex)", out_path, len(payload["days"]), len(payload["features"]))


if __name__ == "__main__":
    main()
