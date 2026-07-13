"""step4_build_daily_snapshots.py -- Adim 4, son 7 gunun GUNLUK yogunluk haritasi.

2026-07-10 (kullanici istegi): "son 7 gun isi haritasi" -- her takvim gunu
icin ayri bir yogunluk (benzersiz ucak sayisi) anlik goruntusu, kullanici
gunler arasinda gezinerek trafigin gun gun nasil degistigini gorebilsin.
Kapsam GLOBAL (ulke secimine bagli degil, TUM hex'ler taranir).

2026-07-11 (kullanici geri bildirimi): bu script'in ilk versiyonu iki gun
arasindaki degisimi z-skorla "anormal" olarak isaretleyen bir katman da
uretiyordu -- kullanici bunu "boş bir özellik" bulup kaldirilmasini istedi,
gunluk gorunum ("işiyi miş") kaldi. Bu yuzden mean/std/min_absolute (SADECE
o anomali hesabi icin vardi) kaldirildi -- artik sadece gun basina flight_count.

ONEMLI -- neden sadece son 7 gun (62 gunluk veri varken)?: bu ozellik
ILERIDE (bu is degil, ayri bir takip -- bkz. proje hafizasi
"project-influxdb-anomaly-hookup") canli InfluxDB'ye baglanacak, ve
InfluxDB'nin kendi mimarisi geregi SADECE 7 gunluk rolling pencere tutuluyor
(bkz. individual/metehan_geo/influx_client.py). Su an statik CSV 62 gun
tasisa da BILEREK sadece son 7 takvim gunu kullaniliyor -- veri kaynagi
CSV'den InfluxDB'ye gecince frontend mantiginda HICBIR SEY degismesin diye
(sadece load_daily_counts()'un veri kaynagi degisecek).

Cikti: tek bir GeoJSON (`daily_hex_snapshots.geojson`) -- her hex feature'i
7 gunun HER BIRI icin flight_count tasir (`counts` dizisi, days ile ayni
sirada). Hangi gunun gosterilecegi TAMAMEN istemci tarafinda (index.html)
secilir -- kullanici hangi gunu secerse secsin yeniden fetch gerekmez.

Kullanim:
    python -m individual.metehan_geo_country.step4_build_daily_snapshots
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

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
    """Her hex icin gun-sirali flight_count dizisi."""
    days = sorted(daily["day"].unique())
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
    return {
        "type": "FeatureCollection",
        "days": days,
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
