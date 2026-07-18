"""step2_build_country_layers.py -- Adim 2, gorsellestirme katmanlari.

2026-07-08 revizyonu (kullanici geri bildirimi uzerine 3 degisiklik):
  1. H3 resolution res3 -> res5 ("hexler cok buyuk").
  2. Bir ulke secildiginde sadece BASKIN oldugu degil, UGRADIGI TUM hexler
     gosterilsin -- bu yuzden her hex'in "countries" (o hucreden gecen tum
     ulkeler) ve "per_country_counts" (ulke basina benzersiz ucak) listesini
     de tasiyoruz, sadece dominant_country'i degil.
  3. Zaman filtresi: veri kendi ic takviminde 62 gun kapsiyor (2026-05-06 -
     2026-07-07) -- GERCEK "simdi"ye gore degil, VERININ KENDI
     max(seen_at) degerine GORE "son 1 gun/1 hafta/30 gun" hesaplanir.

Choropleth (ulke bazli renklendirme) modu kaldirildi (hoca istemedi,
2026-07-08 karari) -- Natural Earth poligonlari artik SADECE notr bir
basemap referans katmani icin kullaniliyor, ulke adi eslestirme/alias
mantigina gerek kalmadi.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from individual.metehan_geo.geo import assign_h3_cell, h3_cell_to_polygon
from src.common.hex_country import HexCountryLookup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
VIZ_DATA_DIR = Path(__file__).parent / "viz" / "data"
AIRCRAFT_CSV = DATA_DIR / "aircraft_dump_20260707_181141.csv"
NE_COUNTRIES_GEOJSON = DATA_DIR / "ne_50m_admin_0_countries.geojson"
# 2026-07-10 (kullanici istegi): bu projenin ham CSV dump'i (aircraft_dump_*)
# dbFlags/is_military TASIMIYOR (sadece hex/seen_at/lat/lon/alt/velocity/
# heading/callsign/source, bkz. modul disi arastirma). is_military UCAK-
# BAZLI sabit bir ozellik oldugu icin, ana projenin (individual/metehan_geo)
# TUM Gold verisini zaten tarayan build_flight_density.py'sinin URETTIGI
# paylasilan lookup dosyasi -- her ucagin (hex/icao24) en az bir kere askeri
# gorulup gorulmedigi -- burada "hex" uzerinden JOIN edilir. Ayri bir tam
# taramaya GEREK YOK.
MILITARY_LOOKUP_PARQUET = (
    Path(__file__).parent.parent / "metehan_geo" / "viz" / "data" / "aircraft_military_lookup.parquet"
)

H3_RESOLUTION = 5

# Veri kendi ic takviminde 62 gun kapsiyor (2026-05-06 - 2026-07-07) --
# pencereler dataset'in KENDI max(seen_at) degerine GORE (gercek "simdi"ye
# degil) hesaplaniyor, bkz. modul docstring. 2026-07-08: 1 gun/1 hafta/30 gun
# + "all" (tum 62 gun, "historical" -- kullanici: "historical da kalsin").
TIME_WINDOWS: dict[str, pd.Timedelta | None] = {
    "1d": pd.Timedelta(days=1),
    "7d": pd.Timedelta(days=7),
    "30d": pd.Timedelta(days=30),
    "all": None,
}


def load_military_lookup() -> dict[str, bool]:
    """hex (icao24) -> is_military. Dosya yoksa (main proje henuz
    build_flight_density.py'yi calistirmadiysa) bos sozluk doner --
    her ucak varsayilan olarak sivil sayilir, feature sessizce devre disi
    kalir (hata firlatmaz)."""
    if not MILITARY_LOOKUP_PARQUET.exists():
        logger.warning(
            "%s bulunamadi -- once individual/metehan_geo/build_flight_density.py "
            "calistirilmali, simdilik tum ucaklar sivil sayilacak", MILITARY_LOOKUP_PARQUET,
        )
        return {}
    lookup_df = pd.read_parquet(MILITARY_LOOKUP_PARQUET)
    return dict(zip(lookup_df["source_id"], lookup_df["is_military"]))


def load_enriched_aircraft_df() -> pd.DataFrame:
    df = pd.read_csv(AIRCRAFT_CSV)
    df["seen_at"] = pd.to_datetime(df["seen_at"], utc=True, format="ISO8601")

    lookup = HexCountryLookup()
    unique_hex = df["hex"].dropna().unique()
    hex_to_country = {h: lookup.lookup(h)[0] for h in unique_hex}
    df["country"] = df["hex"].map(hex_to_country)

    military_lookup = load_military_lookup()
    df["is_military"] = df["hex"].map(military_lookup).fillna(False).astype(bool)
    logger.info(
        "Askeri esleme: %d/%d benzersiz ucak (hex) askeri isaretli (lookup: %d ucak)",
        int(df.loc[df["is_military"], "hex"].nunique()), df["hex"].nunique(), len(military_lookup),
    )
    return df


def build_country_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Ulke basina (TUM zaman araligi) benzersiz hex (~ucak) ve satir sayisi.

    Sadece referans/adim-1 ciktisi olarak tutuluyor -- gorsellestirme artik
    zaman-pencereli H3 katmanlarini kullaniyor (asagida).
    """
    with_country = df[df["country"].notna()]
    grouped = with_country.groupby("country").agg(
        hex_count=("hex", "nunique"),
        row_count=("hex", "size"),
    ).reset_index()
    return grouped.sort_values("hex_count", ascending=False)


def build_basemap_geojson() -> dict:
    """Natural Earth ulke poligonlari -- SADECE geometri + isim, notr
    basemap referans katmani icin (choropleth join'i yok, kaldirildi)."""
    with open(NE_COUNTRIES_GEOJSON, encoding="utf-8") as f:
        ne = json.load(f)
    features = [
        {
            "type": "Feature",
            "properties": {"name": feat["properties"].get("NAME")},
            "geometry": feat["geometry"],
        }
        for feat in ne["features"]
    ]
    return {"type": "FeatureCollection", "features": features}


def build_h3_layer_for_window(df: pd.DataFrame, window_label: str, delta: pd.Timedelta | None) -> dict:
    with_country = df[df["country"].notna()]
    if delta is not None:
        max_t = with_country["seen_at"].max()
        with_country = with_country[with_country["seen_at"] >= max_t - delta]

    if with_country.empty:
        logger.warning("%s: pencerede veri yok", window_label)
        return {"type": "FeatureCollection", "features": [], "breaks": [1, 1, 1, 1, 1]}

    renamed = with_country.rename(columns={"latitude": "lat", "longitude": "lon"})
    renamed = assign_h3_cell(renamed, H3_RESOLUTION)

    # (hucre, ulke) basina BENZERSIZ hex (~ucak) sayisi -- ayni ucagin
    # binlerce trace noktasi tek "ucak" olarak sayilsin.
    per_cell_country = (
        renamed.groupby(["h3_cell", "country"])["hex"].nunique().reset_index(name="hex_count")
    )
    # 2026-07-10 (kullanici istegi): Uçak Türü (Sivil/Askeri) filtresi icin
    # AYNI sayimi is_military kirilimiyla da hesapliyoruz -- dominant_country
    # HALA toplam uzerinden secilir (degismiyor), sadece per_country_counts'un
    # yaninda iki EK kirilim (civil/military) tasiniyor, frontend'de secime
    # gore hangisinin kullanilacagina karar verilir (main projedeki
    # flight_count_civil/flight_count_military ile AYNI desen).
    per_cell_country_mil = (
        renamed.groupby(["h3_cell", "country", "is_military"])["hex"].nunique().reset_index(name="hex_count")
    )
    mil_by_cell = {cell: g for cell, g in per_cell_country_mil.groupby("h3_cell")}

    features = []
    totals = []
    for cell, group in per_cell_country.groupby("h3_cell"):
        ring = h3_cell_to_polygon(cell)
        if ring is None:
            continue
        total = int(group["hex_count"].sum())
        top = group.loc[group["hex_count"].idxmax()]
        per_country_counts = {row.country: int(row.hex_count) for row in group.itertuples(index=False)}

        per_country_counts_civil = {c: 0 for c in per_country_counts}
        per_country_counts_military = {c: 0 for c in per_country_counts}
        mil_group = mil_by_cell.get(cell)
        if mil_group is not None:
            for row in mil_group.itertuples(index=False):
                target = per_country_counts_military if row.is_military else per_country_counts_civil
                target[row.country] = target.get(row.country, 0) + int(row.hex_count)

        totals.append(total)
        features.append({
            "type": "Feature",
            "properties": {
                "h3_cell": cell,
                "dominant_country": top["country"],
                "dominant_hex_count": int(top["hex_count"]),
                "total_hex_count": total,
                "dominant_ratio": round(int(top["hex_count"]) / total, 3),
                "countries": sorted(per_country_counts.keys()),
                "per_country_counts": per_country_counts,
                "per_country_counts_civil": per_country_counts_civil,
                "per_country_counts_military": per_country_counts_military,
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    breaks_series = pd.Series(totals) if totals else pd.Series([1])
    breaks = [
        max(1, int(breaks_series.quantile(p)))
        for p in (0.10, 0.50, 0.75, 0.90, 0.99)
    ]
    logger.info(
        "H3 (res%d, %s): %d hex, breaks=%s", H3_RESOLUTION, window_label, len(features), breaks,
    )
    return {"type": "FeatureCollection", "features": features, "breaks": breaks}


def main() -> None:
    df = load_enriched_aircraft_df()

    country_counts = build_country_counts(df)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    country_counts.to_csv(DATA_DIR / "country_counts.csv", index=False)
    logger.info("Yazildi: %s (%d ulke)", DATA_DIR / "country_counts.csv", len(country_counts))

    VIZ_DATA_DIR.mkdir(parents=True, exist_ok=True)

    basemap = build_basemap_geojson()
    with (VIZ_DATA_DIR / "country_basemap.geojson").open("w", encoding="utf-8") as f:
        json.dump(basemap, f)
    logger.info("Yazildi: %s (%d feature)", VIZ_DATA_DIR / "country_basemap.geojson", len(basemap["features"]))

    for window_label, delta in TIME_WINDOWS.items():
        layer = build_h3_layer_for_window(df, window_label, delta)
        out_path = VIZ_DATA_DIR / f"h3_{window_label}.geojson"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(layer, f)
        logger.info("Yazildi: %s (%d feature)", out_path, len(layer["features"]))


if __name__ == "__main__":
    main()
