"""build_flight_density.py -- duzeltilmis yogunluk metrigi (kullanicinin
2026-07-07 kavramsal duzeltmesi).

SORUN: compute_hex_density() (main.py) ham TRACE NOKTASI sayisini topluyordu
(groupby(h3_cell).size()). ADS-B saniyede birkac kez pozisyon uretiyor --
havaalani yakininda yavas/bekleyen ucak ayni hex'te onlarca nokta birakirken,
cruise hizindaki bir ucak sadece birkac nokta birakiyor. Bu "en sik kullanilan
ROTA" sorusuna yanlis cevap veriyor: havaalani/yavas bolgeler yapay sisiyor.

DUZELTME: Asil metrik BENZERSIZ UCUS SAYISI (distinct source_id+date, ham nokta
degil). Ek metrik: gun-tutarliligi (0-11/12 -- kac ayri günde bu hex'te trafik
gorulmus).

ONEMLI -- coklu resolution NEDEN AYRI AYRI HESAPLANIYOR (parent'tan toplanmiyor):
point_count icin cell_to_parent ile ust cozunurluge toplama matematiksel olarak
dogrudur (toplam nokta = alt hucrelerin toplami). AMA flight_count icin YANLIS:
ayni ucak res5'te 3 farkli hex'ten gecip hepsi ayni res4 ebeveynine bagliysa,
cocuklari toplamak o ucagi 3 kez sayar (mukerrer). Bu yuzden flight_count/
day_count HER resolution icin dogrudan (kendi cozunurlugunde distinct-count ile)
hesaplaniyor -- tek gecist,e 3 resolution birden (r3/r4/r5), ekstra CPU maliyeti
(satir basina 3 h3.latlng_to_cell + 3 dedup) var ama 3 ayri tam-veri taramasindan
(3 x ~30dk) çok daha ucuz.

Varsayim (bellek verimliligi icin, GERCEK VERIYLE DOGRULANDI): Ilk beklentim
"bir chunk = bir tarih" idi ama gercek veri boyle degil -- trace_full dosyalari
~birkac gunluk rolling history tasiyor, tek bir chunk 2-4 farkli tarihe ait
satir icerebiliyor (bkz. smoke test, chunk 0: hem 2025-10-14 hem 2025-10-15).
Chunk-ici dedup (drop_duplicates) bunu zaten dogru ele aliyor.

2026-07-13 GUNCELLEME: yukaridaki "ardisik tar'lar arasi tarih araligi
cakismiyor" varsayimi artik GECERSIZ -- kullanici 19 yeni tar ekledi, bazilari
(orn. v2025.08.21 + v2025.08.26 = 5 gun, v2025.10.25 + v2025.10.28 = 3 gun)
birbirine COK yakin, rolling-window trace'leri CAKISABILIR. Bu yuzden
flight_count/civil/military artik Counter TOPLAMA (chunk'lar arasi cift
sayim riski tasiyordu) DEGIL, hex basina GERCEK GLOBAL SET
{(source_id, date)} ile hesaplaniyor -- ayni ucus iki farkli tar/chunk'ta
gorulse bile set'e sadece bir kez girer. day_count zaten (hex_days) bastan
beri gercek set kullaniyordu, o kisim etkilenmedi.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import h3
import pandas as pd

from individual.metehan_geo.data import clean_coordinates, load_adsb_gold_data
from individual.metehan_geo.viz import build_density_geojson, save_geojson

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "viz" / "data"
RESOLUTIONS = (3, 4, 5)


def run() -> None:
    point_counts = {r: Counter() for r in RESOLUTIONS}
    # 2026-07-13: flight_count/civil/military artik Counter DEGIL -- hex basina
    # gercek global {(source_id, date)} set'i (bkz. modul basi notu, tar'lar
    # arasi cakisma cift-sayimi onlemek icin). Sivil/askeri ayrimi finalize
    # asamasinda military_lookup'a bakilarak set'ten turetiliyor.
    flight_hex_sets: dict[int, dict[str, set]] = {r: {} for r in RESOLUTIONS}
    hex_days: dict[int, dict[str, set]] = {r: {} for r in RESOLUTIONS}
    # source_id (icao24) -> is_military -- TEK SEFERLIK global lookup, ayni
    # taramada bedava toplanir. metehan_geo_country projesi kendi ham CSV
    # dump'inda dbFlags TASIMADIGI icin bunu "hex" (=icao24) uzerinden
    # AYRI bir dosyadan (aircraft_military_lookup.parquet) join edecek.
    military_lookup: dict[str, bool] = {}

    total_rows = 0
    for i, raw_chunk in enumerate(load_adsb_gold_data()):
        cleaned = clean_coordinates(raw_chunk)
        if cleaned.empty:
            continue
        total_rows += len(cleaned)

        if "is_military" in cleaned.columns:
            is_mil = cleaned["is_military"].fillna(False).astype(bool)
        else:
            is_mil = pd.Series(False, index=cleaned.index)
        cleaned = cleaned.assign(is_military=is_mil)
        for source_id, mil in zip(cleaned["source_id"], is_mil):
            military_lookup[source_id] = military_lookup.get(source_id, False) or bool(mil)

        dt = pd.to_datetime(cleaned["timestamp_utc"], unit="s", errors="coerce")
        cleaned = cleaned.assign(_date=dt.dt.date)
        cleaned = cleaned.dropna(subset=["_date"])

        for r in RESOLUTIONS:
            cells = [h3.latlng_to_cell(lat, lon, r) for lat, lon in zip(cleaned["lat"], cleaned["lon"])]
            chunk = cleaned.assign(h3_cell=cells)

            point_counts[r].update(chunk["h3_cell"].value_counts().to_dict())

            flight_dedup = chunk.drop_duplicates(subset=["h3_cell", "source_id", "_date"])
            sets_r = flight_hex_sets[r]
            for h3_cell, source_id, date in zip(
                flight_dedup["h3_cell"], flight_dedup["source_id"], flight_dedup["_date"]
            ):
                sets_r.setdefault(h3_cell, set()).add((source_id, date))

            day_dedup = chunk.drop_duplicates(subset=["h3_cell", "_date"])
            days_r = hex_days[r]
            for h3_cell, date in zip(day_dedup["h3_cell"], day_dedup["_date"]):
                days_r.setdefault(h3_cell, set()).add(date)

        if (i + 1) % 200 == 0:
            logger.info(
                "  %d chunk, %d satir, res5 hex(point/flight)=%d/%d",
                i + 1, total_rows, len(point_counts[5]), len(flight_hex_sets[5]),
            )

    logger.info("Tamamlandi: %d satir taniındi", total_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_military = sum(military_lookup.values())
    logger.info(
        "Askeri ucak lookup: %d/%d ucak (icao24) askeri isaretli",
        n_military, len(military_lookup),
    )
    lookup_df = pd.DataFrame(
        {"source_id": list(military_lookup.keys()), "is_military": list(military_lookup.values())}
    )
    lookup_df.to_parquet(OUT_DIR / "aircraft_military_lookup.parquet", index=False)
    logger.info("Yazildi: %s (%d ucak)", OUT_DIR / "aircraft_military_lookup.parquet", len(lookup_df))

    for r in RESOLUTIONS:
        sets_r = flight_hex_sets[r]
        all_hexes = set(point_counts[r]) | set(sets_r) | set(hex_days[r])
        flight_count, flight_count_civil, flight_count_military = {}, {}, {}
        for h in all_hexes:
            entries = sets_r.get(h, ())
            mil = sum(1 for source_id, _ in entries if military_lookup.get(source_id, False))
            flight_count[h] = len(entries)
            flight_count_military[h] = mil
            flight_count_civil[h] = len(entries) - mil
        density_df = pd.DataFrame({
            "h3_cell": list(all_hexes),
            "point_count": [point_counts[r].get(h, 0) for h in all_hexes],
            "flight_count": [flight_count[h] for h in all_hexes],
            "flight_count_civil": [flight_count_civil[h] for h in all_hexes],
            "flight_count_military": [flight_count_military[h] for h in all_hexes],
            "day_count": [len(hex_days[r].get(h, ())) for h in all_hexes],
        })
        density_df.to_parquet(OUT_DIR / f"density_flights_res{r}.parquet", index=False)
        logger.info(
            "res%d: %d hex, flight_count medyan=%.0f, day_count medyan=%.0f, max point/flight orani=%.0f, "
            "askeri-only trafik gozlenen hex=%d",
            r, len(density_df), density_df["flight_count"].median(), density_df["day_count"].median(),
            (density_df["point_count"] / density_df["flight_count"].replace(0, 1)).max(),
            int((density_df["flight_count_military"] > 0).sum()),
        )

        # GeoJSON: build_density_geojson point_count kolonu bekliyor -- flight_count'u
        # gecici olarak o isimle kullanip ayni fonksiyonu tekrar kullaniyoruz.
        geojson_input = density_df[["h3_cell", "flight_count"]].rename(columns={"flight_count": "point_count"})
        geojson = build_density_geojson(geojson_input)
        day_count_by_hex = dict(zip(density_df["h3_cell"], density_df["day_count"]))
        point_count_by_hex = dict(zip(density_df["h3_cell"], density_df["point_count"]))
        civil_by_hex = dict(zip(density_df["h3_cell"], density_df["flight_count_civil"]))
        military_by_hex = dict(zip(density_df["h3_cell"], density_df["flight_count_military"]))
        for feature in geojson["features"]:
            h = feature["properties"]["h3_cell"]
            feature["properties"]["day_count"] = int(day_count_by_hex[h])
            feature["properties"]["flight_count_civil"] = int(civil_by_hex[h])
            feature["properties"]["flight_count_military"] = int(military_by_hex[h])
            feature["properties"]["point_count_raw"] = int(point_count_by_hex[h])
        save_geojson(geojson, OUT_DIR / f"density_flights_res{r}.geojson")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
