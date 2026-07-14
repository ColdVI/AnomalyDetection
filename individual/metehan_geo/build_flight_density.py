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
sayim riski tasiyordu) DEGIL, hex basina GERCEK GLOBAL SET ile hesaplaniyor.

2026-07-14 BELLEK DUZELTMESI (kritik): ilk versiyon set'e dogrudan
(source_id, date) TUPLE'i (str + date nesnesi) ekliyordu -- 30 tar'lik veride
(res3+res4+res5 toplaminda) bu, tek process'i 32GB+ ozel bellege (fiziksel
RAM'in 2 katindan fazla) sisirip agir disk swap'ine soktu, pratikte
DURMUS gibi goruntu verdi (saatlerce tek bir 200-chunk ilerlemedi). Kok
neden: Python'da bir (str, date) tuple'i set icinde ~150-250+ bayt kaplarken
tek bir int ~28 bayt kapliyor. Duzeltme: source_id VE date, ilk gorulusunde
kucuk birer tam sayiya (idx) donusturulup TEK bir paketlenmis int
(`source_idx * DATE_MULTIPLIER + date_idx`) olarak set'e ekleniyor -- ayni
mantik (global dedup), ~5-8 kat daha az bellek.
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter
from datetime import date as _date_cls
from pathlib import Path

import h3
import pandas as pd

from individual.metehan_geo.data import clean_coordinates, load_adsb_gold_data
from individual.metehan_geo.viz import build_density_geojson, save_geojson

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "viz" / "data"
RESOLUTIONS = (3, 4, 5)

# (source_idx * DATE_MULTIPLIER + date_idx) paketlemesi icin -- veri seti
# gercekte ~330 gun kapsiyor, 100_000 cok genis bir pay birakiyor (unsigned
# int carpimlarinda anlamsiz cakisma olmasin diye).
_DATE_MULTIPLIER = 100_000
_EPOCH = _date_cls(2020, 1, 1)

# 2026-07-14 (kullanici istegi -- "3 saate PC kapanacak, checkpoint ekleyelim"):
# bu is Silver reprocess'ten FARKLI bir sekilde checkpoint'lenir -- orada her
# tar BAGIMSIZ bir dosyaya yaziliyordu, burada TUM Gold parcalarinin sonucu
# TEK bir bellek-ici birikimde (point_counts/flight_hex_sets/hex_days/...)
# tutulup en sonda yaziliyor. Bu yuzden checkpoint, o birikimin TAMAMINI
# periyodik olarak (her CHECKPOINT_EVERY_PARTS parcada bir) pickle ile diske
# yazar -- Silver'daki gibi "parca-parca dosya" degil "periyodik tam anlik
# goruntu" (snapshot) modeli. Islenmis parca adlari da saklanir ki resume'de
# ayni parcalar TEKRAR indirilip islenmesin (load_adsb_gold_data(skip_names=...)).
CHECKPOINT_PATH = Path("data/state/density_build_checkpoint.pkl")
CHECKPOINT_EVERY_PARTS = 500


def _save_checkpoint(path: Path, state: dict) -> None:
    """Yaz-sonra-degistir (atomic write) -- Silver checkpoint'iyle ayni desen,
    kesinti aninda yarim/bozuk bir checkpoint dosyasi kalmasin diye."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        logger.warning("Checkpoint okunamadi/bozuk, sifirdan basliyor: %s", path, exc_info=True)
        return None


def run(*, fresh: bool = False, checkpoint_path: Path = CHECKPOINT_PATH) -> None:
    checkpoint = None if fresh else _load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        logger.info(
            "Checkpoint bulundu (%s) -- %d parca zaten islenmis, kaldigi yerden devam ediliyor",
            checkpoint_path, len(checkpoint["processed_names"]),
        )
        point_counts = checkpoint["point_counts"]
        flight_hex_sets = checkpoint["flight_hex_sets"]
        hex_days = checkpoint["hex_days"]
        military_lookup = checkpoint["military_lookup"]
        source_id_to_idx = checkpoint["source_id_to_idx"]
        processed_names: set[str] = checkpoint["processed_names"]
        total_rows = checkpoint["total_rows"]
    else:
        if fresh and checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("--fresh: eski checkpoint silindi")
        point_counts = {r: Counter() for r in RESOLUTIONS}
        # 2026-07-14: hex basina set artik (source_id, date) TUPLE'i degil, TEK
        # paketlenmis int tutuyor (bkz. modul basi bellek-duzeltmesi notu).
        flight_hex_sets: dict[int, dict[str, set]] = {r: {} for r in RESOLUTIONS}
        hex_days: dict[int, dict[str, set]] = {r: {} for r in RESOLUTIONS}
        # source_id (icao24) -> is_military -- TEK SEFERLIK global lookup, ayni
        # taramada bedava toplanir. metehan_geo_country projesi kendi ham CSV
        # dump'inda dbFlags TASIMADIGI icin bunu "hex" (=icao24) uzerinden
        # AYRI bir dosyadan (aircraft_military_lookup.parquet) join edecek.
        military_lookup: dict[str, bool] = {}
        # source_id -> kucuk int idx (ilk gorulusunde atanir) -- set'lerde
        # string yerine bu int tutuluyor, bellek/hiz icin.
        source_id_to_idx: dict[str, int] = {}
        processed_names = set()
        total_rows = 0

    def _source_idx(source_id: str) -> int:
        idx = source_id_to_idx.get(source_id)
        if idx is None:
            idx = len(source_id_to_idx)
            source_id_to_idx[source_id] = idx
        return idx

    def _checkpoint_now() -> None:
        _save_checkpoint(checkpoint_path, {
            "point_counts": point_counts, "flight_hex_sets": flight_hex_sets,
            "hex_days": hex_days, "military_lookup": military_lookup,
            "source_id_to_idx": source_id_to_idx, "processed_names": processed_names,
            "total_rows": total_rows,
        })
        logger.info("  Checkpoint kaydedildi (%d parca islenmis)", len(processed_names))

    parts_since_checkpoint = 0
    for i, (part_name, raw_chunk) in enumerate(load_adsb_gold_data(skip_names=frozenset(processed_names), yield_names=True)):
        cleaned = clean_coordinates(raw_chunk)
        if cleaned.empty:
            processed_names.add(part_name)
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
            for h3_cell, source_id, dte in zip(
                flight_dedup["h3_cell"], flight_dedup["source_id"], flight_dedup["_date"]
            ):
                key = _source_idx(source_id) * _DATE_MULTIPLIER + (dte - _EPOCH).days
                sets_r.setdefault(h3_cell, set()).add(key)

            day_dedup = chunk.drop_duplicates(subset=["h3_cell", "_date"])
            days_r = hex_days[r]
            for h3_cell, date in zip(day_dedup["h3_cell"], day_dedup["_date"]):
                days_r.setdefault(h3_cell, set()).add(date)

        processed_names.add(part_name)
        parts_since_checkpoint += 1

        if (i + 1) % 200 == 0:
            logger.info(
                "  %d chunk (bu calistirmada), toplam %d parca islenmis, %d satir, "
                "res5 hex(point/flight)=%d/%d, %d benzersiz ucak",
                i + 1, len(processed_names), total_rows, len(point_counts[5]), len(flight_hex_sets[5]),
                len(source_id_to_idx),
            )

        if parts_since_checkpoint >= CHECKPOINT_EVERY_PARTS:
            _checkpoint_now()
            parts_since_checkpoint = 0

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

    # idx -> source_id: dict insertion sirasi = atanan idx sirasi (Python 3.7+
    # dict garantisi), bu yuzden liste indeksi dogrudan idx'e karsilik gelir.
    idx_to_source_id = list(source_id_to_idx.keys())
    is_military_by_idx = [military_lookup.get(sid, False) for sid in idx_to_source_id]

    for r in RESOLUTIONS:
        sets_r = flight_hex_sets[r]
        all_hexes = set(point_counts[r]) | set(sets_r) | set(hex_days[r])
        flight_count, flight_count_civil, flight_count_military = {}, {}, {}
        for h in all_hexes:
            entries = sets_r.get(h, ())
            mil = sum(1 for key in entries if is_military_by_idx[key // _DATE_MULTIPLIER])
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

    # Basariyla bitti -- checkpoint'e artik gerek yok, bir sonraki calistirma
    # yanlislikla eski (tamamlanmis) bir checkpoint'i devam ediyormus gibi
    # okumasin diye silinir.
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint silindi (is basariyla tamamlandi): %s", checkpoint_path)


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="ADS-B H3 yogunluk + askeri lookup uretimi")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Var olan checkpoint'i yok sayip sifirdan basla",
    )
    args = parser.parse_args()
    run(fresh=args.fresh)


if __name__ == "__main__":
    main()
