"""country_lookup.py -- Gold'daki adsb.lol kaynakli benzersiz ucaklarin
(source_id = ICAO24 hex) kayitli oldugu ulkeyi (ISO_A2 kodu) hesaplayip
diske onbelleklayen script.

NEDEN GEREKLI: export panosunun harita-tiklama ulke filtresi, her export
istegi icin milyarlarca satiri tek tek hex_country.lookup()'tan gecirmek
yerine, bu KUCUK (benzersiz ucak sayisi kadar, ~400 bin satir) onbellegi
kullanir -- filtreleme o zaman sadece bir sozluk/merge islemi olur.

2026-07-18 OPTIMIZASYON (artimlı tarama): Ilk versiyon HER calistirmada
Gold/unified'daki TUM parcalari (6861 parca) bastan taciyordu (~38 dk).
Artik hangi parcalarin ONCEDEN tarandigi bir manifest dosyasinda (SCANNED_
PARTS_PATH) tutuluyor -- bir sonraki calistirmada sadece YENI (manifestte
olmayan) parcalar okunuyor, cozulen hex->ulke eslemeleri MEVCUT onbellege
EKLENIYOR (bir hex'in ulkesi zamanla degismez, eski cozumler gecerliligini
korur). Ayrica artik source_id+source_type disinda hicbir kolon okunmuyor
(read_parquet_object'in columns= destegi) -- Gold satiri 10+ kolon
tasiyor, bu iki tanesi disinda hepsi gereksiz I/O idi.

Salt okunur: Gold'dan sadece source_id + source_type kolonlarini okur,
hicbir sey yazmaz (aircraft_country_lookup.parquet + manifest haric).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from src.common.hex_country import HexCountryLookup
from src.common.minio_io import ObjectStoreClient, get_minio_client, list_layer_objects, read_parquet_object

logger = logging.getLogger(__name__)

LOOKUP_PATH = Path("data/state/aircraft_country_lookup.parquet")
SCANNED_PARTS_PATH = Path("data/state/aircraft_country_lookup_scanned_parts.json")

# Sadece maskeleme icin gereken kolonlar -- Gold satirinin geri kalani
# (lat/lon/alt/velocity/heading/timestamp...) bu tarama icin gereksiz.
_SCAN_COLUMNS = ["source_id", "source_type"]

# Bolgesel filtre sadece adsb.lol kaynakli dataset'ler icin anlamli --
# ALFA/UAV Attack'in source_id'leri ICAO24 hex formatinda degil (sentetik/
# farkli bir kimlik semasi), hex_country lookup'i onlar icin GECERSIZ olur.
ADSBLOL_DATASETS = frozenset({"adsblol_historical", "adsblol_realtime"})


def _load_existing(lookup_path: Path, scanned_parts_path: Path) -> tuple[dict[str, str], set[str]]:
    hex_to_iso2: dict[str, str] = {}
    if lookup_path.exists():
        df = pd.read_parquet(lookup_path)
        hex_to_iso2 = dict(zip(df["source_id"], df["country_iso2"]))
    scanned: set[str] = set()
    if scanned_parts_path.exists():
        try:
            scanned = set(json.loads(scanned_parts_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            logger.warning("Taranmis-parca manifesti okunamadi (%s), sifirdan basliyor", scanned_parts_path)
    return hex_to_iso2, scanned


def build_country_lookup(
    client: ObjectStoreClient | None = None,
    *,
    lookup_path: Path = LOOKUP_PATH,
    scanned_parts_path: Path = SCANNED_PARTS_PATH,
    fresh: bool = False,
) -> pd.DataFrame:
    """Gold/unified'daki adsb.lol kaynakli parcalari tarayip benzersiz
    source_id kumesini toplar, HENUZ cozulmemis olanlari HexCountryLookup'tan
    gecirip (source_id, country_iso2) tablosunu diske yazar.

    `fresh=True`: manifesti ve mevcut onbellegi yok sayip sifirdan tarar
    (ör. hex->ulke eslemesinin kendisi -- ICAOHexRange.csv -- guncellenmisse)."""
    client = client or get_minio_client()
    gold_bucket = os.getenv("MINIO_GOLD_BUCKET", "gold")

    hex_to_iso2, scanned = ({}, set()) if fresh else _load_existing(lookup_path, scanned_parts_path)
    logger.info("Baslangic durumu: %d hex zaten cozulmus, %d parca zaten taranmis", len(hex_to_iso2), len(scanned))

    object_names = list_layer_objects(client, gold_bucket, "unified")
    new_names = [n for n in object_names if n not in scanned]
    logger.info("Gold/unified: %d parca toplam, %d parca YENI (taranacak)", len(object_names), len(new_names))

    unique_hex: set[str] = set()
    for i, name in enumerate(new_names):
        df = read_parquet_object(client, gold_bucket, name, columns=_SCAN_COLUMNS)
        subset = df[df["source_type"].isin(ADSBLOL_DATASETS)]
        unique_hex.update(subset["source_id"].dropna().unique().tolist())
        scanned.add(name)
        if (i + 1) % 500 == 0:
            logger.info("  %d/%d yeni parca tarandi, su ana kadar %d benzersiz hex", i + 1, len(new_names), len(unique_hex))

    new_hex = unique_hex - hex_to_iso2.keys()
    logger.info("Toplam %d benzersiz hex (bunlarin %d'i yeni), ulke cozumleniyor...", len(unique_hex), len(new_hex))

    lookup = HexCountryLookup()
    for hex_str in new_hex:
        iso2 = lookup.lookup_iso2(hex_str)
        if iso2:
            hex_to_iso2[hex_str] = iso2

    result = pd.DataFrame(
        [{"source_id": h, "country_iso2": c} for h, c in hex_to_iso2.items()]
    )
    lookup_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(lookup_path, index=False)
    scanned_parts_path.write_text(json.dumps(sorted(scanned)), encoding="utf-8")
    logger.info(
        "Ulke onbellegi yazildi: %s -- toplam %d hex bir ulkeye eslendi (%d ulke), %d parca manifestte",
        lookup_path, len(result), result["country_iso2"].nunique() if not result.empty else 0, len(scanned),
    )
    return result


def load_country_lookup(lookup_path: Path = LOOKUP_PATH) -> dict[str, str]:
    if not lookup_path.exists():
        raise FileNotFoundError(f"Ulke onbellegi bulunamadi: {lookup_path} -- once build_country_lookup() calistirilmali.")
    df = pd.read_parquet(lookup_path)
    return dict(zip(df["source_id"], df["country_iso2"]))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Gold adsb.lol ucak -> ulke onbellegini (yeniden) olustur")
    parser.add_argument("--fresh", action="store_true", help="Manifesti/onbellegi yok say, sifirdan tara")
    args = parser.parse_args()
    build_country_lookup(fresh=args.fresh)
