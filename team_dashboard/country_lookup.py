"""country_lookup.py -- Gold'daki adsb.lol kaynakli benzersiz ucaklarin
(source_id = ICAO24 hex) kayitli oldugu ulkeyi (ISO_A2 kodu) bir kere
hesaplayip diske onbelleklayen script.

NEDEN GEREKLI: export panosunun harita-tiklama ulke filtresi, her export
istegi icin milyarlarca satiri tek tek hex_country.lookup()'tan gecirmek
yerine, bu KUCUK (benzersiz ucak sayisi kadar, ~400 bin satir) onbellegi
kullanir -- filtreleme o zaman sadece bir sozluk/merge islemi olur.

Salt okunur: Gold'dan sadece source_id + source_type kolonlarini okur,
hicbir sey yazmaz (aircraft_country_lookup.parquet haric).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from src.common.hex_country import HexCountryLookup
from src.common.minio_io import ObjectStoreClient, get_minio_client, list_layer_objects, read_parquet_object

logger = logging.getLogger(__name__)

LOOKUP_PATH = Path("data/state/aircraft_country_lookup.parquet")

# Bolgesel filtre sadece adsb.lol kaynakli dataset'ler icin anlamli --
# ALFA/UAV Attack'in source_id'leri ICAO24 hex formatinda degil (sentetik/
# farkli bir kimlik semasi), hex_country lookup'i onlar icin GECERSIZ olur.
ADSBLOL_DATASETS = frozenset({"adsblol_historical", "adsblol_realtime"})


def build_country_lookup(
    client: ObjectStoreClient | None = None, *, lookup_path: Path = LOOKUP_PATH
) -> pd.DataFrame:
    """Gold/unified'daki adsb.lol kaynakli TUM parcalari tarayip benzersiz
    source_id kumesini toplar, her birini BIR KERE HexCountryLookup'tan
    gecirip (source_id, country_iso2) tablosunu diske yazar."""
    client = client or get_minio_client()
    gold_bucket = os.getenv("MINIO_GOLD_BUCKET", "gold")

    object_names = list_layer_objects(client, gold_bucket, "unified")
    logger.info("Gold/unified: %d parca taranacak (sadece adsb.lol kaynaklari)", len(object_names))

    # _iter_part_stats (layer_index.py) sadece timestamp_utc+source_type
    # okur -- burada source_id de gerektigi icin kendi tarama dongumuzu
    # yaziyoruz (read_parquet_object kolon secimi desteklemiyor, Gold zaten
    # sadece 10 kolonlu oldugu icin tum satiri okumak onemli bir maliyet
    # degil).
    unique_hex: set[str] = set()
    for i, name in enumerate(object_names):
        df = read_parquet_object(client, gold_bucket, name)
        subset = df[df["source_type"].isin(ADSBLOL_DATASETS)]
        unique_hex.update(subset["source_id"].dropna().unique().tolist())
        if (i + 1) % 500 == 0:
            logger.info("  %d/%d parca tarandi, su ana kadar %d benzersiz hex", i + 1, len(object_names), len(unique_hex))

    logger.info("Toplam %d benzersiz adsb.lol ucak (hex) bulundu, ulke cozumleniyor...", len(unique_hex))

    lookup = HexCountryLookup()
    rows = []
    for hex_str in unique_hex:
        iso2 = lookup.lookup_iso2(hex_str)
        if iso2:
            rows.append({"source_id": hex_str, "country_iso2": iso2})

    result = pd.DataFrame(rows)
    lookup_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(lookup_path, index=False)
    logger.info(
        "Ulke onbellegi yazildi: %s -- %d/%d hex bir ulkeye eslendi (%d ulke)",
        lookup_path, len(result), len(unique_hex), result["country_iso2"].nunique() if not result.empty else 0,
    )
    return result


def load_country_lookup(lookup_path: Path = LOOKUP_PATH) -> dict[str, str]:
    if not lookup_path.exists():
        raise FileNotFoundError(f"Ulke onbellegi bulunamadi: {lookup_path} -- once build_country_lookup() calistirilmali.")
    df = pd.read_parquet(lookup_path)
    return dict(zip(df["source_id"], df["country_iso2"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_country_lookup()
