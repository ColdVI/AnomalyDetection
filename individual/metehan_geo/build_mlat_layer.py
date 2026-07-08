"""build_mlat_layer.py -- MLAT (multilateration) kapsama katmani.

Kullanicinin onerisi: adsb.lol'un ham trace formatinda her nokta icin bir
kaynak tipi var (Silver'da `ads_source_type` -- adsb_icao, mlat, tisb_icao vb).
MLAT noktalari, tek istasyonun sinyali dogrudan alamadigi ama birden fazla yer
istasyonunun ucgenleme yaptigi durumlarda uretiliyor -- kiyi/ada istasyon
kumeleri civarinda kapsamayi biraz genisletebiliyor (acik okyanusta MLAT da
calismaz, senkronize istasyon yok).

Bu alan GOLD'da YOK (Gold sadece ortak 7+3 semayi tasir, kaynaga-ozel
kolonlari duser -- ADR-003). Bu yuzden burada Gold degil, SILVER'in
adsblol_historical prefix'i DOGRUDAN taranir -- docs/BIREYSEL_PROJE_MASTER (1).md'nin
"sadece Gold'dan oku" varsayimina bilinen/belgelenen tek istisna.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import h3
import pandas as pd

from src.common.minio_io import get_minio_client, list_layer_objects, read_parquet_object

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "viz" / "data"


def run(h3_resolution: int = 5) -> None:
    client = get_minio_client()
    names = list_layer_objects(client, "silver", "adsblol_historical")
    logger.info("Silver'da %d parca, ads_source_type taraniyor (Gold'da olmayan alan)", len(names))

    total_counts: Counter[str] = Counter()
    mlat_counts: Counter[str] = Counter()
    total_rows = 0
    total_mlat = 0

    for i, name in enumerate(names):
        df = read_parquet_object(client, "silver", name)
        df = df.dropna(subset=["lat", "lon"])
        df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
        if df.empty:
            continue

        cells = [h3.latlng_to_cell(lat, lon, h3_resolution) for lat, lon in zip(df["lat"], df["lon"])]
        df = df.assign(h3_cell=cells)

        total_counts.update(df["h3_cell"].value_counts().to_dict())
        total_rows += len(df)

        mlat_rows = df[df["ads_source_type"] == "mlat"]
        if not mlat_rows.empty:
            mlat_counts.update(mlat_rows["h3_cell"].value_counts().to_dict())
            total_mlat += len(mlat_rows)

        if (i + 1) % 200 == 0:
            logger.info(
                "  %d/%d parca, %d satir, %d MLAT satiri (%.2f%%)",
                i + 1, len(names), total_rows, total_mlat, 100 * total_mlat / max(total_rows, 1),
            )

    logger.info(
        "Bitti: %d satir, %d MLAT satiri (%.2f%%), %d hex'te MLAT var",
        total_rows, total_mlat, 100 * total_mlat / max(total_rows, 1), len(mlat_counts),
    )

    # Her hex icin: toplam nokta + mlat nokta + mlat orani (bu bolgenin
    # kapsaminin ne kadari MLAT'a bagli -- yuksek oran = tek-istasyon
    # kapsaminin sinirinda/disinda, dusuk oran = doğrudan ADS-B baskin).
    rows = []
    for h3_cell, mlat_count in mlat_counts.items():
        total = total_counts.get(h3_cell, mlat_count)
        rows.append({
            "h3_cell": h3_cell,
            "mlat_count": mlat_count,
            "total_count": total,
            "mlat_ratio": round(mlat_count / total, 4) if total else 0.0,
        })
    mlat_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mlat_df.to_parquet(OUT_DIR / "mlat_density.parquet", index=False)

    from individual.metehan_geo.geo import h3_cell_to_polygon

    features = []
    for row in mlat_df.itertuples(index=False):
        ring = h3_cell_to_polygon(row.h3_cell)
        if ring is None:
            continue  # kutup-noktasi hucresi, render edilemiyor (bkz. h3_cell_to_polygon)
        features.append({
            "type": "Feature",
            "properties": {
                "h3_cell": row.h3_cell,
                "mlat_count": int(row.mlat_count),
                "total_count": int(row.total_count),
                "mlat_ratio": float(row.mlat_ratio),
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    geojson = {"type": "FeatureCollection", "features": features}

    import json
    with open(OUT_DIR / "mlat_density.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson, f)
    logger.info("Yazildi: %s (%d hex, MLAT verisi olan)", OUT_DIR / "mlat_density.geojson", len(features))


def main() -> None:
    parser = argparse.ArgumentParser(description="MLAT kapsama katmani (Silver'dan dogrudan)")
    parser.add_argument("--h3-resolution", type=int, default=5)
    args = parser.parse_args()
    run(args.h3_resolution)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
