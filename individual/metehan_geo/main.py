"""main.py -- Bolum 5'i uctan uca calistiran CLI (docs/BIREYSEL_PROJE_MASTER (1).md).

Su an icin Adim 1-8 + 6.1 (yogunluk haritasi + kumeleme icin ornekleme).
Kumeleme (Adim 9-13) ayri bir asamada eklenecek -- bu script'in urettigi
sample parquet'i onun girdisi olacak.

Tek streaming gecis: load_adsb_gold_data() 2.500 Gold parcasini bir kere
okur, ayni dongude hem hex yogunlugu (compute_hex_density) hem kumeleme
ornegi (reservoir_sample_chunks) biriktirilir -- veri iki kere okunmaz.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from individual.metehan_geo.data import clean_coordinates, load_adsb_gold_data
from individual.metehan_geo.geo import assign_h3_cell, h3_cell_to_polygon
from individual.metehan_geo.viz import build_density_geojson, save_geojson

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "viz" / "data"


def run(h3_resolution: int, sample_max_points: int, seed: int = 42) -> None:
    density_counts: Counter[str] = Counter()
    rng = np.random.default_rng(seed)
    sample_parts: list[pd.DataFrame] = []
    sample_size = 0
    total_rows = 0
    total_dropped = 0

    for i, raw_chunk in enumerate(load_adsb_gold_data()):
        cleaned = clean_coordinates(raw_chunk)
        total_dropped += len(raw_chunk) - len(cleaned)
        if cleaned.empty:
            continue
        chunk = assign_h3_cell(cleaned, h3_resolution)
        total_rows += len(chunk)

        vc = chunk["h3_cell"].value_counts()
        density_counts.update(vc.to_dict())

        n = len(chunk)
        take = min(n, sample_max_points)
        idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        sample_parts.append(chunk.iloc[idx])
        sample_size += take
        if sample_size > sample_max_points * 2:
            combined = pd.concat(sample_parts, ignore_index=True)
            if len(combined) > sample_max_points:
                keep = rng.choice(len(combined), size=sample_max_points, replace=False)
                combined = combined.iloc[keep]
            sample_parts = [combined]
            sample_size = len(combined)

        if (i + 1) % 200 == 0:
            logger.info(
                "  %d chunk islendi, %d satir, %d benzersiz hex, %d ornek noktasi",
                i + 1, total_rows, len(density_counts), sample_size,
            )

    logger.info(
        "Tamamlandi: %d satir (%d dusuruldu), %d benzersiz hex, %d ornek noktasi",
        total_rows, total_dropped, len(density_counts), sample_size,
    )

    density_df = pd.DataFrame({
        "h3_cell": list(density_counts.keys()),
        "point_count": list(density_counts.values()),
    })
    density_geojson = build_density_geojson(density_df)
    save_geojson(density_geojson, OUT_DIR / "density.geojson")
    density_df.to_parquet(OUT_DIR / "density.parquet", index=False)

    sample_df = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    if len(sample_df) > sample_max_points:
        keep = rng.choice(len(sample_df), size=sample_max_points, replace=False)
        sample_df = sample_df.iloc[keep].reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(OUT_DIR / "cluster_sample.parquet", index=False)
    logger.info("Kayit: %s (density), %s (ornek, %d satir)", OUT_DIR / "density.geojson", OUT_DIR / "cluster_sample.parquet", len(sample_df))


def main() -> None:
    parser = argparse.ArgumentParser(description="FAZ A: yogunluk haritasi + kumeleme ornegi")
    parser.add_argument("--h3-resolution", type=int, default=5)
    parser.add_argument("--sample-max-points", type=int, default=2_000_000)
    args = parser.parse_args()

    run(args.h3_resolution, args.sample_max_points)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
