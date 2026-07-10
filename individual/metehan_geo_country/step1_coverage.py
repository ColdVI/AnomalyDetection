"""step1_coverage.py -- Adim 1: hex -> ulke coverage testi.

country_heatmap_prompt.md Adim 1: "Bizim veri setimizdeki hex degerlerinin
kacinin bu tabloyla eslestigini (coverage oranini) test et".

Kullanim:
    python -m individual.metehan_geo_country.step1_coverage
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from individual.metehan_geo_country.hex_country import HexCountryLookup

DATA_DIR = Path(__file__).parent / "data"
AIRCRAFT_CSV = DATA_DIR / "aircraft_dump_20260707_181141.csv"


def main() -> None:
    df = pd.read_csv(AIRCRAFT_CSV)
    print(f"Toplam satir: {len(df):,}")

    unique_hex = df["hex"].dropna().unique()
    print(f"Benzersiz hex: {len(unique_hex):,}")

    lookup = HexCountryLookup()
    results = {h: lookup.lookup(h) for h in unique_hex}

    hex_country = pd.DataFrame(
        [(h, country, category) for h, (country, category) in results.items()],
        columns=["hex", "country", "category"],
    )

    counts = hex_country["category"].value_counts()
    total = len(hex_country)
    print("\n--- Benzersiz HEX bazinda kapsama ---")
    for cat in ("country", "reserved", "no_match"):
        n = counts.get(cat, 0)
        print(f"  {cat:10s}: {n:6,} ({100 * n / total:5.1f}%)")

    # Satir (row) bazinda da bak -- bazi ucaklar cok daha fazla trace
    # birakmis olabilir, hex-bazinda %90 olsa da satir-bazinda farkli
    # cikabilir (agirlikli coverage).
    merged = df.merge(hex_country, on="hex", how="left")
    row_counts = merged["category"].value_counts()
    total_rows = len(merged)
    print("\n--- SATIR bazinda kapsama (agirlikli) ---")
    for cat in ("country", "reserved", "no_match"):
        n = row_counts.get(cat, 0)
        print(f"  {cat:10s}: {n:9,} ({100 * n / total_rows:5.1f}%)")

    print("\n--- En cok gorulen 15 ulke (benzersiz hex sayisina gore) ---")
    top_countries = (
        hex_country[hex_country["category"] == "country"]["country"]
        .value_counts()
        .head(15)
    )
    for country, n in top_countries.items():
        print(f"  {country:30s} {n:5,} hex")

    no_match = hex_country[hex_country["category"] == "no_match"]["hex"].tolist()
    if no_match:
        print(f"\n--- Eslesmeyen (no_match) hex ornekleri (ilk 10/{len(no_match)}) ---")
        for h in no_match[:10]:
            print(f"  {h}")

    reserved = hex_country[hex_country["category"] == "reserved"]["hex"].tolist()
    if reserved:
        print(f"\n--- Rezerve/tahsis-edilmemis (reserved) hex ornekleri (ilk 10/{len(reserved)}) ---")
        for h in reserved[:10]:
            print(f"  {h} -> {lookup.lookup(h)}")

    out_path = DATA_DIR / "hex_country_lookup.csv"
    hex_country.to_csv(out_path, index=False)
    print(f"\nYazildi: {out_path} ({len(hex_country):,} satir)")


if __name__ == "__main__":
    main()
