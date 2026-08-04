"""OpenSky 2022 verisinde, adsb.lol'daki NADIR rotalarimizla (2-19 kez
uculmus, session_split.parquet'ten) eslesen session var mi diye kontrol eder
-- amac bu rotalari GERCEKTEN zenginlestirebilir miyiz sorusuna varsayimla
degil olcerek cevap vermek (bkz. gunluk 2026-07-27, kullanicinin "seyrek
rotalari soyle, opensky'a bakalim" sorusu).

opensky_comparison_test.py ile AYNI metodoloji (session segmentasyonu +
gercek bolge testi + en-yakin-havalimani heuristigi) -- sadece sonucta
rota TUM adsb.lol rotalariyla degil, SADECE nadir-rota kumesiyle
kesistiriliyor.

Ayni sekilde dusuk oncelikli (BelowNormal) calistirilmali -- arka planda
suren Focal Loss egitimini yavaslatmasin diye.
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"c:\Users\PC_6276\Desktop\github\AnomalyDetection")

import json
from pathlib import Path

import pandas as pd

from individual.metehan_varis_tahmini.opensky_comparison_test import (
    process_day, SAMPLE_DAYS,
)

OUT_PATH = Path(__file__).parent / "opensky_rare_route_check_results.json"


def load_rare_routes() -> dict[str, int]:
    """session_split.parquet'ten 2-19 kez uculmus rotalarin (origin||dest -> kac kez) haritasi."""
    split_df = pd.read_parquet(Path(__file__).parent / "session_split.parquet")
    counts = split_df.groupby(["origin_airport", "dest_airport"]).size()
    rare = counts[(counts >= 2) & (counts <= 19)]
    return {f"{o}||{d}": int(v) for (o, d), v in rare.items()}


def main():
    rare_routes = load_rare_routes()
    print(f"Nadir rota sayisi (2-19 kez, adsb.lol'da): {len(rare_routes)}", flush=True)

    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())
    else:
        results = {"days": {}}

    for day in SAMPLE_DAYS:
        if day in results["days"]:
            print(f"[{day}] zaten islenmis, atlaniyor", flush=True)
            continue
        day_result = process_day(day)
        results["days"][day] = day_result
        OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    all_routes: dict[str, int] = {}
    for day, r in results["days"].items():
        for k, v in r["routes"].items():
            all_routes[k] = all_routes.get(k, 0) + v

    matched_rare = {k: v for k, v in all_routes.items() if k in rare_routes}
    total_extra_examples = sum(matched_rare.values())

    print("\n=== NADIR ROTA KESISIM SONUCU ===", flush=True)
    print(f"OpenSky'de gorulen farkli rota: {len(all_routes)}", flush=True)
    print(f"Bunlardan adsb.lol'un NADIR (2-19 kez) kumesiyle eslesen: {len(matched_rare)} "
          f"(nadir rotalarin %{100*len(matched_rare)/len(rare_routes):.2f}'i)", flush=True)
    print(f"Bu eslesen rotalar icin OpenSky'de toplam kac EK session var: {total_extra_examples}", flush=True)
    if matched_rare:
        print("\nEn cok ek session getiren ilk 20 rota (adsb.lol_kac_kez -> opensky_kac_ek):", flush=True)
        for k, v in sorted(matched_rare.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {k}: adsb.lol={rare_routes[k]}, opensky_ek={v}", flush=True)

    results["summary"] = {
        "n_rare_routes_adsblol": len(rare_routes),
        "n_opensky_routes_total": len(all_routes),
        "n_matched_rare": len(matched_rare),
        "total_extra_examples": total_extra_examples,
    }
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
