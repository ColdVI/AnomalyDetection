"""25 haftalik TUM OpenSky arsivini (4 ornek gunun disindaki kalan gunler)
nadir-rota kesisim testinden gecirir -- opensky_rare_route_check.py'nin ayni
mantigi/sonuc dosyasi (opensky_rare_route_check_results.json), ama artik
yerel CPU egitimi Colab'a tasindigi icin (bkz. gunluk 2026-07-28) BelowNormal
kisitlamasina eskisi kadar GEREK YOK -- gunler PARALEL islenerek hizlandiriliyor
(ProcessPoolExecutor). Her gun bagimsiz (kendi .tar dosyalari, kendi
DataFrame'i), tek dosyaya YAZMA islemi sadece parent process'te yapiliyor
(race condition onlemek icin).

RAM UYARISI: bir gun (en buyugu ~50M satir) read_day_points() sirasinda
gecici olarak birkac GB kullanabiliyor -- bu makinede su an (2026-07-28)
sadece ~5GB bos RAM vardi (Docker Desktop + Chrome + Claude ~10.75/15.7GB
kullaniyordu). Bu yuzden varsayilan worker sayisi DUSUK (2) tutuldu --
--workers ile artirilabilir, ama once bellek-yogun uygulamalari (Docker
Desktop gibi) kapatmak daha guvenli olur.
"""
from __future__ import annotations

import argparse
import sys
sys.path.insert(0, r"c:\Users\PC_6276\Desktop\github\AnomalyDetection")

import json
from concurrent.futures import ProcessPoolExecutor, as_completed

from individual.metehan_varis_tahmini.opensky_comparison_test import process_day, STATES_DIR
from individual.metehan_varis_tahmini.opensky_rare_route_check import load_rare_routes, OUT_PATH


def all_downloaded_days() -> list[str]:
    return sorted(p.name for p in STATES_DIR.iterdir() if p.is_dir())


def main(n_workers: int = 2):
    rare_routes = load_rare_routes()
    print(f"Nadir rota sayisi (2-19 kez, adsb.lol'da): {len(rare_routes)}", flush=True)

    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {"days": {}}
    todo = [d for d in all_downloaded_days() if d not in results["days"]]
    total_target = len(results["days"]) + len(todo)
    print(f"Islenecek gun sayisi: {len(todo)} (zaten islenmis: {len(results['days'])}/{total_target}), "
          f"{n_workers} paralel worker ile", flush=True)

    if not todo:
        print("Islenecek yeni gun yok, dogrudan ozet uretiliyor.", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(process_day, day): day for day in todo}
        for fut in as_completed(futures):
            day = futures[fut]
            day_result = fut.result()
            results["days"][day] = day_result
            OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"[{day}] kaydedildi ({len(results['days'])}/{total_target})", flush=True)

    all_routes: dict[str, int] = {}
    for day, r in results["days"].items():
        for k, v in r["routes"].items():
            all_routes[k] = all_routes.get(k, 0) + v

    matched_rare = {k: v for k, v in all_routes.items() if k in rare_routes}
    total_extra_examples = sum(matched_rare.values())

    print("\n=== NADIR ROTA KESISIM SONUCU (TUM ARSIV) ===", flush=True)
    print(f"OpenSky'de gorulen farkli rota: {len(all_routes)}", flush=True)
    print(f"Bunlardan adsb.lol'un NADIR (2-19 kez) kumesiyle eslesen: {len(matched_rare)} "
          f"(nadir rotalarin %{100*len(matched_rare)/len(rare_routes):.2f}'i)", flush=True)
    print(f"Bu eslesen rotalar icin OpenSky'de toplam kac EK session var: {total_extra_examples}", flush=True)
    if matched_rare:
        print("\nEn cok ek session getiren ilk 30 rota (adsb.lol_kac_kez -> opensky_kac_ek):", flush=True)
        for k, v in sorted(matched_rare.items(), key=lambda kv: -kv[1])[:30]:
            print(f"  {k}: adsb.lol={rare_routes[k]}, opensky_ek={v}", flush=True)

    results["summary"] = {
        "n_rare_routes_adsblol": len(rare_routes),
        "n_opensky_routes_total": len(all_routes),
        "n_matched_rare": len(matched_rare),
        "total_extra_examples": total_extra_examples,
    }
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2,
                         help="Paralel islenecek gun sayisi (varsayilan 2 -- RAM kisitlamasina bkz. dosya basi)")
    args = parser.parse_args()
    main(n_workers=args.workers)
