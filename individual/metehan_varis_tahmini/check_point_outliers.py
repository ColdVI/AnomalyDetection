"""Nokta-seviyesinde outlier riski taramasi: egitim verisindeki (velocities/
alts/vrates/headings dizileri) fiziksel olarak imkansiz/asiri degerleri
olcerek bulur -- varsayimla degil (bkz. proje boyunca tekrar eden "olc,
varsayma" prensibi, ör. gunluk 2026-07-22/23).

Iki ayri sey raporlar:
  (1) SERT ESIK ihlalleri (TUM veri uzerinde, tam sayim) -- ör. negatif
      irtifa, saatte binlerce km hiz gibi acikca imkansiz degerler.
  (2) Persentil dagilimi (dosyalarin bir ORNEKLEMi uzerinden, bellek icin)
      -- gercek dagilimin nasil goruundugunu (p50/p95/p99/p99.9/max) gormek
      icin, esik degerlerinin MAKUL olup olmadigini degerlendirmeye yarar.

Yerel (Windows) ortamda training_sessions/ bos oldugu icin (sadece Drive'da
zip olarak var), bu script Colab'da (zaten acilmis training_sessions/
klasoruyle) calistirilmak uzere tasarlandi.
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["velocities", "alts", "vrates", "headings"]

# Fiziksel olarak MAKUL ust/alt sinirlar -- asiri gevsek tutuldu (yanlislikla
# gercek/nadir ama gecerli durumlari (ör. hizli inis, askeri ucak) elemesin
# diye), sadece ACIKCA imkansiz degerleri yakalamak icin.
HARD_BOUNDS = {
    "velocities": (0.0, 400.0),   # m/s -- 400 m/s = 1440 km/h, en hizli yolcu ucaklarinin bile USTUNDE
    "alts": (-500.0, 15000.0),    # metre -- deniz seviyesi altina kucuk tolerans, 15km ticari tavanin ustunde
    "vrates": (-50.0, 50.0),      # m/s dikey hiz -- ticari ucaklar nadiren 15-20 m/s'yi gecer
}


def main(sessions_dir: str, sample_every: int = 7):
    sessions_path = Path(sessions_dir)
    files = sorted(glob.glob(str(sessions_path / "*.parquet")))
    print(f"{len(files)} dosya bulundu.", flush=True)

    violations = {k: 0 for k in HARD_BOUNDS}
    totals = {k: 0 for k in HARD_BOUNDS}
    sample_values = {c: [] for c in COLS}

    t0 = time.time()
    for fi, f in enumerate(files, 1):
        df = pd.read_parquet(f, columns=COLS)
        take_sample = (fi % sample_every == 0)
        for col in COLS:
            for arr in df[col]:
                arr = np.asarray(arr, dtype=np.float64)
                if col in HARD_BOUNDS:
                    lo, hi = HARD_BOUNDS[col]
                    violations[col] += int(((arr < lo) | (arr > hi)).sum())
                    totals[col] += arr.shape[0]
                if take_sample:
                    sample_values[col].append(arr)
        if fi % 1000 == 0 or fi == len(files):
            print(f"  {fi}/{len(files)} dosya tarandi ({time.time()-t0:.1f}sn)", flush=True)

    print("\n=== SERT ESIK IHLALLERI (TUM veri, tam sayim) ===")
    for col, (lo, hi) in HARD_BOUNDS.items():
        n = totals[col]
        v = violations[col]
        print(f"  {col}: [{lo},{hi}] disinda {v}/{n} nokta (%{100*v/n:.5f})")

    print(f"\n=== PERSENTIL DAGILIMI (her {sample_every}. dosyadan ornekleme) ===")
    for col in COLS:
        arr = np.concatenate(sample_values[col]) if sample_values[col] else np.array([])
        print(f"\n{col} (n={len(arr)}):")
        for p in [0, 0.1, 1, 5, 50, 95, 99, 99.9, 100]:
            print(f"  p{p:<5}: {np.percentile(arr, p):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-dir", default="/content/training_sessions")
    parser.add_argument("--sample-every", type=int, default=7,
                         help="Persentil ornegi icin her N dosyadan biri kullanilir (bellek icin)")
    args = parser.parse_args()
    main(sessions_dir=args.sessions_dir, sample_every=args.sample_every)
