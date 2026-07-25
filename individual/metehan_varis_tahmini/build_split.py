"""training_sessions/*.parquet uzerinde SESSION bazli train/val/test ayrimi
kurar (satir/kesme-noktasi bazinda DEGIL) -- aksi halde ayni ucusun farkli
kesmeleri train ve test'e dagilip veri sizintisi (data leakage) yaratir.

Ayrica 1-KEZ uçulan rotalari eleyip sadece >1 kez uçulan rotalarin session'larini
tutar (bkz. gunluk: rota-tekrari kararı).

Cikti: session_split.parquet -- kolonlar: session_id, origin_airport,
dest_airport, split ("train"/"val"/"test").
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

SESSIONS_DIR = Path(__file__).parent / "training_sessions"
OUT_PATH = Path(__file__).parent / "session_split.parquet"

TRAIN_FRAC, VAL_FRAC = 0.8, 0.1  # kalan 0.1 test


def main():
    files = sorted(glob.glob(str(SESSIONS_DIR / "*.parquet")))
    print(f"{len(files)} parquet dosyasi taraniyor (sadece metadata kolonlari)...")

    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["session_id", "origin_airport", "dest_airport"])
        frames.append(df.drop_duplicates(subset=["session_id"]))
    all_sessions = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["session_id"])
    print(f"Toplam benzersiz session: {len(all_sessions)}")

    # 1-kez uçulan rotalari ele -- eğitim etiketi olarak anlamsiz
    route_counts = all_sessions.groupby(["origin_airport", "dest_airport"]).size()
    all_sessions["route_count"] = all_sessions.set_index(["origin_airport", "dest_airport"]).index.map(route_counts)
    kept = all_sessions[all_sessions["route_count"] > 1].copy()
    print(f"1-kez uçulan rotalar elendi: {len(all_sessions)} -> {len(kept)} session "
          f"(%{100*len(kept)/len(all_sessions):.1f} korunuyor)")

    # session bazinda rastgele (deterministik seed) train/val/test
    shuffled = kept.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val))
    shuffled["split"] = split

    print(f"Split: train={n_train} ({100*n_train/n:.1f}%), val={n_val} ({100*n_val/n:.1f}%), "
          f"test={n - n_train - n_val} ({100*(n - n_train - n_val)/n:.1f}%)")

    shuffled[["session_id", "origin_airport", "dest_airport", "split"]].to_parquet(OUT_PATH, index=False)
    print(f"Yazildi: {OUT_PATH}")


if __name__ == "__main__":
    main()
