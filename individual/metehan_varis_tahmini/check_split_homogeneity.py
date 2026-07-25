"""build_split.py'nin RASTGELE (kronolojik olmayan) train/val/test ayrimi
gercekten zaman-homojen mi -- kullanici sorusu: "mevsim/saat farkli olabilir,
degisimler hep minimal oldugu icin (bkz. epoch-bazli kucuk iyilesmeler) iyi
karismis olmasi cok onemli."

training_sessions/*.parquet'te MUTLAK zaman damgasi YOK (sadece session-ici
GORELI dt_norm var) -- bu yuzden session_id'den (f"{object_name}::{local_sid}")
KAYNAK Silver parcasina geri donup GERCEK timestamp_utc okunuyor.

Tam populasyon (78.104 session, ~6.850 farkli parca) yeniden taramak
saatler surer -- bunun yerine session_split.parquet'teki BENZERSIZ
object_name'lerden RASTGELE bir ORNEKLEM (n=400 parca) okunup, o parcalardaki
TUM session'lar icin gercek baslangic zamani (ts_min) cikarilir, sonra
train/val/test arasinda ay / haftanin-gunu / gunun-saati dagilimi
karsilastirilir.
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"c:\Users\PC_6276\Desktop\github\AnomalyDetection")

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.common.minio_io import get_minio_client, read_parquet_object

SPLIT_PATH = Path(__file__).parent / "session_split.parquet"
RAW_CACHE_PATH = Path(__file__).parent / "split_homogeneity_raw_sample.parquet"
N_SAMPLE_PARTS = 1500
SESSION_GAP_MIN = 180  # build_training_dataset.py ile AYNI esik


def session_start_times(client, object_name: str) -> dict[int, pd.Timestamp]:
    """build_training_dataset.py'deki AYNI segmentasyon mantigini tekrarlayip
    (dropna, source_id+ts sirala, >180dk bosluk = yeni session, GLOBAL cumsum)
    her local session_id -> gercek baslangic zamani (ts_min) dondurur."""
    d = read_parquet_object(client, "silver", object_name, columns=["source_id", "timestamp_utc"])
    d = d.dropna(subset=["source_id", "timestamp_utc"])
    if d.empty:
        return {}
    d["ts"] = pd.to_datetime(d["timestamp_utc"], unit="s", utc=True)
    d = d.sort_values(["source_id", "ts"]).reset_index(drop=True)
    gap_min = d.groupby("source_id")["ts"].diff().dt.total_seconds() / 60
    new_session = gap_min.isna() | (gap_min > SESSION_GAP_MIN)
    d["session_id"] = new_session.astype("int64").cumsum()
    return d.groupby("session_id")["ts"].min().to_dict()


def main():
    if RAW_CACHE_PATH.exists():
        print(f"Onbellekten okunuyor: {RAW_CACHE_PATH}", flush=True)
        result = pd.read_parquet(RAW_CACHE_PATH)
        t0 = time.time()
    else:
        split_df = pd.read_parquet(SPLIT_PATH)
        split_df["object_name"] = split_df["session_id"].str.split("::").str[0]
        split_df["local_sid"] = split_df["session_id"].str.split("::").str[1].astype(int)

        all_objects = split_df["object_name"].unique().tolist()
        random.seed(42)
        sample_objects = random.sample(all_objects, min(N_SAMPLE_PARTS, len(all_objects)))
        print(f"Toplam benzersiz parca: {len(all_objects)}, orneklenen: {len(sample_objects)}", flush=True)

        client = get_minio_client()
        rows = []
        t0 = time.time()
        for i, obj in enumerate(sample_objects):
            try:
                starts = session_start_times(client, obj)
            except Exception as e:
                print(f"  atlandi ({obj}): {e}", flush=True)
                continue
            sub = split_df[split_df["object_name"] == obj]
            for _, r in sub.iterrows():
                ts = starts.get(r["local_sid"])
                if ts is not None:
                    rows.append({"split": r["split"], "ts": ts})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(sample_objects)} parca islendi ({time.time()-t0:.0f}sn)", flush=True)

        result = pd.DataFrame(rows)
        # HAM ornegi kaydet -- bir daha MinIO'yu yeniden taramadan (chi-kare
        # testi gibi) ek analiz yapilabilsin diye.
        result.to_parquet(RAW_CACHE_PATH, index=False)
        print(f"Ham ornek kaydedildi: {RAW_CACHE_PATH}", flush=True)

    print(f"\nToplam eslenen session: {len(result)} (train/val/test toplamindan)", flush=True)
    if result.empty:
        print("HATA: hic session eslenemedi.")
        return

    result["month"] = result["ts"].dt.month
    result["dow"] = result["ts"].dt.dayofweek  # 0=Pazartesi
    result["hour"] = result["ts"].dt.hour

    for col, label in [("month", "AY"), ("dow", "HAFTANIN GUNU (0=Pzt)"), ("hour", "SAAT")]:
        print(f"\n--- {label} dagilimi, split'e gore (%) ---")
        table = pd.crosstab(result[col], result["split"], normalize="columns") * 100
        print(table.round(1).to_string())

        # Ki-kare bagimsizlik testi: H0 = "split" ile bu zaman degiskeni
        # BAGIMSIZ (yani split'e gore dagilim FARKLI DEGIL -- tam da
        # istedigimiz "homojen" durum). p > 0.05 => H0 REDDEDILEMEDI, yani
        # gozle bakip "yakin" demek yerine somut bir istatistiksel dogrulama.
        counts = pd.crosstab(result[col], result["split"])
        chi2, p, dof, _ = chi2_contingency(counts)
        yorum = "H0 reddedilemedi -> homojenlikle TUTARLI" if p > 0.05 else "H0 REDDEDILDI -> anlamli fark var"
        print(f"  Ki-kare testi: chi2={chi2:.2f}, dof={dof}, p={p:.4f} -> {yorum}")

    print(f"\nToplam calisma suresi: {time.time()-t0:.0f}sn")


if __name__ == "__main__":
    main()
