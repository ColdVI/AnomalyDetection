"""Pencereleme (ML-2 fazi): feature tablosu -> sabit uzunluklu sequence'lar.

LSTM-AE sabit araliklı pencere ister. Veri zaten ~sabit hizda (ALFA ~4 Hz,
UAV/SEAD ~5 Hz); satir-bazli pencere kullanilir ve buyuk zaman bosluğu iceren
pencereler ATILIR (ucuslar arasi/kayit kopmasi uzerinden ogrenme olmaz).
Interpolasyon yok: kisa bosluklari zaten merge_asof toleransi emdi, uzun
bosluk pencereyi gecersiz kilar (docs/ANOMALI_METOT_ARASTIRMASI.md, H1-impute dersi).

Pencere etiketi: icindeki HERHANGI bir satir anomali etiketliyse pencere anomali
(degerlendirme icin; egitim zaten yalnizca normal ucuslardan pencere alir).

Maske kanali: NaN'lar 0 ile doldurulur ve her feature icin is_missing maskesi
dondurulur -- model imputed degeri "dogru tahmin etti" diye odullenmesin diye
kayip maske-agirlikli hesaplanabilir (H1 dersinin dogrudan uygulamasi).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NORMALS = {"normal", "benign"}


def build_windows(df: pd.DataFrame, feature_cols: list[str], *,
                  window: int, stride: int, max_gap_s: float,
                  time_col: str = "t_rel_s"):
    """Ucus-ici kayan pencereler uretir.

    Returns:
        X    : float32 (n, window, f) -- NaN'lar 0
        M    : float32 (n, window, f) -- 1=deger var, 0=eksik
        meta : DataFrame (n,) source_id, t_start, t_end, is_anomaly, label
    """
    xs, ms, metas = [], [], []
    for source_id, g in df.groupby("source_id", sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        vals = g[feature_cols].to_numpy(dtype=np.float32)
        mask = np.isfinite(vals)
        t = g[time_col].to_numpy(dtype=float)
        anom = (~g["label"].isin(NORMALS)).to_numpy()
        labels = g["label"].to_numpy()
        for start in range(0, len(g) - window + 1, stride):
            end = start + window
            # pencere icinde kayit kopmasi varsa atla
            if np.diff(t[start:end]).max(initial=0.0) > max_gap_s:
                continue
            xs.append(np.nan_to_num(vals[start:end], nan=0.0))
            ms.append(mask[start:end].astype(np.float32))
            win_anom = bool(anom[start:end].any())
            win_label = (pd.Series(labels[start:end])[anom[start:end]].mode().iloc[0]
                         if win_anom else labels[start])
            metas.append({"source_id": source_id, "t_start": t[start], "t_end": t[end - 1],
                          "is_anomaly": win_anom, "label": win_label})
    if not xs:
        f = len(feature_cols)
        return (np.zeros((0, window, f), np.float32), np.zeros((0, window, f), np.float32),
                pd.DataFrame(columns=["source_id", "t_start", "t_end", "is_anomaly", "label"]))
    X = np.stack(xs)
    M = np.stack(ms)
    meta = pd.DataFrame(metas)
    logger.info("pencere: %d adet (%d x %d), anomali orani %.2f",
                len(X), window, len(feature_cols), meta["is_anomaly"].mean())
    return X, M, meta
