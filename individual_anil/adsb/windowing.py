"""Ucus-duzeyi feature tablosunu sabit uzunluklu pencerelere cevirir (model girdisi).

adsb.lol ornekleme araligi degisken (0.1 envanterinde gorulen: cogunlukla 0-20s, ama
uzun bosluklar da var) -- bu yuzden `windowing.py` (ML tarafinin ayni-isimli modulu
DEGIL, ADR kurali: kod kopyalanmaz) buyuk zaman-boslugu iceren pencereleri ATAR
(ucuslar/kopukluklar uzerinden ogrenme olmaz). Kayip degerler icin maske kanali
doner -- model imputed degeri "dogru tahmin etti" diye odullenmesin (ALFA/SEAD
dersi: bkz. archive/2026-07-10_legacy_non_adsb_ml/docs/ML1_BULGULAR_VE_HATALAR.md H1).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from adsb.truth import WINDOW_TRUTH_META_COLUMNS, summarize_window_truth

logger = logging.getLogger(__name__)


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    window: int,
    stride: int,
    max_gap_s: float,
    flight_id_col: str = "flight_id",
    time_col: str = "timestamp_utc",
    truth_architecture: str | None = None,
    forecast_target_rows: int | None = None,
):
    """Ucus-ici kayan pencereler uretir.

    Returns:
        X    : float32 (n, window, f) -- NaN'lar 0
        M    : float32 (n, window, f) -- 1=deger var, 0=eksik
        meta : DataFrame (n,) flight_id, t_start, t_end. ``truth_architecture``
               verilirse truth v2'den q_w/y_any/steady/history alanlari da eklenir.
    """
    xs, ms, metas = [], [], []
    for flight_id, g in df.groupby(flight_id_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        if len(g) < window:
            continue
        vals = g[feature_cols].to_numpy(dtype=np.float32)
        mask = np.isfinite(vals)
        t = g[time_col].to_numpy(dtype=float)

        for start in range(0, len(g) - window + 1, stride):
            end = start + window
            if np.diff(t[start:end]).max(initial=0.0) > max_gap_s:
                continue
            xs.append(np.nan_to_num(vals[start:end], nan=0.0))
            ms.append(mask[start:end].astype(np.float32))
            meta = {"flight_id": flight_id, "t_start": t[start], "t_end": t[end - 1]}
            if truth_architecture is not None:
                meta.update(
                    summarize_window_truth(
                        g.iloc[start:end],
                        architecture=truth_architecture,
                        forecast_target_rows=forecast_target_rows,
                    )
                )
            metas.append(meta)

    if not xs:
        f = len(feature_cols)
        return (
            np.zeros((0, window, f), np.float32),
            np.zeros((0, window, f), np.float32),
            pd.DataFrame(
                columns=["flight_id", "t_start", "t_end"]
                + (list(WINDOW_TRUTH_META_COLUMNS) if truth_architecture is not None else [])
            ),
        )

    X = np.stack(xs)
    M = np.stack(ms)
    meta = pd.DataFrame(metas)
    logger.info("pencere: %d adet (%d x %d)", len(X), window, len(feature_cols))
    return X, M, meta


def masked_mse(x: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Ornek-basi maskeli MSE, sekil (batch, window, features) -> (batch,)."""
    sq_err = (x - reconstruction) ** 2 * mask
    denom = mask.sum(dim=(-2, -1)).clamp(min=1.0)
    return sq_err.sum(dim=(-2, -1)) / denom


def masked_mse_per_channel(
    x: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kanal-bazinda ayristirilmis pay/payda -- (numerator, denominator), ikisi de
    (batch, features) sekilli. Yeniden-birlestirme ozdesligi: numerator.sum(-1) /
    denominator.sum(-1).clamp(min=1.0) == masked_mse(x, reconstruction, mask)."""
    sq_err = (x - reconstruction) ** 2 * mask
    numerator = sq_err.sum(dim=-2)
    denominator = mask.sum(dim=-2)
    return numerator, denominator
