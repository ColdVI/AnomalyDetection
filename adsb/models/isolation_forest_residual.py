"""isolation_forest_contextual_v1 -- paralel kesif (bkz. docs/adsb_isolation_forest_
contextual_v1_prereg_2026-07-14.md). contextual_physics_v1'in AYNI 5 residual kanalini
ve AYNI StrictNaturalRobustScaler'ini kullanir, ama zamansal gecmise degil COK-
DEGISKENLI izolasyona bakar -- farkli bir kor-nokta profili beklenir.

Yalniz natural_clean_fit satirlariyla fit edilir (StrictNaturalRobustScaler bunu
calisma zamaninda zorlar). Sentetik veri hicbir zaman fit'e girmez, yalniz ayri
degerlendirmede (anomaly-development rolu) kullanilabilir.

Bilinen sinirlama: LSTM tarafinin availability mask'i burada yok -- aktif 5 kanaldan
herhangi biri NaN olan satirlar TAMAMEN atilir (complete-case). Eksik veri deseni zaten
ayri S2 katmaninda yakalaniyor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from adsb.contextual_scaling import StrictNaturalRobustScaler

RESIDUAL_CHANNELS: tuple[str, ...] = (
    "speed_residual",
    "vertical_rate_residual",
    "heading_residual",
    "east_velocity_residual",
    "north_velocity_residual",
)

# Sonuc gorulmeden dondurulmus hiperparametreler (prereg'e bkz.) -- contamination
# yalniz sklearn'in ic predict() ofsetini etkiler, predict() hicbir yerde kullanilmiyor.
N_ESTIMATORS = 200
MAX_SAMPLES = "auto"
CONTAMINATION = "auto"
RANDOM_STATE = 0


def _complete_case_matrix(frame: pd.DataFrame, channels: tuple[str, ...]) -> tuple[np.ndarray, pd.Index]:
    """Aktif kanallarin TAMAMI sonlu olan satirlari secer -- IF NaN kabul etmez."""
    sub = frame.loc[:, list(channels)].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(sub.to_numpy(dtype=float)).all(axis=1)
    kept = sub.loc[finite_mask]
    return kept.to_numpy(dtype=float), kept.index


def fit_isolation_forest_residual(
    natural_fit_frame: pd.DataFrame,
    *,
    channels: tuple[str, ...] = RESIDUAL_CHANNELS,
    contains_synthetic: bool = False,
    clip: float = 5.0,
) -> tuple[IsolationForest, StrictNaturalRobustScaler, tuple[str, ...]]:
    """natural_clean_fit rolundeki satirlardan olcekleyici + Isolation Forest fit eder.

    Doner: (model, fitted_scaler, active_channels) -- active_channels, MAD=0 nedeniyle
    dislanan kanallar cikarilmis nihai listedir (contextual_physics_v1'deki
    altitude_source_residual disliamasiyla ayni disiplin).
    """
    from adsb.contextual_scaling import NATURAL_FIT_ROLE, StrictScalingConfig

    scaler = StrictNaturalRobustScaler(StrictScalingConfig(clip=clip)).fit(
        natural_fit_frame, channels, data_role=NATURAL_FIT_ROLE, contains_synthetic=contains_synthetic
    )
    active_channels = scaler.active_channels

    scaled = scaler.transform(natural_fit_frame)
    X, _ = _complete_case_matrix(scaled, active_channels)
    if len(X) == 0:
        raise ValueError("Fit icin tam-durumlu (NaN'siz) hicbir satir kalmadi")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION, random_state=RANDOM_STATE,
    ).fit(X)
    return model, scaler, active_channels


def score_isolation_forest_residual(
    model: IsolationForest, scaler: StrictNaturalRobustScaler, frame: pd.DataFrame,
) -> pd.Series:
    """Satir-bazi anomali skoru dondurur: BUYUK = daha anormal (proje-geneli sozlesme).

    sklearn'in score_samples() 'buyuk = daha normal' doner -- burada isareti ceviriyoruz.
    NaN kanalli satirlar skorlanamiyor, NaN olarak dondurulur (dusurulmuyor -- caginin
    orijinal index'iyle hizali kalir).
    """
    active_channels = scaler.active_channels
    scaled = scaler.transform(frame)
    X, kept_index = _complete_case_matrix(scaled, active_channels)

    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    if len(X) > 0:
        scores.loc[kept_index] = -model.score_samples(X)  # isaret cevrildi: buyuk=anormal
    return scores
