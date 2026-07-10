"""Sentetik ariza enjeksiyonu (ML-3 fazi) -- FDI literaturunun standart degerlendirme yontemi.

Kaggle-tarzi "tum veri sentetik" YAKLASIMI DEGIL: gercek normal ucus verisinin uzerine
kontrollu, fiziksel olarak anlamli ariza enjekte edilir. Kapsam tablosundaki gercek-veri
boslugu olan aileleri (IMU/sensor donmasi, bias/drift, gurultu artisi, stealthy GPS
spoofing, telemetri dropout) test tarafinda kapatir.

KRITIK KURAL: enjekte edilen ucuslar ASLA egitime girmez -- yalnizca test. Novelty
detection paradigmasi bozulmaz; sadece degerlendirme yuzeyi genisler. Rapor dili:
"gercek arizalar + fiziksel olarak modellenen enjekte arizalarla degerlendirildi".

Enjeksiyonlar SILVER duzeyinde (ham sinyal) yapilir, feature'lar enjekte edilmis
silver'dan yeniden uretilir -- boylece rolling/CUSUM/residual feature'lari arizayi
gercek bir arizada oldugu gibi "gorur".

Her fonksiyon: (df, onset_frac, rng) -> (df_bozuk, onset_t) . Ucus zaman-sirali tek
ucustur; onset_frac ucusun yuzde kacinda arizanin basladigi. label kolonu onset
sonrasi "inj_<tip>" yapilir (degerlendirme icin; egitim zaten gormez).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ~111.32 km / derece (enlem); kucuk aci yaklasimi stealthy ramp icin yeterli
_DEG_PER_M_LAT = 1.0 / 111_320.0


def _onset_index(df: pd.DataFrame, onset_frac: float) -> int:
    return int(len(df) * onset_frac)


def _mark(df: pd.DataFrame, i0: int, tag: str) -> pd.DataFrame:
    df = df.copy()
    df.iloc[i0:, df.columns.get_loc("label")] = f"inj_{tag}"
    return df


def inject_freeze(df: pd.DataFrame, col: str, *, onset_frac: float = 0.5, rng=None) -> pd.DataFrame:
    """Sensor donmasi: onset'ten itibaren kolon son gecerli degerinde sabit kalir."""
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    out.iloc[i0:, out.columns.get_loc(col)] = out[col].iloc[max(i0 - 1, 0)]
    return _mark(out, i0, "freeze")


def inject_bias(df: pd.DataFrame, col: str, *, sigma_mult: float = 4.0,
                onset_frac: float = 0.5, rng=None) -> pd.DataFrame:
    """Ani bias: onset'ten itibaren kolona +k*sigma eklenir (adim degisimi)."""
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    sigma = float(out[col].std()) or 1.0
    out.iloc[i0:, out.columns.get_loc(col)] = out[col].iloc[i0:] + sigma_mult * sigma
    return _mark(out, i0, "bias")


def inject_drift(df: pd.DataFrame, col: str, time_col: str, *, sigma_per_min: float = 3.0,
                 onset_frac: float = 0.5, rng=None) -> pd.DataFrame:
    """Yavas drift: onset'ten itibaren dogrusal buyuyen sapma (dakikada k*sigma)."""
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    sigma = float(out[col].std()) or 1.0
    t = out[time_col].astype(float)
    ramp = (t - t.iloc[i0]).clip(lower=0) / 60.0 * sigma_per_min * sigma
    out[col] = out[col] + ramp
    return _mark(out, i0, "drift")


def inject_noise(df: pd.DataFrame, col: str, *, sigma_mult: float = 3.0,
                 onset_frac: float = 0.5, rng=None) -> pd.DataFrame:
    """Gurultu artisi: onset sonrasi kolona k*sigma olcekli Gauss gurultusu."""
    rng = rng or np.random.default_rng(0)
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    sigma = float(out[col].std()) or 1.0
    n = len(out) - i0
    out.iloc[i0:, out.columns.get_loc(col)] = (
        out[col].iloc[i0:] + rng.normal(0, sigma_mult * sigma, n))
    return _mark(out, i0, "noise")


def inject_gps_ramp(df: pd.DataFrame, *, meters_per_s: float = 2.0,
                    onset_frac: float = 0.5, time_col: str = "timestamp",
                    time_scale: float = 1e-6, rng=None) -> pd.DataFrame:
    """Stealthy GPS spoofing: konuma saniyede birkac metre buyuyen kuzey-yonlu ramp.

    UAV Attack'taki kaba (78 km sicramali) SITL spoofing'in yakalayamadigi senaryo:
    sicrama yok, kademeli sapma var. Literaturdeki slow-drift spoofing'in birebir
    simulasyonu. Receiver hizi (vel_m_s) DEGISTIRILMEZ -- gercek spoofing'de receiver
    sahte konumu izler ama IMU/hiz tutarsizligi dogar; bizim gps_speed_residual
    feature'inin tam olarak yakalamasi gereken durum.
    """
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    t = out[time_col].astype(float) * time_scale
    dt = (t - t.iloc[i0]).clip(lower=0)
    out["lat"] = out["lat"] + dt * meters_per_s * _DEG_PER_M_LAT
    return _mark(out, i0, "gps_ramp")


def inject_dropout(df: pd.DataFrame, cols: list[str], *, onset_frac: float = 0.5,
                   block_frac: float = 0.3, rng=None) -> pd.DataFrame:
    """Telemetri dropout: onset sonrasi rastgele bloklar halinde kolon grubu NaN olur.

    Gercek DoS'un merge-artifact'inden bagimsiz, kontrollu versiyonu (H3).
    """
    rng = rng or np.random.default_rng(0)
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    n = len(out) - i0
    n_drop = int(n * block_frac)
    if n_drop:
        start = i0 + int(rng.integers(0, max(n - n_drop, 1)))
        for c in cols:
            if c in out.columns:
                out.iloc[start : start + n_drop, out.columns.get_loc(c)] = np.nan
    return _mark(out, i0, "dropout")
