"""Ortak zamansal feature yardimcilari (ML-0 fazi).

Butun fonksiyonlar TEK BIR ucusun (source_id) zaman-sirali DataFrame'i/Series'i
uzerinde calisir -- cagiran taraf groupby(source_id) yapar, ucuslar arasi bilgi
sizintisi burada tasarimla imkansizdir. Rolling pencereler yalnizca GECMISE bakar
(center=False): gercek zamanli kullanimda gelecekteki ornek elimizde olmaz.

Kaynak kararlar (docs/PIPELINE_PLAN + FableChat/LastChat tartismasi):
- Yaw/heading farklari acisal wrap ile alinir: ((fark + 180) % 360) - 180.
  Ornek: measured=179, commanded=-179 -> naive fark 358, gercek hata -2.
- CUSUM: kucuk ama ISRARLI sapmalari biriktiren degisim-noktasi istatistigi.
  Tek buyuk sicramayi Isolation Forest yakalar; yavas drift'i (stealthy GPS
  spoofing gibi) CUSUM yakalar.
- Donma (freeze) tespiti: ardisik degismeyen deger sayaci -- sensor donmasi ile
  gercek sabit ucusu ayirt etmek icin rolling varyansla birlikte kullanilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


def wrap_angle_deg(angle):
    """Herhangi bir aciyi/aci farkini [-180, 180) araligina indirger."""
    return ((np.asarray(angle, dtype=float) + 180.0) % 360.0) - 180.0


def angular_error_deg(measured, commanded):
    """Acisal takip hatasi (wrap-aware): measured - commanded, [-180, 180)."""
    return wrap_angle_deg(np.asarray(measured, dtype=float) - np.asarray(commanded, dtype=float))


def haversine_m(lat1, lon1, lat2, lon2):
    """Iki koordinat dizisi arasi buyuk-daire mesafesi (metre)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def rate_per_s(series: pd.Series, t_s: pd.Series, *, angular: bool = False) -> pd.Series:
    """Birinci fark / dt (birim/saniye). Ilk ornek NaN kalir.

    angular=True ise fark wrap edilir (yaw_rate icin 359->1 gecisi +2 olur, -358 degil).
    dt<=0 olan (duplicate timestamp) orneklerde NaN doner -- sonsuz rate uydurmayiz.
    """
    diff = series.astype(float).diff()
    if angular:
        diff = pd.Series(wrap_angle_deg(diff), index=series.index)
    dt = t_s.astype(float).diff()
    out = diff / dt
    return out.where(dt > 0)


def rolling_stats(series: pd.Series, window_rows: int, prefix: str,
                  stats: tuple[str, ...] = ("mean", "std", "max", "rms")) -> pd.DataFrame:
    """Gecmise-bakan rolling istatistikler (center=False, min_periods=1).

    window_rows satir cinsindendir: cagiran taraf ~ornekleme hizina gore
    saniyeyi satira cevirir (orn. 4 Hz veride 2 sn ~= 8 satir).
    """
    roll = series.astype(float).rolling(window=window_rows, min_periods=1)
    out = pd.DataFrame(index=series.index)
    if "mean" in stats:
        out[f"{prefix}_mean"] = roll.mean()
    if "std" in stats:
        out[f"{prefix}_std"] = roll.std()
    if "max" in stats:
        out[f"{prefix}_max"] = roll.max()
    if "min" in stats:
        out[f"{prefix}_min"] = roll.min()
    if "rms" in stats:
        out[f"{prefix}_rms"] = np.sqrt((series.astype(float) ** 2).rolling(window=window_rows, min_periods=1).mean())
    return out


def cusum(series: pd.Series, *, k: float | None = None) -> pd.DataFrame:
    """Iki yonlu CUSUM: residual'in isrararli pozitif/negatif kaymasini biriktirir.

    S+_t = max(0, S+_{t-1} + x_t - k),  S-_t = max(0, S-_{t-1} - x_t - k)

    k (allowance/slack) verilmezse ucus-ici medyan mutlak sapmadan turetilir
    (0.5 * MAD-tabanli sigma) -- kucuk gurultu birikmez, kalici sapma birikir.
    NaN'lar 0 katkiyla gecilir (birikimi sifirlamaz, sismirmez).
    """
    x = series.astype(float)
    if k is None:
        mad = (x - x.median()).abs().median()
        sigma = 1.4826 * mad if mad > 0 else (x.std() or 0.0)
        k = 0.5 * float(sigma if np.isfinite(sigma) else 0.0)
    vals = x.fillna(0.0).to_numpy()
    pos = np.zeros(len(vals))
    neg = np.zeros(len(vals))
    for i in range(1, len(vals)):
        pos[i] = max(0.0, pos[i - 1] + vals[i] - k)
        neg[i] = max(0.0, neg[i - 1] - vals[i] - k)
    return pd.DataFrame({"cusum_pos": pos, "cusum_neg": neg}, index=series.index)


def consecutive_unchanged(series: pd.Series) -> pd.Series:
    """Ardisik ayni-deger sayaci (donma/stale tespiti). Ilk ornek 0."""
    changed = series.ne(series.shift())
    groups = changed.cumsum()
    return series.groupby(groups).cumcount()


def ewma_deviation(series: pd.Series, *, halflife_rows: float = 10.0) -> pd.Series:
    """Deger ile gecmise-bakan EWMA'si arasindaki sapma (ani rejim degisimi sinyali)."""
    x = series.astype(float)
    return x - x.ewm(halflife=halflife_rows, min_periods=1).mean().shift(1)


def spectral_band_energy(series: pd.Series, window_rows: int, *, low_bin: int = 1) -> pd.Series:
    """Gecmise-bakan pencerede DC-disi FFT bant enerjisi (osilasyon imzasi).

    Aktuator arizalari cogu zaman zaman-duzleminde belirsiz, frekans-duzleminde
    bariz osilasyonlar uretir. Her ornek icin son window_rows ornegin
    |FFT|^2 toplamini (DC haric) dondurur. NaN'lar pencere ortalamasiyla doldurulur.
    """
    x = series.astype(float).to_numpy()
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        start = max(0, i - window_rows + 1)
        win = x[start : i + 1]
        if len(win) < 4:
            continue
        win = np.where(np.isfinite(win), win, np.nanmean(win) if np.isfinite(np.nanmean(win)) else 0.0)
        spec = np.abs(np.fft.rfft(win - win.mean())) ** 2
        out[i] = float(spec[low_bin:].sum())
    return pd.Series(out, index=series.index)


def add_relative_time(df: pd.DataFrame, time_col: str, *, scale: float = 1.0) -> pd.Series:
    """Ucus-ici goreli zaman (saniye): mutlak timestamp feature olarak SIZDIRILMAZ."""
    t = df[time_col].astype(float)
    return (t - t.iloc[0]) * scale
