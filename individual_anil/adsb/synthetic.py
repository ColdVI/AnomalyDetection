"""Faz 0.7 / ADSB-1: test-ONLY sentetik bozulma enjeksiyonu.

KRITIK KURAL (proje geneli, ML tarafindan devralinan disiplin): enjekte edilmis
ucuslar/pencereler ASLA egitime girmez, yalniz degerlendirme/test icin kullanilir.
Novelty-detection paradigmasi bozulmaz -- sadece degerlendirme yuzeyi genisler.

Cikti AYRI bir konuma yazilir (varsayilan `artifacts/adsb/synthetic/`); gercek
Silver/parse edilmis veriye hicbir zaman yazilmaz veya uzerine yazilmaz --
`save_synthetic_batch` bunun icin ayri bir dizin PARAMETRESI ister, cagiran
yanlislikla gercek veri yoluna yazamaz (fonksiyon path'in "synthetic" icerdigini
dogrular).

Her enjektor truth v2 kolonlarini ekler. ``label`` yalniz geriye-donuk okunabilir
bir injection-active isaretidir; degerlendirme etiketi degildir. Pencere truth'u
``observable_changed``/``evaluable_truth`` alanlarindan uretilir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adsb.truth import attach_event_truth_v2

_M_PER_DEG_LAT = 111_320.0


def _onset_index(df: pd.DataFrame, onset_frac: float) -> int:
    if len(df) == 0:
        raise ValueError("Sentetik event bos DataFrame'e uygulanamaz.")
    if not np.isfinite(onset_frac) or not 0.0 <= onset_frac < 1.0:
        raise ValueError("onset_frac 0 <= onset_frac < 1 araliginda olmali.")
    return int(len(df) * onset_frac)


def _mark(df: pd.DataFrame, active: np.ndarray, tag: str) -> pd.DataFrame:
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = None
    df.loc[active, "label"] = f"inj_{tag}"
    return df


def _active_from(df: pd.DataFrame, i0: int) -> np.ndarray:
    active = np.zeros(len(df), dtype=bool)
    active[i0:] = True
    return active


def inject_freeze(
    df: pd.DataFrame,
    col: str,
    *,
    onset_frac: float = 0.5,
    rng=None,
    event_type: str = "freeze",
) -> pd.DataFrame:
    """Sensor donmasi: onset sonrasi kolon son gecerli degerinde sabit kalir.

    Bildirilen kanal fiziksel gercekligiyle ayrisir -- orn. irtifa degismeye devam
    ederken bildirilen dikey hiz sabit kalir. Donmus deger sikca "normal araliktaki"
    bir sayidir (orn. 0), sirf-buyukluge-bakan bir dedektorun kacirabilecegi durum.
    """
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    out.iloc[i0:, out.columns.get_loc(col)] = out[col].iloc[max(i0 - 1, 0)]
    active = _active_from(out, i0)
    out = _mark(out, active, event_type)
    return attach_event_truth_v2(
        df,
        out,
        event_type=event_type,
        injection_active=active,
        observable_cols=[col],
    )


def inject_bias(df: pd.DataFrame, col: str, *, sigma_mult: float = 4.0,
                onset_frac: float = 0.5, rng=None,
                event_type: str = "bias") -> pd.DataFrame:
    """Ani bias: onset sonrasi kolona +k*sigma eklenir (sigma=0 ise 1.0 varsayilir)."""
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    sigma = float(out[col].std()) or 1.0
    out.iloc[i0:, out.columns.get_loc(col)] = out[col].iloc[i0:] + sigma_mult * sigma
    active = _active_from(out, i0)
    out = _mark(out, active, event_type)
    return attach_event_truth_v2(
        df,
        out,
        event_type=event_type,
        injection_active=active,
        observable_cols=[col],
    )


def inject_noise(df: pd.DataFrame, col: str, *, sigma_mult: float = 3.0,
                  onset_frac: float = 0.5, rng=None,
                  event_type: str = "noise") -> pd.DataFrame:
    """Gurultu artisi: onset sonrasi k*sigma olcekli Gauss gurultusu eklenir."""
    rng = rng or np.random.default_rng(0)
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    sigma = float(out[col].std()) or 1.0
    n = len(out) - i0
    out.iloc[i0:, out.columns.get_loc(col)] = out[col].iloc[i0:] + rng.normal(0, sigma_mult * sigma, n)
    active = _active_from(out, i0)
    out = _mark(out, active, event_type)
    return attach_event_truth_v2(
        df,
        out,
        event_type=event_type,
        injection_active=active,
        observable_cols=[col],
    )


def inject_dropout(df: pd.DataFrame, cols: list[str], *, onset_frac: float = 0.5,
                    block_frac: float = 0.3, rng=None,
                    event_type: str = "dropout") -> pd.DataFrame:
    """Telemetri dropout: onset sonrasi rastgele bir blok, verilen kolonlarda NaN olur."""
    if not np.isfinite(block_frac) or not 0.0 <= block_frac <= 1.0:
        raise ValueError("block_frac 0..1 araliginda olmali.")
    rng = rng or np.random.default_rng(0)
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    n = len(out) - i0
    n_drop = int(n * block_frac)
    active = np.zeros(len(out), dtype=bool)
    if n_drop:
        start = i0 + int(rng.integers(0, n - n_drop + 1))
        active[start:start + n_drop] = True
        for c in cols:
            if c in out.columns:
                out.iloc[start:start + n_drop, out.columns.get_loc(c)] = np.nan
    out = _mark(out, active, event_type)
    observable_cols = [c for c in cols if c in out.columns]
    if not observable_cols:
        raise KeyError(f"Dropout kolonlari DataFrame'de yok: {cols}")
    return attach_event_truth_v2(
        df,
        out,
        event_type=event_type,
        injection_active=active,
        observable_cols=observable_cols,
    )


def inject_position_ramp(
    df: pd.DataFrame, *, meters_per_s: float = 2.0, bearing_deg: float = 0.0,
    onset_frac: float = 0.5, time_col: str = "timestamp_utc",
    lat_col: str = "lat", lon_col: str = "lon", rng=None,
    event_type: str = "position_ramp",
) -> pd.DataFrame:
    """Yavas/stealthy konum kaymasi -- bildirilen hiz/track DEGISMEZ, speed_residual/
    heading_residual'in tam yakalamasi gereken durum (adsb'nin saniye-cinsinden
    `timestamp_utc`'una gore, keyfi kerterizde)."""
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    t = out[time_col].astype(float)
    dt = (t - t.iloc[i0]).clip(lower=0)
    ramp_m = dt * meters_per_s
    bearing_rad = np.radians(bearing_deg)
    lat0 = out[lat_col].iloc[max(i0 - 1, 0)]
    dlat_deg = (ramp_m * np.cos(bearing_rad)) / _M_PER_DEG_LAT
    dlon_deg = (ramp_m * np.sin(bearing_rad)) / (_M_PER_DEG_LAT * np.cos(np.radians(lat0)).clip(min=1e-6))
    out.iloc[i0:, out.columns.get_loc(lat_col)] = out[lat_col].iloc[i0:] + dlat_deg.iloc[i0:]
    out.iloc[i0:, out.columns.get_loc(lon_col)] = out[lon_col].iloc[i0:] + dlon_deg.iloc[i0:]
    active = _active_from(out, i0)
    out = _mark(out, active, event_type)
    return attach_event_truth_v2(
        df,
        out,
        event_type=event_type,
        injection_active=active,
        observable_cols=[lat_col, lon_col],
        time_col=time_col,
    )


# Fiziksel-anlam/gozlenebilirlik on-incelemesinden gecmis, adlandirilmis senaryolar
# (Faz 0.7). Codex'in arsivlenen denemesinin dersi: sentetik recall tek basina hicbir
# sey kanitlamaz -- bu senaryolarin her biri natural-data FA orani ile BIRLIKTE
# raporlanmali (bkz. ADSB1 plani).
PHYSICS_BREAK_RECIPES: dict[str, tuple] = {
    "vertical_rate_frozen": (
        inject_freeze,
        {"col": "vertical_rate_ms", "event_type": "vertical_rate_frozen"},
    ),
    "ground_speed_biased": (
        inject_bias,
        {"col": "ground_speed_ms", "event_type": "ground_speed_biased"},
    ),
    "track_frozen": (
        inject_freeze,
        {"col": "track_deg", "event_type": "track_frozen"},
    ),
    "position_ramp_stealthy": (
        inject_position_ramp,
        {"meters_per_s": 2.0, "event_type": "position_ramp_stealthy"},
    ),
    "altitude_dropout": (
        inject_dropout,
        {"cols": ["alt"], "event_type": "altitude_dropout"},
    ),
}


def save_synthetic_batch(df: pd.DataFrame, *, out_dir: str | Path, name: str) -> Path:
    """Sentetik bozulmus bir DataFrame'i AYRI bir konuma yazar -- gercek Silver
    veriyle karismasin diye `out_dir` yolunda "synthetic" gecmek ZORUNDA."""
    out_dir = Path(out_dir).resolve()
    if not any(part.lower() == "synthetic" for part in out_dir.parts):
        raise ValueError(
            f"out_dir ('{out_dir}') tam bir 'synthetic' path parcasi icermiyor -- gercek veriye "
            "yanlislikla yazmayi onlemek icin bu zorunlu."
        )
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("name tek bir guvenli dosya adi olmali; path parcasi iceremez.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f"{name}.parquet").resolve()
    if out_path.parent != out_dir:
        raise ValueError("Sentetik cikti out_dir disina tasamaz.")
    if out_path.exists():
        raise FileExistsError(f"Sentetik cikti zaten var; uzerine yazilmaz: {out_path}")
    df.to_parquet(out_path, index=False)
    return out_path
