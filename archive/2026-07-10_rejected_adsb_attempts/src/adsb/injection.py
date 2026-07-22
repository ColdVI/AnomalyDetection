"""ADSB-1: sentetik enjeksiyon -- fiziksel-tutarlilik detektorunun dogrulama araci.

Gercek adsb.lol trafiginde neredeyse hic gercek etiketli anomali yok (ticari
havacilik, hic drone/UAV kategorisi gorulmedi -- format referansi §5.1). Bu yuzden
ADSB-1'in ana degerlendirmesi SENTETIK: temiz bir gercek ucusa bilinen bir bozulma
enjekte edilir, detektorun (a) bozulani yakalayip (b) orijinali yakalamamasi
beklenir -- zemin gercek-ama-gurultulu etiket degil, bilinen sentetik gercek
oldugu icin tamamen kapatilabilir bir test (bkz. docs/ADSB1_PHYSICS_DETECTOR_PLAN.md).

`src/ml/injection.py`deki freeze/bias/noise/dropout kolon-adi-agnostik ve ADS-B
Silver kolonlarinda (`alt`, `ground_speed_ms`, `track_deg`, `vertical_rate_ms`, ...)
degisiklik gerekmeden calisir -- burada oldugu gibi yeniden disa aktarilir. Onemli
nokta: bunlarin cogu, "bildirilen kanali fiziksel karsiliginda KOPARAN" senaryolari
zaten kapsiyor -- orn. `inject_freeze(df, "vertical_rate_ms", ...)` = "irtifa
gercekte degisirken bildirilen dikey hiz sabit kaliyor" (frozen deger sikca
sirf-buyukluge-bakan bir dedektorun normal sayacagi bir deger, orn. 0 m/s;
fiziksel-tutarlilik residual'i ise bunu yakalar). `inject_position_ramp`, adsb'nin
saniye-cinsinden `timestamp_utc`'una gore genellenmis (kuzey-sabit degil, herhangi
bir kerterizde) `src.ml.injection.inject_gps_ramp`'in yerini tutar -- konum yavasca
kayarken bildirilen hiz DEGISMEZ, speed_residual'in tam yakalamasi gereken durum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.injection import (  # noqa: F401 (yeniden disa aktarim -- kolon-adi-agnostik, adsb kolonlarinda dogrudan calisir)
    _mark,
    _onset_index,
    inject_bias,
    inject_dropout,
    inject_freeze,
    inject_noise,
)

_M_PER_DEG_LAT = 111_320.0


def inject_position_ramp(
    df: pd.DataFrame,
    *,
    meters_per_s: float = 2.0,
    bearing_deg: float = 0.0,
    onset_frac: float = 0.5,
    time_col: str = "timestamp_utc",
    lat_col: str = "lat",
    lon_col: str = "lon",
    rng=None,
) -> pd.DataFrame:
    """Yavas/stealthy konum kaymasi: onset'ten itibaren `bearing_deg` yonunde,
    saniyede `meters_per_s` buyuyen kademeli ramp. Bildirilen hiz/track DEGISMEZ --
    `speed_residual`/`heading_residual`'in yakalamasi gereken tam durum bu.

    `src.ml.injection.inject_gps_ramp`'in genellemesidir: o fonksiyon yalniz kuzeye
    (lat) kayar ve PX4 mikrosaniye zaman damgasi varsayar (`time_scale=1e-6`); bu
    surum lat+lon ikisini de kaydirir ve adsb'nin saniye-cinsinden `timestamp_utc`'una
    gore olceklenir (time_scale=1.0 -- parametre olarak acik birak, farkli kaynaklar
    icin degistirilebilir).
    """
    i0 = _onset_index(df, onset_frac)
    out = df.copy()
    t = out[time_col].astype(float)
    dt = (t - t.iloc[i0]).clip(lower=0)
    ramp_m = dt * meters_per_s
    bearing_rad = np.radians(bearing_deg)
    lat0 = out[lat_col].iloc[max(i0 - 1, 0)]
    dlat_deg = (ramp_m * np.cos(bearing_rad)) / _M_PER_DEG_LAT
    dlon_deg = (ramp_m * np.sin(bearing_rad)) / (
        _M_PER_DEG_LAT * np.cos(np.radians(lat0)).clip(min=1e-6)
    )
    out.iloc[i0:, out.columns.get_loc(lat_col)] = out[lat_col].iloc[i0:] + dlat_deg.iloc[i0:]
    out.iloc[i0:, out.columns.get_loc(lon_col)] = out[lon_col].iloc[i0:] + dlon_deg.iloc[i0:]
    return _mark(out, i0, "position_ramp")


# Adlandirilmis fizik-koparma senaryolari -- ADSB-1 dogrulama scripti bu isimleri
# dogrudan kullanir (script: scripts/run_adsb1_synthetic_validation.py, henuz yazilmadi).
PHYSICS_BREAK_RECIPES: dict[str, tuple] = {
    "vertical_rate_frozen": (inject_freeze, {"col": "vertical_rate_ms"}),
    "ground_speed_biased": (inject_bias, {"col": "ground_speed_ms"}),
    "track_frozen": (inject_freeze, {"col": "track_deg"}),
    "position_ramp_stealthy": (inject_position_ramp, {"meters_per_s": 2.0}),
    "altitude_dropout": (inject_dropout, {"cols": ["alt"]}),
}
