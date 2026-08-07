"""ADSB-0 fiziksel-tutarlilik residual testleri -- sentetik, elle hesaplanmis beklenen degerlerle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.adsb.injection import inject_bias, inject_freeze
from src.adsb.physics_features import (
    EARTH_RADIUS_M,
    compute_physics_residuals,
    heading_residual,
    speed_residual,
    turn_bank_residual,
    vertical_rate_residual,
)

# haversine (pure kuzey hareket icin) distance = R * dlat_rad TAM esittir -- bu yuzden
# testin kendi kurdugu ucus da AYNI R'yi kullanmali, aksi halde "tutarli" ucus bile
# residual != 0 verir (WGS84 111320 m/deg gibi FARKLI bir sabit kullanmak residual'a
# sahte bir sistematik hata sokar).
_M_PER_DEG_LAT = EARTH_RADIUS_M * np.pi / 180.0


def _straight_flight(n: int = 5, dt: float = 10.0, speed_mps: float = 100.0,
                      climb_mps: float = 5.0) -> pd.DataFrame:
    """Kuzeye giden, sabit hiz/dikey-hiz ile FIZIKSEL OLARAK TUTARLI sentetik ucus."""
    t = np.arange(n) * dt
    dlat_per_step = (speed_mps * dt) / _M_PER_DEG_LAT
    lat = 40.0 + np.arange(n) * dlat_per_step
    lon = np.full(n, 29.0)
    alt = 1000.0 + np.arange(n) * climb_mps * dt
    return pd.DataFrame({
        "flight_id": "F1",
        "timestamp_utc": t,
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "ground_speed_ms": speed_mps,
        "track_deg": 0.0,
        "vertical_rate_ms": climb_mps,
        "label": None,
    })


def test_vertical_rate_residual_zero_for_consistent_climb():
    df = _straight_flight()
    res = vertical_rate_residual(df)
    assert np.isnan(res.iloc[0])
    assert res.iloc[1:].abs().max() < 1e-9


def test_vertical_rate_residual_nonzero_after_freeze_injection():
    # gercek tirmanma hizi onset'te 5 m/s'den 12 m/s'e degisiyor (dogru bildirilseydi
    # vertical_rate_ms de degisirdi); inject_freeze bildirilen kanali eski degerde (5)
    # kilitliyor -- fiziksel-tutarlilik residual'i bu kopmayi yakalamali.
    n, dt = 10, 10.0
    i0 = 5
    alt = np.concatenate([
        1000.0 + np.arange(i0) * 5.0 * dt,
        1000.0 + (i0 - 1) * 5.0 * dt + np.arange(1, n - i0 + 1) * 12.0 * dt,
    ])
    vrate = np.concatenate([np.full(i0, 5.0), np.full(n - i0, 12.0)])
    df = _straight_flight(n=n, dt=dt)
    df["alt"] = alt
    df["vertical_rate_ms"] = vrate  # dogru bildirim (enjeksiyon oncesi tutarli)

    bozuk = inject_freeze(df, "vertical_rate_ms", onset_frac=0.5)
    res = vertical_rate_residual(bozuk)

    assert res.iloc[1:i0].abs().max() < 1e-9  # onset oncesi hala tutarli
    assert res.iloc[i0:].abs().min() > 5.0  # onset sonrasi: bildirilen donmus, gercek degismis


def test_vertical_rate_residual_nonzero_when_reported_value_is_wrong():
    df = _straight_flight(n=10)
    df["vertical_rate_ms"] = 0.0  # yanlis bildirim: hep 0 diyor ama irtifa artiyor
    res = vertical_rate_residual(df)
    assert res.iloc[1:].abs().min() > 4.0  # gercek climb 5 m/s, bildirilen 0 -> residual ~ -5


def test_speed_residual_zero_for_consistent_straight_flight():
    df = _straight_flight()
    res = speed_residual(df)
    assert np.isnan(res.iloc[0])
    assert res.iloc[1:].abs().max() < 1e-6


def test_speed_residual_nonzero_after_bias_injection():
    df = _straight_flight(n=10)
    bozuk = inject_bias(df, "ground_speed_ms", sigma_mult=4.0, onset_frac=0.5)
    res = speed_residual(bozuk)
    i0 = 5
    assert res.iloc[1:i0].abs().max() < 1e-6
    assert res.iloc[i0:].abs().min() > 0.5  # bias sonrasi belirgin sapma


def test_heading_residual_zero_for_northbound_flight():
    df = _straight_flight()
    res = heading_residual(df)
    assert np.isnan(res.iloc[0])
    assert res.iloc[1:].abs().max() < 0.1


def test_turn_bank_residual_zero_for_coordinated_turn():
    n, dt, speed, roll = 5, 10.0, 100.0, 10.0
    g = 9.80665
    rate_deg_s = np.degrees(g * np.tan(np.radians(roll)) / speed)
    track = np.arange(n) * rate_deg_s * dt
    df = pd.DataFrame({
        "flight_id": "F1",
        "timestamp_utc": np.arange(n) * dt,
        "track_deg": track,
        "roll_deg": roll,
        "ground_speed_ms": speed,
    })
    res = turn_bank_residual(df)
    assert np.isnan(res.iloc[0])
    assert res.iloc[1:].abs().max() < 1e-6


def test_compute_physics_residuals_missing_roll_gives_nan_column():
    df = _straight_flight()
    out = compute_physics_residuals(df)
    assert "turn_bank_residual" in out.columns
    assert out["turn_bank_residual"].isna().all()


def test_compute_physics_residuals_does_not_leak_across_flight_boundary():
    a = _straight_flight(n=3)
    b = _straight_flight(n=3)
    b["flight_id"] = "F2"
    b["timestamp_utc"] = b["timestamp_utc"] + 10_000  # zamanda cok uzak, ama flight_id zaten ayri
    df = pd.concat([a, b], ignore_index=True)
    out = compute_physics_residuals(df)
    # ikinci ucusun ilk satiri, birinci ucusun son satirina gore diff ALMAMALI (NaN olmali)
    first_row_of_b = out[out["flight_id"] == "F2"].iloc[0]
    assert np.isnan(first_row_of_b["vertical_rate_residual"])
    assert np.isnan(first_row_of_b["speed_residual"])
    assert np.isnan(first_row_of_b["heading_residual"])
