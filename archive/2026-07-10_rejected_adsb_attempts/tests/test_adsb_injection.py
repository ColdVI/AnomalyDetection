"""ADSB-1 enjeksiyon testleri: src/ml/injection.py yeniden-kullaniminin adsb kolonlarinda
calistigini, ve yeni inject_position_ramp'in dogru calistigini dogrular."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.adsb.injection import PHYSICS_BREAK_RECIPES, inject_freeze, inject_position_ramp

_M_PER_DEG_LAT = 111_320.0


def _flight(n: int = 10) -> pd.DataFrame:
    t = np.arange(n) * 10.0
    return pd.DataFrame({
        "timestamp_utc": t,
        "lat": 40.0 + np.arange(n) * 0.001,
        "lon": 29.0 + np.arange(n) * 0.001,
        "alt": 1000.0 + np.arange(n) * 10.0,
        "ground_speed_ms": 100.0,
        "track_deg": 45.0,
        "vertical_rate_ms": 1.0,
        "label": None,
    })


def test_generic_injectors_reused_directly_on_adsb_columns():
    df = _flight()
    bozuk = inject_freeze(df, "ground_speed_ms", onset_frac=0.5)
    assert (bozuk["label"].iloc[5:] == "inj_freeze").all()
    assert (bozuk["label"].iloc[:5].isna()).all()
    assert (bozuk["ground_speed_ms"].iloc[5:] == bozuk["ground_speed_ms"].iloc[4]).all()


def test_inject_position_ramp_moves_north_after_onset_only():
    df = _flight(n=10)
    bozuk = inject_position_ramp(df, meters_per_s=10.0, bearing_deg=0.0, onset_frac=0.5)

    # onset oncesi degismemis
    pd.testing.assert_series_equal(bozuk["lat"].iloc[:5], df["lat"].iloc[:5])
    pd.testing.assert_series_equal(bozuk["lon"].iloc[:5], df["lon"].iloc[:5])

    # onset sonrasi: kuzeye (bearing=0) kayma, lon degismemeli
    pd.testing.assert_series_equal(bozuk["lon"].iloc[5:], df["lon"].iloc[5:])
    drift_deg = bozuk["lat"].iloc[9] - df["lat"].iloc[9]
    # inject_position_ramp, dt'yi t.iloc[i0]'a (onset satirinin kendisi) gore hesaplar
    expected_drift_m = 10.0 * (df["timestamp_utc"].iloc[9] - df["timestamp_utc"].iloc[5])
    expected_drift_deg = expected_drift_m / _M_PER_DEG_LAT
    assert abs(drift_deg - expected_drift_deg) < 1e-6
    assert (bozuk["label"].iloc[5:] == "inj_position_ramp").all()


def test_inject_position_ramp_east_bearing_moves_lon_not_lat():
    df = _flight(n=10)
    bozuk = inject_position_ramp(df, meters_per_s=10.0, bearing_deg=90.0, onset_frac=0.5)
    pd.testing.assert_series_equal(
        bozuk["lat"].iloc[5:].round(9), df["lat"].iloc[5:].round(9)
    )
    # onset satirinin kendisinde (dt=0) henuz kayma yok; sonraki satirlarda kesin var
    assert (bozuk["lon"].iloc[5:] >= df["lon"].iloc[5:]).all()
    assert (bozuk["lon"].iloc[6:] > df["lon"].iloc[6:]).all()


def test_speed_and_track_unchanged_by_position_ramp():
    """position_ramp'in butun amaci bu: konum kayar ama bildirilen hiz/track SABIT kalir."""
    df = _flight(n=10)
    bozuk = inject_position_ramp(df, meters_per_s=10.0, onset_frac=0.5)
    pd.testing.assert_series_equal(bozuk["ground_speed_ms"], df["ground_speed_ms"])
    pd.testing.assert_series_equal(bozuk["track_deg"], df["track_deg"])


def test_physics_break_recipes_are_all_callable():
    df = _flight(n=10)
    for name, (fn, kwargs) in PHYSICS_BREAK_RECIPES.items():
        bozuk = fn(df, onset_frac=0.5, **kwargs)
        assert len(bozuk) == len(df), name
        assert bozuk["label"].iloc[5:].notna().all(), name
