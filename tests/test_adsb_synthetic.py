"""adsb/synthetic.py testleri."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adsb.synthetic import (
    PHYSICS_BREAK_RECIPES,
    inject_bias,
    inject_dropout,
    inject_freeze,
    inject_noise,
    inject_position_ramp,
    save_synthetic_batch,
)

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


def test_inject_freeze_marks_label_and_freezes_value():
    df = _flight()
    bozuk = inject_freeze(df, "ground_speed_ms", onset_frac=0.5)
    assert (bozuk["label"].iloc[5:] == "inj_freeze").all()
    assert bozuk["label"].iloc[:5].isna().all()
    assert (bozuk["ground_speed_ms"].iloc[5:] == bozuk["ground_speed_ms"].iloc[4]).all()


def test_inject_bias_shifts_after_onset_only():
    df = _flight()
    bozuk = inject_bias(df, "ground_speed_ms", sigma_mult=4.0, onset_frac=0.5)
    pd.testing.assert_series_equal(bozuk["ground_speed_ms"].iloc[:5], df["ground_speed_ms"].iloc[:5])
    assert (bozuk["ground_speed_ms"].iloc[5:] > df["ground_speed_ms"].iloc[5:]).all()


def test_inject_noise_changes_values_after_onset():
    df = _flight()
    bozuk = inject_noise(df, "ground_speed_ms", sigma_mult=3.0, onset_frac=0.5, rng=np.random.default_rng(1))
    assert not (bozuk["ground_speed_ms"].iloc[5:] == df["ground_speed_ms"].iloc[5:]).all()


def test_inject_dropout_introduces_nans_in_target_cols_only():
    df = _flight()
    bozuk = inject_dropout(df, ["alt"], onset_frac=0.5, block_frac=1.0, rng=np.random.default_rng(0))
    assert bozuk["alt"].iloc[5:].isna().any()
    assert bozuk["ground_speed_ms"].notna().all()


def test_inject_position_ramp_moves_north_after_onset_only():
    df = _flight(n=10)
    bozuk = inject_position_ramp(df, meters_per_s=10.0, bearing_deg=0.0, onset_frac=0.5)
    pd.testing.assert_series_equal(bozuk["lat"].iloc[:5], df["lat"].iloc[:5])
    pd.testing.assert_series_equal(bozuk["lon"].iloc[5:], df["lon"].iloc[5:])
    drift_deg = bozuk["lat"].iloc[9] - df["lat"].iloc[9]
    expected_drift_m = 10.0 * (df["timestamp_utc"].iloc[9] - df["timestamp_utc"].iloc[5])
    assert abs(drift_deg - expected_drift_m / _M_PER_DEG_LAT) < 1e-6


def test_inject_position_ramp_leaves_speed_and_track_untouched():
    df = _flight(n=10)
    bozuk = inject_position_ramp(df, meters_per_s=10.0, onset_frac=0.5)
    pd.testing.assert_series_equal(bozuk["ground_speed_ms"], df["ground_speed_ms"])
    pd.testing.assert_series_equal(bozuk["track_deg"], df["track_deg"])


def test_physics_break_recipes_all_callable():
    df = _flight(n=10)
    for name, (fn, kwargs) in PHYSICS_BREAK_RECIPES.items():
        bozuk = fn(df, onset_frac=0.5, **kwargs)
        assert len(bozuk) == len(df), name
        assert bozuk["label"].iloc[5:].notna().all(), name


def test_save_synthetic_batch_rejects_non_synthetic_path():
    # tmp_path fixture kullanilmiyor -- pytest'in test-adi-tabanli tmp dizini
    # "synthetic" kelimesini kazara icerebiliyor (bu test fonksiyonunun adi gibi),
    # bu yuzden guard'i test etmek icin tamamen ilgisiz, sabit bir yol kullanilir.
    import tempfile
    real_data_path = Path(tempfile.gettempdir()) / "adsb_guard_check" / "real_data"
    df = _flight()
    with pytest.raises(ValueError):
        save_synthetic_batch(df, out_dir=real_data_path, name="x")


def test_save_synthetic_batch_writes_and_roundtrips(tmp_path):
    df = inject_freeze(_flight(), "ground_speed_ms", onset_frac=0.5)
    out_dir = tmp_path / "synthetic" / "adsb"
    path = save_synthetic_batch(df, out_dir=out_dir, name="a12345_000__vertical_rate_frozen__of0.5")
    assert path.exists()

    reloaded = pd.read_parquet(path)
    numeric_cols = [c for c in df.columns if c != "label"]
    pd.testing.assert_frame_equal(
        reloaded[numeric_cols], df[numeric_cols].reset_index(drop=True), check_dtype=False
    )
    # label: null-konumlari ve dolu degerler eslesmeli (None vs NaN gosterimi
    # parquet round-trip'inde farkli olabilir, bu onemli degil)
    assert (reloaded["label"].isna() == df["label"].isna().reset_index(drop=True)).all()
    assert (
        reloaded["label"].dropna().reset_index(drop=True).astype(str)
        == df["label"].dropna().reset_index(drop=True).astype(str)
    ).all()
