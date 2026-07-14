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
    assert bozuk["injection_active"].tolist() == [False] * 5 + [True] * 5
    # Bu fixture'da hiz zaten sabit: komut aktif, fakat paired gozlem degismiyor.
    assert not bozuk["observable_changed"].any()
    assert bozuk["evaluable_truth"].all()


def test_inject_bias_shifts_after_onset_only():
    df = _flight()
    bozuk = inject_bias(df, "ground_speed_ms", sigma_mult=4.0, onset_frac=0.5)
    pd.testing.assert_series_equal(bozuk["ground_speed_ms"].iloc[:5], df["ground_speed_ms"].iloc[:5])
    assert (bozuk["ground_speed_ms"].iloc[5:] > df["ground_speed_ms"].iloc[5:]).all()


def test_inject_noise_changes_values_after_onset():
    df = _flight()
    bozuk = inject_noise(df, "ground_speed_ms", sigma_mult=3.0, onset_frac=0.5, rng=np.random.default_rng(1))
    assert not (bozuk["ground_speed_ms"].iloc[5:] == df["ground_speed_ms"].iloc[5:]).all()


def test_inject_dropout_marks_only_exact_random_block():
    df = _flight()
    bozuk = inject_dropout(df, ["alt"], onset_frac=0.5, block_frac=0.4, rng=np.random.default_rng(0))
    active_positions = np.flatnonzero(bozuk["injection_active"].to_numpy())

    assert len(active_positions) == 2
    assert active_positions[0] >= 5
    assert np.diff(active_positions).tolist() == [1]
    assert bozuk.loc[active_positions, "alt"].isna().all()
    assert bozuk["alt"].isna().sum() == 2
    assert bozuk["label"].notna().to_numpy().tolist() == bozuk["injection_active"].tolist()
    assert bozuk["ground_speed_ms"].notna().all()


def test_inject_dropout_existing_nan_is_active_but_not_observable():
    df = _flight()
    first = inject_dropout(
        df, ["alt"], onset_frac=0.5, block_frac=0.4, rng=np.random.default_rng(0)
    )
    active_positions = np.flatnonzero(first["injection_active"].to_numpy())
    df.loc[active_positions[0], "alt"] = np.nan

    bozuk = inject_dropout(
        df, ["alt"], onset_frac=0.5, block_frac=0.4, rng=np.random.default_rng(0)
    )

    assert bozuk["injection_active"].iloc[active_positions[0]]
    assert not bozuk["observable_changed"].iloc[active_positions[0]]
    assert bozuk["observable_changed"].iloc[active_positions[1]]


def test_inject_position_ramp_moves_north_after_onset_only():
    df = _flight(n=10)
    bozuk = inject_position_ramp(df, meters_per_s=10.0, bearing_deg=0.0, onset_frac=0.5)
    pd.testing.assert_series_equal(bozuk["lat"].iloc[:5], df["lat"].iloc[:5])
    pd.testing.assert_series_equal(bozuk["lon"].iloc[5:], df["lon"].iloc[5:])
    drift_deg = bozuk["lat"].iloc[9] - df["lat"].iloc[9]
    expected_drift_m = 10.0 * (df["timestamp_utc"].iloc[9] - df["timestamp_utc"].iloc[5])
    assert abs(drift_deg - expected_drift_m / _M_PER_DEG_LAT) < 1e-6
    assert bozuk["injection_active"].iloc[5]
    assert not bozuk["observable_changed"].iloc[5]  # dt=0: komut var, gozlenir fark yok
    assert bozuk["observable_changed"].iloc[6:].all()
    assert bozuk["attack_onset"].iloc[0] == df["timestamp_utc"].iloc[5]
    assert bozuk["observable_onset"].iloc[0] == df["timestamp_utc"].iloc[6]


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
        assert bozuk["injection_active"].any(), name
        assert (
            bozuk["label"].notna().to_numpy() == bozuk["injection_active"].to_numpy()
        ).all(), name
        assert bozuk["event_type"].eq(name).all(), name


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
    numeric_cols = [
        c for c in df.columns
        if c != "label" and pd.api.types.is_numeric_dtype(df[c].dtype)
    ]
    pd.testing.assert_frame_equal(
        reloaded[numeric_cols], df[numeric_cols].reset_index(drop=True), check_dtype=False
    )
    # Nullable object kolonlarda None/NaN/pd.NA gosterimi Parquet round-trip'inde
    # degisebilir; null konumu ile dolu deger ayni kalmalidir.
    for col in set(df.columns) - set(numeric_cols):
        assert (reloaded[col].isna() == df[col].isna().reset_index(drop=True)).all()
        assert (
            reloaded[col].dropna().reset_index(drop=True).astype(str)
            == df[col].dropna().reset_index(drop=True).astype(str)
        ).all()


def test_save_synthetic_batch_is_fail_if_exists(tmp_path):
    out_dir = tmp_path / "synthetic" / "adsb"
    df = _flight()
    save_synthetic_batch(df, out_dir=out_dir, name="same")

    with pytest.raises(FileExistsError):
        save_synthetic_batch(df, out_dir=out_dir, name="same")


def test_save_synthetic_batch_requires_exact_path_component_and_safe_name(tmp_path):
    df = _flight()
    with pytest.raises(ValueError):
        save_synthetic_batch(df, out_dir=tmp_path / "synthetic_like", name="x")
    with pytest.raises(ValueError):
        save_synthetic_batch(df, out_dir=tmp_path / "synthetic", name="../escape")
