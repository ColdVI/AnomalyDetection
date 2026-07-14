"""adsb/models/isolation_forest_residual.py testleri."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adsb.models.isolation_forest_residual import (
    RESIDUAL_CHANNELS,
    fit_isolation_forest_residual,
    score_isolation_forest_residual,
)


def _natural_fit_frame(n=500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(0.0, 1.0, n) for c in RESIDUAL_CHANNELS})


def test_fit_rejects_synthetic_flag():
    df = _natural_fit_frame()
    with pytest.raises(ValueError):
        fit_isolation_forest_residual(df, contains_synthetic=True)


def test_fit_excludes_zero_mad_channel_without_floor():
    df = _natural_fit_frame()
    df["heading_residual"] = 0.0  # sabit -> MAD=0
    _, scaler, active_channels = fit_isolation_forest_residual(df)
    assert "heading_residual" not in active_channels
    assert "heading_residual" in scaler.excluded_channels_


def test_normal_rows_score_lower_than_extreme_outlier():
    df = _natural_fit_frame(n=1000)
    model, scaler, _ = fit_isolation_forest_residual(df)

    normal_probe = pd.DataFrame({c: [0.0] for c in RESIDUAL_CHANNELS})
    outlier_probe = pd.DataFrame({c: [50.0] for c in RESIDUAL_CHANNELS})  # tum kanallarda asiri

    normal_score = score_isolation_forest_residual(model, scaler, normal_probe).iloc[0]
    outlier_score = score_isolation_forest_residual(model, scaler, outlier_probe).iloc[0]
    assert outlier_score > normal_score


def test_nan_channel_row_scores_nan_not_dropped():
    df = _natural_fit_frame()
    model, scaler, _ = fit_isolation_forest_residual(df)

    probe = pd.DataFrame({c: [0.0, np.nan] for c in RESIDUAL_CHANNELS})
    scores = score_isolation_forest_residual(model, scaler, probe)
    assert len(scores) == 2
    assert np.isfinite(scores.iloc[0])
    assert np.isnan(scores.iloc[1])


def test_predict_binary_output_never_used():
    """Sozlesme: skorlama yalniz score_samples uzerinden, predict() kullanilmaz --
    bu, IsolationForest'in kendi contamination-tabanli ikili esigine bagimli olmadigimizi
    dogrudan test eder (skor fonksiyonu -1/1 degil surekli deger donduruyor mu)."""
    df = _natural_fit_frame()
    model, scaler, _ = fit_isolation_forest_residual(df)
    probe = pd.DataFrame({c: [0.0, 1.0, 2.0] for c in RESIDUAL_CHANNELS})
    scores = score_isolation_forest_residual(model, scaler, probe)
    assert scores.nunique() > 2  # ikili degil, surekli skor


def test_fit_is_deterministic_given_seed():
    df = _natural_fit_frame()
    model1, scaler1, _ = fit_isolation_forest_residual(df)
    model2, scaler2, _ = fit_isolation_forest_residual(df)
    probe = pd.DataFrame({c: [0.5] for c in RESIDUAL_CHANNELS})
    s1 = score_isolation_forest_residual(model1, scaler1, probe)
    s2 = score_isolation_forest_residual(model2, scaler2, probe)
    np.testing.assert_allclose(s1.to_numpy(), s2.to_numpy())
