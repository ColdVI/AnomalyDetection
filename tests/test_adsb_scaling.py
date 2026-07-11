"""adsb/scaling.py testleri."""

from __future__ import annotations

import numpy as np
import pytest

from adsb.scaling import ClippedRobustScaler


def test_fit_transform_centers_and_scales():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=1000.0, scale=50.0, size=(200, 5, 1))  # buyuk-genlikli tek kanal
    M = np.ones_like(X)

    scaler = ClippedRobustScaler(clip=5.0)
    scaled = scaler.fit_transform(X, M)

    assert abs(np.median(scaled)) < 0.5  # medyan sifira yakin
    assert scaled.min() >= -5.0 and scaled.max() <= 5.0  # kirpma calisti


def test_extreme_outlier_is_clipped_not_dominant():
    rng = np.random.default_rng(1)
    X = rng.normal(loc=0.0, scale=1.0, size=(100, 5, 1))
    M = np.ones_like(X)
    X[0, 0, 0] = 1_000_000.0  # asiri deger (sensor hatasi benzeri)

    scaler = ClippedRobustScaler(clip=5.0)
    scaled = scaler.fit_transform(X, M)

    assert scaled.max() <= 5.0  # kirpilmamis olsaydi milyonlarca olurdu


def test_masked_values_excluded_from_fit_statistics():
    X = np.array([[[1.0], [2.0], [3.0], [1_000_000.0]]])  # (1, 4, 1)
    M = np.array([[[1.0], [1.0], [1.0], [0.0]]])  # son deger maskeli (gecersiz)

    scaler = ClippedRobustScaler(clip=5.0)
    scaler.fit(X, M)

    # medyan 1000000'dan etkilenmemis olmali (yalniz 1,2,3 kullanildi)
    assert scaler.median_[0] == pytest.approx(2.0)


def test_masked_positions_remain_zero_after_transform():
    X = np.array([[[1.0, 5.0], [2.0, 999.0]]])
    M = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    scaler = ClippedRobustScaler(clip=5.0).fit(X, M)
    scaled = scaler.transform(X, M)
    assert scaled[0, 0, 1] == 0.0
    assert scaled[0, 1, 1] == 0.0


def test_constant_channel_does_not_divide_by_zero():
    X = np.full((10, 3, 1), 42.0)
    M = np.ones_like(X)
    scaler = ClippedRobustScaler(clip=5.0)
    scaled = scaler.fit_transform(X, M)  # IQR=0 -> fallback 1.0, patlamamali
    assert np.isfinite(scaled).all()


def test_transform_before_fit_raises():
    scaler = ClippedRobustScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.zeros((1, 2, 1)), np.ones((1, 2, 1)))


def test_fit_on_train_applies_consistently_to_val():
    rng = np.random.default_rng(2)
    X_train = rng.normal(loc=100.0, scale=10.0, size=(50, 4, 2))
    M_train = np.ones_like(X_train)
    X_val = rng.normal(loc=100.0, scale=10.0, size=(10, 4, 2))
    M_val = np.ones_like(X_val)

    scaler = ClippedRobustScaler(clip=5.0).fit(X_train, M_train)
    scaled_val = scaler.transform(X_val, M_val)
    # ayni median/iqr val'e de uygulanmis (val'in kendi istatistigi kullanilmamis)
    manual = np.clip((X_val - scaler.median_) / (scaler.iqr_ + 1e-6), -5.0, 5.0)
    np.testing.assert_allclose(scaled_val, manual)
