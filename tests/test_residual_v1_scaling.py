import numpy as np
import pandas as pd
import pytest

from residual_v1.decision.scaling import ZeroMADChannel, fit_robust_scaler, robust_z


def test_scaler_fits_only_train_normal_and_clips_at_eight():
    residual = pd.Series([-2.0, 0.0, 2.0, 1000.0])
    train = pd.Series([True, True, True, False])
    params = fit_robust_scaler(residual, train, channel="c")
    assert params.median == 0.0
    assert params.mad == 2.0
    assert params.fit_rows == 3
    assert np.allclose(robust_z(residual, params), [-1.0, 0.0, 1.0, 8.0])


def test_zero_mad_is_an_explicit_exclusion():
    with pytest.raises(ZeroMADChannel, match="MAD is zero"):
        fit_robust_scaler(pd.Series([1.0, 1.0, 9.0]), pd.Series([True, True, False]), channel="c")


def test_non_train_outlier_does_not_change_parameters():
    base = fit_robust_scaler(pd.Series([0.0, 1.0, 2.0]), pd.Series([True, True, True]), channel="c")
    extended = fit_robust_scaler(
        pd.Series([0.0, 1.0, 2.0, 1e9]),
        pd.Series([True, True, True, False]),
        channel="c",
    )
    assert base == extended
