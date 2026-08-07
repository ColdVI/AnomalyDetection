"""adsb/models/lstm_forecaster.py testleri."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from adsb.models.lstm_forecaster import (
    LSTMForecaster,
    forecast_residual_scores,
    train_lstm_forecaster,
)


def _toy_data(n=40, window=8, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, window, n_features)).astype(np.float32)
    M = np.ones_like(X, dtype=np.float32)
    return X, M


def test_forward_shape():
    model = LSTMForecaster(n_features=3, horizon=2, hidden_size=8)
    x = torch.randn(5, 6, 3)  # history_len=6
    out = model(x)
    assert out.shape == (5, 2, 3)


def test_history_len_must_be_smaller_than_window():
    X, M = _toy_data(window=8)
    with pytest.raises(ValueError):
        train_lstm_forecaster(X, M, history_len=8, n_features=3, epochs=1, seed=0)


def test_training_reduces_loss():
    X, M = _toy_data()
    _, history = train_lstm_forecaster(X, M, history_len=6, n_features=3, hidden_size=8, epochs=20, seed=0)
    assert history[-1] < history[0]


def test_training_is_deterministic_given_seed():
    X, M = _toy_data()
    model1, h1 = train_lstm_forecaster(X, M, history_len=6, n_features=3, hidden_size=8, epochs=5, seed=42)
    model2, h2 = train_lstm_forecaster(X, M, history_len=6, n_features=3, hidden_size=8, epochs=5, seed=42)
    assert h1 == h2
    s1 = forecast_residual_scores(model1, X, M, history_len=6)
    s2 = forecast_residual_scores(model2, X, M, history_len=6)
    np.testing.assert_array_equal(s1, s2)


def test_masking_matches_manual_masked_computation():
    X, M = _toy_data(n=5)
    model, _ = train_lstm_forecaster(X, M, history_len=6, n_features=3, hidden_size=8, epochs=3, seed=0)

    M_masked = M.copy()
    M_masked[:, 6, 0] = 0.0  # hedef bolgesindeki bir pozisyonu maskele

    scores = forecast_residual_scores(model, X, M_masked, history_len=6)

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[:, :6], dtype=torch.float32)).numpy()
    target = X[:, 6:]
    target_mask = M_masked[:, 6:]
    sq_err = (target - pred) ** 2 * target_mask
    denom = target_mask.sum(axis=(1, 2))
    expected = sq_err.sum(axis=(1, 2)) / denom
    np.testing.assert_allclose(scores, expected, rtol=1e-5)
