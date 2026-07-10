"""adsb/models/dense_autoencoder.py testleri."""

from __future__ import annotations

import numpy as np
import torch

from adsb.models.dense_autoencoder import (
    DenseAutoencoder,
    reconstruction_scores,
    train_dense_autoencoder,
)


def _toy_data(n=40, window=6, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, window, n_features)).astype(np.float32)
    M = np.ones_like(X, dtype=np.float32)
    return X, M


def test_forward_shape():
    model = DenseAutoencoder(window=6, n_features=3, hidden_dims=(8, 4))
    x = torch.randn(5, 6, 3)
    out = model(x)
    assert out.shape == (5, 6, 3)


def test_training_reduces_loss():
    X, M = _toy_data()
    _, history = train_dense_autoencoder(X, M, window=6, n_features=3, hidden_dims=(8, 4), epochs=20, seed=0)
    assert history[-1] < history[0]


def test_training_is_deterministic_given_seed():
    X, M = _toy_data()
    model1, h1 = train_dense_autoencoder(X, M, window=6, n_features=3, hidden_dims=(8, 4), epochs=5, seed=42)
    model2, h2 = train_dense_autoencoder(X, M, window=6, n_features=3, hidden_dims=(8, 4), epochs=5, seed=42)
    assert h1 == h2
    s1 = reconstruction_scores(model1, X, M)
    s2 = reconstruction_scores(model2, X, M)
    np.testing.assert_array_equal(s1, s2)


def test_masking_matches_manual_masked_computation():
    """Mask, model GIRDISINI korumaz (kirli deger yine agdan gecer) -- yalniz o
    pozisyonun skor toplamina/paydasina girmesini engeller. Bu yuzden dogru test,
    ayni girdiyle uretilen gercek reconstruction'a karsi elle hesaplanmis maskeli
    ortalamayla karsilastirmak."""
    X, M = _toy_data(n=5)
    model, _ = train_dense_autoencoder(X, M, window=6, n_features=3, hidden_dims=(8, 4), epochs=3, seed=0)

    M_masked = M.copy()
    M_masked[:, 0, 0] = 0.0

    scores = reconstruction_scores(model, X, M_masked)

    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(X, dtype=torch.float32)).numpy()
    sq_err = (X - recon) ** 2 * M_masked
    denom = M_masked.sum(axis=(1, 2))
    expected = sq_err.sum(axis=(1, 2)) / denom
    np.testing.assert_allclose(scores, expected, rtol=1e-5)
