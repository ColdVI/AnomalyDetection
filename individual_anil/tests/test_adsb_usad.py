"""adsb/models/usad.py testleri."""

from __future__ import annotations

import numpy as np
import torch

from adsb.models.usad import USAD, train_usad, usad_scores


def _toy_data(n=40, window=6, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, window, n_features)).astype(np.float32)
    M = np.ones_like(X, dtype=np.float32)
    return X, M


def test_forward_shapes():
    model = USAD(window=6, n_features=3, latent_dim=4, hidden_dim=8)
    x = torch.randn(5, 6, 3)
    ae1, ae2, ae2_of_ae1 = model(x)
    assert ae1.shape == (5, 6, 3)
    assert ae2.shape == (5, 6, 3)
    assert ae2_of_ae1.shape == (5, 6, 3)


def test_training_reduces_raw_reconstruction_error():
    """Alpha/beta agirliklari epoch'a gore degistigi icin ham loss1/loss2 izlemesi
    yaniltici olabilir -- bunun yerine sabit alpha=1 (yalniz AE1 hatasi) ile egitim
    oncesi/sonrasi karsilastirma yapiyoruz."""
    X, M = _toy_data()
    torch.manual_seed(0)
    untrained = USAD(window=6, n_features=3, latent_dim=4, hidden_dim=8)
    baseline_scores = usad_scores(untrained, X, M, alpha=1.0, beta=0.0)

    model, _ = train_usad(X, M, window=6, n_features=3, latent_dim=4, hidden_dim=8, epochs=25, seed=0)
    trained_scores = usad_scores(model, X, M, alpha=1.0, beta=0.0)

    assert trained_scores.mean() < baseline_scores.mean()


def test_training_is_deterministic_given_seed():
    X, M = _toy_data()
    model1, h1 = train_usad(X, M, window=6, n_features=3, latent_dim=4, hidden_dim=8, epochs=5, seed=42)
    model2, h2 = train_usad(X, M, window=6, n_features=3, latent_dim=4, hidden_dim=8, epochs=5, seed=42)
    assert h1 == h2
    np.testing.assert_array_equal(usad_scores(model1, X, M), usad_scores(model2, X, M))


def test_masking_matches_manual_masked_computation():
    X, M = _toy_data(n=5)
    model, _ = train_usad(X, M, window=6, n_features=3, latent_dim=4, hidden_dim=8, epochs=3, seed=0)

    M_masked = M.copy()
    M_masked[:, 0, 0] = 0.0

    scores = usad_scores(model, X, M_masked, alpha=0.5, beta=0.5)

    model.eval()
    with torch.no_grad():
        ae1, _, ae2_of_ae1 = model(torch.tensor(X, dtype=torch.float32))
    ae1, ae2_of_ae1 = ae1.numpy(), ae2_of_ae1.numpy()
    denom = M_masked.sum(axis=(1, 2))
    sq_err1 = ((X - ae1) ** 2 * M_masked).sum(axis=(1, 2)) / denom
    sq_err2 = ((X - ae2_of_ae1) ** 2 * M_masked).sum(axis=(1, 2)) / denom
    expected = 0.5 * sq_err1 + 0.5 * sq_err2
    np.testing.assert_allclose(scores, expected, rtol=1e-5)
