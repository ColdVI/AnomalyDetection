"""Duz (flat) feed-forward otokodlayici -- pencereyi tek vektore acip yeniden kurar.

Yalniz normal (bozulmamis) ucus pencereleriyle egitilir -- novelty detection: etiket
egitimde hic kullanilmaz. `mask` girdisi kayip degerleri egitimden/skordan disler
(bkz. adsb/windowing.py). Skor genligi tek basina anlamli sayilmaz -- egitimden
sonra `adsb/diagnostics.py::magnitude_domination_check` ZORUNLU calistirilir
(SEAD dersi: bkz. archive/2026-07-10_legacy_non_adsb_ml/docs/decisions.md ADR-016).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from adsb.windowing import masked_mse


class DenseAutoencoder(nn.Module):
    def __init__(self, window: int, n_features: int, hidden_dims: tuple[int, ...] = (64, 32, 16)):
        super().__init__()
        self.window = window
        self.n_features = n_features
        flat_dim = window * n_features

        enc_dims = [flat_dim, *hidden_dims]
        encoder_layers: list[nn.Module] = []
        for i in range(len(enc_dims) - 1):
            encoder_layers += [nn.Linear(enc_dims[i], enc_dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # son ReLU'yu at (latent serbest)

        dec_dims = [*hidden_dims[::-1], flat_dim]
        decoder_layers: list[nn.Module] = []
        for i in range(len(dec_dims) - 1):
            decoder_layers += [nn.Linear(dec_dims[i], dec_dims[i + 1])]
            if i < len(dec_dims) - 2:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        z = self.encoder(x.reshape(batch, -1))
        recon = self.decoder(z)
        return recon.reshape(batch, self.window, self.n_features)


def train_dense_autoencoder(
    X: np.ndarray, M: np.ndarray, *, window: int, n_features: int,
    hidden_dims: tuple[int, ...] = (64, 32, 16), epochs: int = 30, batch_size: int = 64,
    lr: float = 1e-3, seed: int = 0, device: str = "cpu",
) -> tuple[DenseAutoencoder, list[float]]:
    torch.manual_seed(seed)
    model = DenseAutoencoder(window, n_features, hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    m_t = torch.tensor(M, dtype=torch.float32, device=device)
    n = len(x_t)
    history: list[float] = []

    for _epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, mb = x_t[idx], m_t[idx]
            optimizer.zero_grad()
            recon = model(xb)
            loss = masked_mse(xb, recon, mb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history.append(epoch_loss / max(n, 1))

    return model, history


def reconstruction_scores(model: DenseAutoencoder, X: np.ndarray, M: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        m_t = torch.tensor(M, dtype=torch.float32, device=device)
        recon = model(x_t)
        scores = masked_mse(x_t, recon, m_t)
    return scores.cpu().numpy()
