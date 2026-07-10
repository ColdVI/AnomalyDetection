"""USAD -- UnSupervised Anomaly Detection (Audibert ve ekibi, 2020).

Paylasilan bir kodlayici E, iki kod cozucu D1/D2. Iki fazli/celismeli egitim:
- AE1 = D1(E(x)), AE2 = D2(E(x))
- Faz agirliklari epoch ilerledikce degisir: alpha = 1/epoch_no (1-indeksli),
  beta = 1-alpha -- egitimin basinda ikisi de sadece kendi reconstruction'ina
  odaklanir, ilerledikce D1 D2'yi "kandirmaya", D2 D1'in ciktisini "gercekten
  ayirt etmeye" calisir (celismeli terim: D2(D1(x))).
- L_AE1 = alpha*||x-AE1(x)||^2 + beta*||x-D2(AE1(x))||^2
- L_AE2 = alpha*||x-AE2(x)||^2 - beta*||x-D2(AE1(x))||^2
- Skor (sabit alpha=beta=0.5): alpha*||x-AE1(x)||^2 + beta*||x-D2(AE1(x))||^2

Orijinal makale duz (Linear) katmanlar kullanir -- burada da ayni, pencere tek
vektore acilir (dense_autoencoder.py ile ayni desen).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class USAD(nn.Module):
    def __init__(self, window: int, n_features: int, latent_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.window = window
        self.n_features = n_features
        flat_dim = window * n_features

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, flat_dim),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    def _unflat(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.reshape(-1, self.window, self.n_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AE1(x), AE2(x), D2(AE1(x)) -- ucu de (batch, window, features)."""
        z = self.encoder(self._flat(x))
        ae1 = self._unflat(self.decoder1(z))
        ae2 = self._unflat(self.decoder2(z))
        z_ae1 = self.encoder(self._flat(ae1))
        ae2_of_ae1 = self._unflat(self.decoder2(z_ae1))
        return ae1, ae2, ae2_of_ae1


def _masked_sq_err_mean(x: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq_err = (x - recon) ** 2 * mask
    denom = mask.sum(dim=(-2, -1)).clamp(min=1.0)
    return (sq_err.sum(dim=(-2, -1)) / denom).mean()


def train_usad(
    X: np.ndarray, M: np.ndarray, *, window: int, n_features: int, latent_dim: int = 16,
    hidden_dim: int = 32, epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
    seed: int = 0, device: str = "cpu",
) -> tuple[USAD, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = USAD(window, n_features, latent_dim, hidden_dim).to(device)
    opt1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=lr
    )
    opt2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=lr
    )

    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    m_t = torch.tensor(M, dtype=torch.float32, device=device)
    n = len(x_t)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        alpha = 1.0 / epoch
        beta = 1.0 - alpha
        perm = torch.randperm(n)
        loss1_sum = loss2_sum = 0.0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, mb = x_t[idx], m_t[idx]

            # --- AE1 (D1) adimi ---
            ae1, _, ae2_of_ae1 = model(xb)
            loss1 = alpha * _masked_sq_err_mean(xb, ae1, mb) + beta * _masked_sq_err_mean(xb, ae2_of_ae1, mb)
            opt1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder.parameters()) + list(model.decoder1.parameters()), max_norm=1.0
            )
            opt1.step()

            # --- AE2 (D2) adimi -- ayri forward (grafik guncellenmis agirliklarla) ---
            ae1_b, ae2_b, ae2_of_ae1_b = model(xb)
            loss2 = alpha * _masked_sq_err_mean(xb, ae2_b, mb) - beta * _masked_sq_err_mean(xb, ae2_of_ae1_b, mb)
            opt2.zero_grad()
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder.parameters()) + list(model.decoder2.parameters()), max_norm=1.0
            )
            opt2.step()

            loss1_sum += loss1.item() * len(idx)
            loss2_sum += loss2.item() * len(idx)

        history.append({"epoch": epoch, "loss1": loss1_sum / max(n, 1), "loss2": loss2_sum / max(n, 1)})

    return model, history


def usad_scores(
    model: USAD, X: np.ndarray, M: np.ndarray, *, alpha: float = 0.5, beta: float = 0.5, device: str = "cpu",
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        m_t = torch.tensor(M, dtype=torch.float32, device=device)
        ae1, _, ae2_of_ae1 = model(x_t)
        sq_err1 = (x_t - ae1) ** 2 * m_t
        sq_err2 = (x_t - ae2_of_ae1) ** 2 * m_t
        denom = m_t.sum(dim=(-2, -1)).clamp(min=1.0)
        score = alpha * sq_err1.sum(dim=(-2, -1)) / denom + beta * sq_err2.sum(dim=(-2, -1)) / denom
    return score.cpu().numpy()
