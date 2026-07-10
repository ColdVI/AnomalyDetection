"""LSTM kodlayici-kodcozucu (seq2seq) otokodlayici -- zamansal orunutu ogrenmesi
umulan mimari. Kodlayici son gizli durumu z'ye sikistirir; kod cozucu z'yi her
zaman adiminda tekrar girdi olarak alip pencereyi yeniden kurar (repeat-vector
deseni). Egitim/skor kurallari dense_autoencoder.py ile ayni (yalniz normal veri,
maskeli MSE, egitimden sonra magnitude_domination_check zorunlu).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from adsb.windowing import masked_mse


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, window, _ = x.shape
        _, (h_n, _) = self.encoder(x)
        z = h_n[-1]  # (batch, hidden_size) -- son katmanin son gizli durumu
        decoder_input = z.unsqueeze(1).expand(batch, window, self.hidden_size)
        dec_out, _ = self.decoder(decoder_input)
        return self.output_layer(dec_out)


def train_lstm_autoencoder(
    X: np.ndarray, M: np.ndarray, *, n_features: int, hidden_size: int = 32, num_layers: int = 1,
    epochs: int = 30, batch_size: int = 64, lr: float = 1e-3, seed: int = 0, device: str = "cpu",
) -> tuple[LSTMAutoencoder, list[float]]:
    torch.manual_seed(seed)
    model = LSTMAutoencoder(n_features, hidden_size, num_layers).to(device)
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


def reconstruction_scores(model: LSTMAutoencoder, X: np.ndarray, M: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        m_t = torch.tensor(M, dtype=torch.float32, device=device)
        recon = model(x_t)
        scores = masked_mse(x_t, recon, m_t)
    return scores.cpu().numpy()
