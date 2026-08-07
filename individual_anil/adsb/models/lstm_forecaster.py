"""LSTM ileri-tahminci (forecaster) -- LSTM-AE'nin "turevi": pencereyi yeniden
kurmak yerine, gecmis kisimdan gelecek `horizon` adimini tahmin eder. Anomali
skoru tahmin-residual'i -- reconstruction'dan FARKLI bir mekanizma (kopyalamayi
ogrenemez, gercekten "sonraki durum ne olmali" sorusuna cevap vermek zorunda).

Pencere `window = history_len + horizon` uzunlugunda gelir; ilk `history_len`
adim girdi, son `horizon` adim hedef.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.n_features = n_features
        self.horizon = horizon
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, n_features * horizon)

    def forward(self, x_history: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x_history)
        z = h_n[-1]
        pred_flat = self.output_layer(z)
        return pred_flat.reshape(-1, self.horizon, self.n_features)


def _split(X: np.ndarray, M: np.ndarray, *, history_len: int):
    return X[:, :history_len], M[:, :history_len], X[:, history_len:], M[:, history_len:]


def _masked_mse(x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq_err = (x - pred) ** 2 * mask
    denom = mask.sum(dim=(-2, -1)).clamp(min=1.0)
    return sq_err.sum(dim=(-2, -1)) / denom


def train_lstm_forecaster(
    X: np.ndarray, M: np.ndarray, *, history_len: int, n_features: int, hidden_size: int = 32,
    num_layers: int = 1, epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
    seed: int = 0, device: str = "cpu",
) -> tuple[LSTMForecaster, list[float]]:
    horizon = X.shape[1] - history_len
    if horizon <= 0:
        raise ValueError(f"window ({X.shape[1]}) history_len'den ({history_len}) buyuk olmali")

    torch.manual_seed(seed)
    model = LSTMForecaster(n_features, horizon, hidden_size, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_hist, m_hist, x_target, m_target = _split(X, M, history_len=history_len)
    x_hist_t = torch.tensor(x_hist, dtype=torch.float32, device=device)
    x_target_t = torch.tensor(x_target, dtype=torch.float32, device=device)
    m_target_t = torch.tensor(m_target, dtype=torch.float32, device=device)
    n = len(x_hist_t)
    history: list[float] = []

    for _epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb, mb = x_hist_t[idx], x_target_t[idx], m_target_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = _masked_mse(yb, pred, mb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history.append(epoch_loss / max(n, 1))

    return model, history


def forecast_residual_scores(
    model: LSTMForecaster, X: np.ndarray, M: np.ndarray, *, history_len: int, device: str = "cpu",
) -> np.ndarray:
    model.eval()
    x_hist, _, x_target, m_target = _split(X, M, history_len=history_len)
    with torch.no_grad():
        x_hist_t = torch.tensor(x_hist, dtype=torch.float32, device=device)
        x_target_t = torch.tensor(x_target, dtype=torch.float32, device=device)
        m_target_t = torch.tensor(m_target, dtype=torch.float32, device=device)
        pred = model(x_hist_t)
        scores = _masked_mse(x_target_t, pred, m_target_t)
    return scores.cpu().numpy()
