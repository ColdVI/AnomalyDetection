"""Capacity-controlled deep models and normal-only training for RflyMAD."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rfly_dl.config import (
    BATCH_SIZE,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
)


def masked_mse(
    x: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-window MSE over observed cells only."""
    error = (x - reconstruction).square() * mask
    denominator = mask.sum(dim=(-2, -1)).clamp(min=1.0)
    return error.sum(dim=(-2, -1)) / denominator


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, latent: int = 16):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = self.to_latent(hidden[-1])
        repeated = self.from_latent(latent).unsqueeze(1).expand(-1, x.shape[1], -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


class DenseAutoencoder(nn.Module):
    def __init__(
        self, window: int, n_features: int, hidden: int = 6, latent: int = 4
    ):
        super().__init__()
        self.window = window
        self.n_features = n_features
        flat = window * n_features
        self.encoder = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU())
        self.to_latent = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, flat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        reconstruction = self.decoder(self.to_latent(self.encoder(flat)))
        return reconstruction.reshape(-1, self.window, self.n_features)


class USAD(nn.Module):
    def __init__(
        self, window: int, n_features: int, hidden: int = 6, latent: int = 4
    ):
        super().__init__()
        self.window = window
        self.n_features = n_features
        flat = window * n_features
        self.encoder = nn.Sequential(
            nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, latent), nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, flat)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, flat)
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.reshape(x.shape[0], -1))

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self.window, self.n_features)

    def ae1(self, x: torch.Tensor) -> torch.Tensor:
        return self._reshape(self.decoder1(self._encode(x)))

    def ae2(self, x: torch.Tensor) -> torch.Tensor:
        return self._reshape(self.decoder2(self._encode(x)))

    def ae2_of_ae1(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae2(self.ae1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae1(x)


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    history: list[dict[str, float | int]]
    best_epoch: int
    best_val_loss: float
    parameter_count: int


def make_model(
    model_name: str, *, window: int, n_features: int, seed: int
) -> nn.Module:
    torch.manual_seed(seed)
    if model_name == "lstm_ae":
        return LSTMAutoencoder(n_features)
    if model_name == "dense_ae":
        return DenseAutoencoder(window, n_features)
    if model_name == "usad":
        return USAD(window, n_features)
    raise ValueError(f"Unknown model: {model_name}")


def _validation_loss(
    model_name: str,
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size]
            mb = mask[start : start + batch_size]
            if model_name == "usad":
                assert isinstance(model, USAD)
                loss = masked_mse(xb, model.ae1(xb), mb) + masked_mse(
                    xb, model.ae2(xb), mb
                )
            else:
                loss = masked_mse(xb, model(xb), mb)
            values.append(loss.cpu().numpy())
    return float(np.concatenate(values).mean()) if values else float("inf")


def train_model(
    model_name: str,
    model: nn.Module,
    x_train: np.ndarray,
    m_train: np.ndarray,
    x_val: np.ndarray,
    m_val: np.ndarray,
    *,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    device: str = "cpu",
) -> TrainingResult:
    if not len(x_train) or not len(x_val):
        raise ValueError("Training and validation windows must be non-empty")
    torch.manual_seed(seed)
    model = model.to(device)
    xt = torch.from_numpy(x_train).to(device)
    mt = torch.from_numpy(m_train).to(device)
    xv = torch.from_numpy(x_val).to(device)
    mv = torch.from_numpy(m_val).to(device)

    if model_name == "usad":
        assert isinstance(model, USAD)
        optimizer1 = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder1.parameters()),
            lr=learning_rate,
        )
        optimizer2 = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder2.parameters()),
            lr=learning_rate,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(xt), device=device)
        train_losses: list[float] = []
        for start in range(0, len(permutation), batch_size):
            index = permutation[start : start + batch_size]
            xb, mb = xt[index], mt[index]
            if model_name == "usad":
                assert isinstance(model, USAD)
                alpha = 1.0 / epoch
                beta = 1.0 - alpha

                ae1 = model.ae1(xb)
                loss1 = (
                    alpha * masked_mse(xb, ae1, mb).mean()
                    + beta * masked_mse(xb, model.ae2(ae1), mb).mean()
                )
                optimizer1.zero_grad()
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.encoder.parameters())
                    + list(model.decoder1.parameters()),
                    GRADIENT_CLIP,
                )
                optimizer1.step()

                ae2 = model.ae2(xb)
                ae1_second = model.ae1(xb)
                loss2 = (
                    alpha * masked_mse(xb, ae2, mb).mean()
                    - beta * masked_mse(xb, model.ae2(ae1_second), mb).mean()
                )
                optimizer2.zero_grad()
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.encoder.parameters())
                    + list(model.decoder2.parameters()),
                    GRADIENT_CLIP,
                )
                optimizer2.step()
                train_losses.append(float((loss1 + loss2).detach().cpu()))
            else:
                reconstruction = model(xb)
                loss = masked_mse(xb, reconstruction, mb).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

        val_loss = _validation_loss(
            model_name, model, xv, mv, batch_size=max(batch_size, 512)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
            }
        )
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"{model_name} did not produce a finite validation state")
    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def reconstruction_scores(
    model_name: str,
    model: nn.Module,
    x: np.ndarray,
    mask: np.ndarray,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            mb = torch.from_numpy(mask[start : start + batch_size]).to(device)
            if model_name == "usad":
                assert isinstance(model, USAD)
                ae1 = model.ae1(xb)
                score = 0.5 * masked_mse(xb, ae1, mb) + 0.5 * masked_mse(
                    xb, model.ae2(ae1), mb
                )
            else:
                score = masked_mse(xb, model(xb), mb)
            result.append(score.cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, dtype=np.float32)
