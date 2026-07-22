"""Small natural-only location/scale LSTM forecaster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

NATURAL_FIT_ROLE = "natural_clean_fit"


@dataclass(frozen=True)
class ForecasterConfig:
    input_features: int
    target_channels: int
    hidden_size: int
    num_layers: int
    min_scale: float
    max_scale: float

    def __post_init__(self) -> None:
        if min(self.input_features, self.target_channels, self.hidden_size, self.num_layers) < 1:
            raise ValueError("model dimensions must be positive")
        if not 0 < self.min_scale < self.max_scale:
            raise ValueError("scale limits must satisfy 0 < min < max")


class ResidualForecaster(nn.Module):
    def __init__(self, config: ForecasterConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.LSTM(
            config.input_features * 2,
            config.hidden_size,
            config.num_layers,
            batch_first=True,
        )
        self.location_head = nn.Linear(config.hidden_size, config.target_channels)
        self.scale_head = nn.Linear(config.hidden_size, config.target_channels)

    def forward(self, values: torch.Tensor, mask: torch.Tensor):
        if values.shape != mask.shape:
            raise ValueError("values and mask must have identical shapes")
        encoded = torch.cat((values * mask, mask), dim=-1)
        _, (hidden, _) = self.encoder(encoded)
        state = hidden[-1]
        location = self.location_head(state)
        scale = self.config.min_scale + (
            self.config.max_scale - self.config.min_scale
        ) * torch.sigmoid(self.scale_head(state))
        return location, scale


def train_forecaster(
    X: np.ndarray,
    X_mask: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    *,
    config: ForecasterConfig,
    channel_weights: tuple[float, ...],
    data_role: str,
    contains_synthetic: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[ResidualForecaster, list[float]]:
    if data_role != NATURAL_FIT_ROLE or contains_synthetic:
        raise ValueError("forecaster fit requires natural_clean_fit without synthetic rows")
    if X.shape != X_mask.shape or y.shape != y_mask.shape or len(X) == 0:
        raise ValueError("non-empty aligned value/mask arrays are required")
    torch.manual_seed(seed)
    model = ResidualForecaster(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tensors = [
        torch.as_tensor(array, dtype=torch.float32)
        for array in (X, X_mask, y, y_mask)
    ]
    weights = torch.as_tensor(channel_weights, dtype=torch.float32)
    history: list[float] = []
    for _ in range(epochs):
        permutation = torch.randperm(len(X))
        total = 0.0
        for start in range(0, len(X), batch_size):
            index = permutation[start : start + batch_size]
            xb, mb, yb, ymb = (tensor[index] for tensor in tensors)
            optimizer.zero_grad()
            location, scale = model(xb, mb)
            safe_scale = scale.clamp_min(torch.finfo(scale.dtype).tiny)
            nll = (torch.log(safe_scale) + 0.5 * ((yb - location) / safe_scale).square())
            weighted_mask = ymb * weights.unsqueeze(0)
            loss = (nll * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        history.append(total / len(X))
    return model, history


def score_forecaster(
    model: ResidualForecaster,
    X: np.ndarray,
    X_mask: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        values = torch.as_tensor(X, dtype=torch.float32)
        masks = torch.as_tensor(X_mask, dtype=torch.float32)
        target = torch.as_tensor(y, dtype=torch.float32)
        target_mask = torch.as_tensor(y_mask, dtype=torch.float32)
        location, scale = model(values, masks)
        scores = ((target - location).abs() / scale.clamp_min(1e-6)) * target_mask
    return scores.numpy()

