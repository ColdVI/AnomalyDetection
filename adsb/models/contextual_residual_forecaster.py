"""Small context-aware next-step forecaster for physics residual channels.

Unlike the legacy autoencoders, this model does not reconstruct raw altitude,
speed, or track.  It predicts a location and scale for each *separate* physics
residual at the next row.  Channel scores remain separate until an explicit,
externally budgeted decision layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NATURAL_FIT_ROLE = "natural_clean_fit"


@dataclass(frozen=True)
class ContextualForecasterConfig:
    input_features: int
    target_channels: int
    hidden_size: int
    num_layers: int
    min_scale: float
    max_scale: float

    def __post_init__(self) -> None:
        if min(self.input_features, self.target_channels, self.hidden_size, self.num_layers) < 1:
            raise ValueError("model dimensions and num_layers must be >= 1")
        if not np.isfinite(self.min_scale) or self.min_scale <= 0:
            raise ValueError("min_scale must be finite and > 0")
        if not np.isfinite(self.max_scale) or self.max_scale <= self.min_scale:
            raise ValueError("max_scale must be finite and exceed min_scale")


class ContextualResidualForecaster(nn.Module):
    """LSTM forecaster receiving values and explicit availability masks."""

    def __init__(self, config: ContextualForecasterConfig):
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

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.shape != mask.shape:
            raise ValueError("values and mask must have identical shapes")
        if values.ndim != 3 or values.shape[-1] != self.config.input_features:
            raise ValueError("unexpected contextual forecaster input shape")
        encoded_input = torch.cat((values * mask, mask), dim=-1)
        _, (hidden, _) = self.encoder(encoded_input)
        state = hidden[-1]
        location = self.location_head(state)
        unit_scale = torch.sigmoid(self.scale_head(state))
        scale = self.config.min_scale + (
            self.config.max_scale - self.config.min_scale
        ) * unit_scale
        return location, scale


def channelwise_gaussian_nll(
    target: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-cell NLL and standardized absolute surprise."""

    if not (target.shape == location.shape == scale.shape == target_mask.shape):
        raise ValueError("target, prediction, scale, and mask shapes must match")
    if target.ndim != 2:
        raise ValueError("targets must have shape (batch, channels)")
    safe_scale = scale.clamp_min(torch.finfo(scale.dtype).tiny)
    standardized = (target - location).abs() / safe_scale
    nll = (torch.log(safe_scale) + 0.5 * standardized.square()) * target_mask
    return nll, standardized * target_mask


def weighted_masked_channel_loss(
    nll: torch.Tensor,
    target_mask: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    """Explicitly weighted loss; silent equal-weight aggregation is forbidden."""

    if nll.shape != target_mask.shape or nll.ndim != 2:
        raise ValueError("nll and target_mask must be matching 2-D tensors")
    if channel_weights.ndim != 1 or len(channel_weights) != nll.shape[1]:
        raise ValueError("one explicit weight is required per target channel")
    if not torch.isfinite(channel_weights).all() or torch.any(channel_weights <= 0):
        raise ValueError("channel weights must be finite and > 0")
    weighted_mask = target_mask * channel_weights.unsqueeze(0)
    denominator = weighted_mask.sum().clamp_min(1.0)
    return (nll * channel_weights.unsqueeze(0)).sum() / denominator


def train_contextual_residual_forecaster(
    X: np.ndarray,
    X_mask: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    *,
    config: ContextualForecasterConfig,
    channel_weights: tuple[float, ...],
    data_role: str,
    contains_synthetic: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str = "cpu",
) -> tuple[ContextualResidualForecaster, list[float]]:
    """Fit on natural-clean normal data only and return epoch losses."""

    if data_role != NATURAL_FIT_ROLE:
        raise ValueError("Only natural_clean_fit may train the contextual forecaster")
    if contains_synthetic:
        raise ValueError("Synthetic data cannot enter contextual forecaster training")
    if X.shape != X_mask.shape or y.shape != y_mask.shape:
        raise ValueError("value/mask array shapes must match")
    if X.ndim != 3 or y.ndim != 2 or len(X) != len(y) or len(X) == 0:
        raise ValueError("non-empty aligned 3-D inputs and 2-D targets are required")
    if X.shape[-1] != config.input_features or y.shape[-1] != config.target_channels:
        raise ValueError("arrays do not match the frozen model config")
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    torch.manual_seed(seed)
    model = ContextualResidualForecaster(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tensors = [
        torch.as_tensor(array, dtype=torch.float32, device=device)
        for array in (X, X_mask, y, y_mask)
    ]
    weights = torch.as_tensor(channel_weights, dtype=torch.float32, device=device)
    if len(weights) != config.target_channels:
        raise ValueError("channel_weights must match target_channels")

    history: list[float] = []
    for _ in range(epochs):
        permutation = torch.randperm(len(X), device=device)
        epoch_sum = 0.0
        for start in range(0, len(X), batch_size):
            index = permutation[start : start + batch_size]
            xb, mb, yb, ymb = (tensor[index] for tensor in tensors)
            optimizer.zero_grad()
            location, scale = model(xb, mb)
            nll, _ = channelwise_gaussian_nll(yb, location, scale, ymb)
            loss = weighted_masked_channel_loss(nll, ymb, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_sum += float(loss.detach()) * len(index)
        history.append(epoch_sum / len(X))
    return model, history


def contextual_channel_scores(
    model: ContextualResidualForecaster,
    X: np.ndarray,
    X_mask: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return channel-wise surprise, predicted location, and predicted scale."""

    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(X, dtype=torch.float32, device=device)
        mb = torch.as_tensor(X_mask, dtype=torch.float32, device=device)
        yb = torch.as_tensor(y, dtype=torch.float32, device=device)
        ymb = torch.as_tensor(y_mask, dtype=torch.float32, device=device)
        location, scale = model(xb, mb)
        _, scores = channelwise_gaussian_nll(yb, location, scale, ymb)
    return (
        scores.cpu().numpy(),
        location.cpu().numpy(),
        scale.cpu().numpy(),
    )
