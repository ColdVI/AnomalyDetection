"""ML-16 Kol N: magnitude-normalized post-hoc reconstruction scoring.

Two variants computed from an ALREADY-TRAINED, frozen AE/USAD checkpoint's raw
reconstruction (x, x_hat, mask) -- no retraining, no new model code, no change to
`masked_mse`/`reconstruction_scores`/`usad_reconstruction_scores`. See
docs/ML16_KOL_N_GENLIK_NORMALIZE_SKOR_PLAN.md SS1 for the pre-registered formulas and
SS4 for why this lives in one shared module instead of being duplicated per architecture
(no training code here, so the "each model family owns its runner" convention that
motivated Kol L/D/U's separate files does not apply).

House rule enforced throughout: a channel/window with no valid (non-masked) support is
NEVER silently imputed with a neutral/zero value -- it is excluded from the relevant
average, and a window with zero available channels stays NaN.
"""

from __future__ import annotations

import numpy as np
import torch

from src.ml.evaluation.score_fusion import empirical_probability
from src.ml.models.lstm_autoencoder import masked_mse_per_channel

# SS1(a): 10% of one RobustScaler IQR (RobustScaler centers on the median and scales by
# IQR, so scaled-unit Q1~=-0.5/Q3~=+0.5 -- see src/ml/data/scaling.py::fit_scaler_params).
# Large enough to avoid a division blow-up right at the scaled median (x~=0), small
# enough not to swallow genuine relative deviation for typical mid-range values. Fixed
# before any result was seen; not tuned post-hoc.
RELATIVE_ERROR_EPS = 0.1


def channel_squared_error(x: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray):
    """Per-window, per-channel mean squared error (averaged over valid timesteps only).

    Numpy convenience wrapper around `masked_mse_per_channel` -- kept torch-free at this
    boundary so the two new score formulas can be unit tested on plain numpy arrays.
    Returns (channel_mse (n, f) float, channel_valid (n, f) bool).
    """
    numerator, denominator = masked_mse_per_channel(
        torch.as_tensor(np.asarray(x, dtype=np.float32)),
        torch.as_tensor(np.asarray(reconstruction, dtype=np.float32)),
        torch.as_tensor(np.asarray(mask, dtype=np.float32)),
        per_sample=True,
    )
    numerator = numerator.numpy()
    denominator = denominator.numpy()
    channel_mse = numerator / np.clip(denominator, 1.0, None)
    return channel_mse, denominator > 0


def channel_relative_error(x: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray,
                           *, eps: float = RELATIVE_ERROR_EPS):
    """Per-window, per-channel mean of |x - x_hat| / (|x| + eps) over valid timesteps.

    Same reduction shape as `channel_squared_error` (sum over valid timesteps / valid
    count) but for the SS1(a) relative-error formula, which is a different quantity
    from masked_mse's squared error term and is not derivable from it.
    Returns (channel_relative_error (n, f) float, channel_valid (n, f) bool).

    Deliberately kept in float32 for the (n, window, f)-shaped intermediates (`x`/
    `reconstruction`/`mask` are float32 everywhere else in this codebase's AE pipeline;
    upcasting them to float64 here would roughly double peak memory on a full,
    unbatched split -- e.g. ~250k SEAD test windows x 50 x 22 -- for no accuracy benefit
    at this scale). Only the small (n, f) reduction outputs use float64.
    """
    x = np.asarray(x, dtype=np.float32)
    reconstruction = np.asarray(reconstruction, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    rel = (np.abs(x - reconstruction) / (np.abs(x) + np.float32(eps))) * mask
    numerator = rel.sum(axis=1, dtype=np.float64)
    denominator = mask.sum(axis=1, dtype=np.float64)
    channel_rel = numerator / np.clip(denominator, 1.0, None)
    return channel_rel, denominator > 0


def compute_channel_errors_zero_baseline(x: np.ndarray, mask: np.ndarray, *,
                                         eps: float = RELATIVE_ERROR_EPS, batch_size: int = 2048):
    """Model-free 'reconstruction = 0' magnitude-only null (SS3), batched the same way
    `compute_channel_errors_single_recon`/`compute_channel_errors_usad` batch a real
    model's forward pass -- bounds peak memory on a full split's windows instead of
    materializing one huge (n, window, f) intermediate array in a single call.
    Returns (channel_mse (n, f), channel_valid (n, f) bool, channel_relative_error (n, f)).
    """
    n = len(x)
    n_features = x.shape[-1] if x.ndim == 3 else 0
    mse_chunks, valid_chunks, rel_chunks = [], [], []
    for start in range(0, n, batch_size):
        xb = x[start:start + batch_size]
        mb = mask[start:start + batch_size]
        mse, valid = channel_squared_error(xb, 0.0, mb)
        rel, _ = channel_relative_error(xb, 0.0, mb, eps=eps)
        mse_chunks.append(mse)
        valid_chunks.append(valid)
        rel_chunks.append(rel)
    if not mse_chunks:
        empty = np.zeros((0, n_features))
        return empty, empty.astype(bool), empty.copy()
    return np.concatenate(mse_chunks), np.concatenate(valid_chunks), np.concatenate(rel_chunks)


def average_available_channels(channel_values: np.ndarray, channel_valid: np.ndarray) -> np.ndarray:
    """Row-wise mean over available (True) channels; NaN where none are available.

    This is the "average, not summed-then-squared" reduction that SS1 relies on to stop
    a single extreme-magnitude channel from dominating a window's score.
    """
    channel_values = np.asarray(channel_values, dtype=np.float64)
    valid = np.asarray(channel_valid, dtype=bool)
    counts = valid.sum(axis=1)
    total = np.where(valid, channel_values, 0.0).sum(axis=1)
    result = np.full(len(channel_values), np.nan)
    has_any = counts > 0
    result[has_any] = total[has_any] / counts[has_any]
    return result


def relative_error_window_scores(x: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray,
                                 *, eps: float = RELATIVE_ERROR_EPS) -> np.ndarray:
    """SS1(a) end-to-end: per-window relative-error score, averaged across available
    channels. This is the score BEFORE `_align_score`/`empirical_probability`
    calibration (matches how `lstm_recon_raw` etc. are raw window scores upstream of the
    same calibration steps)."""
    channel_rel, channel_valid = channel_relative_error(x, reconstruction, mask, eps=eps)
    return average_available_channels(channel_rel, channel_valid)


def per_channel_rank_normalized_scores(
    reference_channel_mse: np.ndarray, reference_channel_valid: np.ndarray,
    query_channel_mse: np.ndarray, query_channel_valid: np.ndarray,
) -> np.ndarray:
    """SS1(b): per-channel percentile of query error against that SAME channel's own
    reference (train-normal) error distribution, then averaged across channels
    available in BOTH the query window and the reference. A channel with an empty
    reference distribution (no valid train-normal windows for it) is excluded from
    every query window's average -- never defaulted to a neutral 0.5 percentile.
    """
    reference_channel_mse = np.asarray(reference_channel_mse, dtype=np.float64)
    reference_channel_valid = np.asarray(reference_channel_valid, dtype=bool)
    query_channel_mse = np.asarray(query_channel_mse, dtype=np.float64)
    query_channel_valid = np.asarray(query_channel_valid, dtype=bool)

    n_channels = reference_channel_mse.shape[1]
    percentiles = np.full(query_channel_mse.shape, np.nan)
    channel_has_reference = np.zeros(n_channels, dtype=bool)
    for c in range(n_channels):
        reference = reference_channel_mse[reference_channel_valid[:, c], c]
        if not len(reference):
            continue
        channel_has_reference[c] = True
        percentiles[:, c] = empirical_probability(reference, query_channel_mse[:, c])
    channel_valid = query_channel_valid & channel_has_reference[np.newaxis, :]
    return average_available_channels(percentiles, channel_valid)


def compute_channel_errors_single_recon(model, x: np.ndarray, mask: np.ndarray, *,
                                        eps: float = RELATIVE_ERROR_EPS, batch_size: int = 512):
    """Channel-level squared-error and relative-error arrays for single-reconstruction
    architectures (LSTM-AE, Dense-AE): one forward pass per batch feeds both formulas.

    Returns (channel_mse (n, f), channel_valid (n, f) bool, channel_relative_error (n, f)).
    """
    model.eval()
    n_features = x.shape[-1] if x.ndim == 3 else 0
    mse_chunks, valid_chunks, rel_chunks = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size])
            mb = torch.as_tensor(mask[start:start + batch_size])
            recon = model(xb)
            numerator, denominator = masked_mse_per_channel(xb, recon, mb, per_sample=True)
            denom_clamped = denominator.clamp(min=1.0)
            mse_chunks.append((numerator / denom_clamped).numpy())
            valid_chunks.append((denominator.numpy() > 0))
            rel = (torch.abs(xb - recon) / (torch.abs(xb) + eps)) * mb
            rel_chunks.append((rel.sum(dim=1) / denom_clamped).numpy())
    if not mse_chunks:
        empty = np.zeros((0, n_features))
        return empty, empty.astype(bool), empty.copy()
    return np.concatenate(mse_chunks), np.concatenate(valid_chunks), np.concatenate(rel_chunks)


def compute_channel_errors_usad(model, x: np.ndarray, mask: np.ndarray, *,
                                eps: float = RELATIVE_ERROR_EPS, batch_size: int = 512,
                                alpha: float = 0.5, beta: float = 0.5):
    """USAD variant: channel arrays are the SAME alpha/beta-weighted combination of
    AE1(w) and AE2(AE1(w)) that `usad_reconstruction_scores` uses for the raw MSE score
    -- applied identically to the relative-error formula so the "which reconstruction(s)
    feed the score" convention does not silently change between formulas.
    """
    model.eval()
    n_features = x.shape[-1] if x.ndim == 3 else 0
    mse_chunks, valid_chunks, rel_chunks = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size])
            mb = torch.as_tensor(mask[start:start + batch_size])
            w1 = model.ae1(xb)
            w3 = model.ae2(w1)
            num1, denominator = masked_mse_per_channel(xb, w1, mb, per_sample=True)
            num3, _ = masked_mse_per_channel(xb, w3, mb, per_sample=True)
            denom_clamped = denominator.clamp(min=1.0)
            channel_mse = alpha * (num1 / denom_clamped) + beta * (num3 / denom_clamped)
            mse_chunks.append(channel_mse.numpy())
            valid_chunks.append((denominator.numpy() > 0))
            rel1 = (torch.abs(xb - w1) / (torch.abs(xb) + eps)) * mb
            rel3 = (torch.abs(xb - w3) / (torch.abs(xb) + eps)) * mb
            channel_rel = alpha * (rel1.sum(dim=1) / denom_clamped) + beta * (rel3.sum(dim=1) / denom_clamped)
            rel_chunks.append(channel_rel.numpy())
    if not mse_chunks:
        empty = np.zeros((0, n_features))
        return empty, empty.astype(bool), empty.copy()
    return np.concatenate(mse_chunks), np.concatenate(valid_chunks), np.concatenate(rel_chunks)
