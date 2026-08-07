"""USAD (UnSupervised Anomaly Detection) -- ML-16 Kol U.

Audibert, J. et al. (2020), "USAD: UnSupervised Anomaly Detection on
Multivariate Time Series", KDD 2020. Implements the paper's shared-encoder,
twin-decoder architecture with the two-phase adversarial training schedule
(Section 3.3/Algorithm 1 of the paper) and the standard alpha=0.5/beta=0.5
inference-time score combination (Section 3.4). The per-batch training-step
formulation and the "call forward twice per batch, once per optimizer" update
pattern below follow the canonical public reference implementation
(manigalati/usad on GitHub, the de-facto USAD reference used across follow-up
papers) rather than re-deriving an ad hoc variant.

Architecture:
  - Shared encoder E: flattened window -> latent z
  - Decoder1 D1, Decoder2 D2 (independent, same shapes): z -> reconstruction
  - AE1(w) = D1(E(w)), AE2(w) = D2(E(w))

Two-phase loss at epoch n (1-indexed, matches the paper's alpha=1/n,
beta=1-1/n schedule -- early epochs weight plain reconstruction, later epochs
weight the adversarial term more):
  L_AE1 = (1/n) * ||w - AE1(w)||^2 + (1 - 1/n) * ||w - AE2(AE1(w))||^2
  L_AE2 = (1/n) * ||w - AE2(w)||^2 - (1 - 1/n) * ||w - AE2(AE1(w))||^2

AE1 (encoder+D1) is trained to minimize L_AE1: reconstruct w directly AND fool
AE2 into reconstructing AE1's output well. AE2 (encoder+D2) is trained to
minimize L_AE2: reconstruct w directly AND (via the minus sign) get WORSE at
reconstructing AE1's output -- i.e. learn to tell real windows from AE1's
reconstructions apart. This is exactly the paper's adversarial pair.

Same masked-input handling as the rest of this codebase's AE family:
`masked_mse` from `src/ml/models/lstm_autoencoder.py` is reused unmodified
(house rule against ghost cross-flight/cross-source imputation -- see
docs/ML_YETERSIZLIKLER_KAYDI.md; missing channels are 0-filled by
`build_windows` and excluded from the loss via the mask, never imputed with
another flight's statistics).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from src.ml.models.lstm_autoencoder import masked_mse

__all__ = ["USAD", "masked_mse", "train_usad", "usad_reconstruction_scores"]


class USAD(nn.Module):
    """Shared encoder + twin decoders over a flattened (window, n_features) input.

    `hidden`/`latent` reuse the same values as `DenseAutoencoder` for
    consistency between the two new model families (docs/ML16_KOL_U_USAD_SEAD_PLAN.md
    SS2); USAD's total parameter count is necessarily larger than a single AE's
    because it has two decoders sharing one encoder -- this is an architectural
    property of USAD itself (Audibert et al. 2020), not a capacity choice made
    here to favor it.
    """

    def __init__(self, window: int, n_features: int, hidden: int = 7, latent: int = 4):
        super().__init__()
        self.window = window
        self.n_features = n_features
        flat = window * n_features
        self.encoder = nn.Sequential(
            nn.Linear(flat, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, flat))
        self.decoder2 = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, flat))

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    def _unflatten(self, w: torch.Tensor) -> torch.Tensor:
        return w.reshape(w.shape[0], self.window, self.n_features)

    def ae1(self, x: torch.Tensor) -> torch.Tensor:
        """AE1(x) = D1(E(x))."""
        return self._unflatten(self.decoder1(self.encoder(self._flatten(x))))

    def ae2(self, x: torch.Tensor) -> torch.Tensor:
        """AE2(x) = D2(E(x))."""
        return self._unflatten(self.decoder2(self.encoder(self._flatten(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default single-reconstruction path (AE1) -- used only where a
        generic "one model, one reconstruction" interface is required (none of
        the USAD training/scoring code below uses this; it exists so USAD can
        be smoke-tested with the same shape conventions as the other AE
        families)."""
        return self.ae1(x)

    def training_losses(self, x: torch.Tensor, mask: torch.Tensor, epoch_n: int):
        """Return (loss1, loss2) for epoch ``epoch_n`` (1-indexed) per the
        paper's alpha=1/n, beta=1-1/n schedule, masked-MSE terms throughout."""
        if epoch_n < 1:
            raise ValueError("epoch_n 1-indexed olmali (>=1)")
        w1 = self.ae1(x)
        w2 = self.ae2(x)
        w3 = self.ae2(w1)
        term1 = masked_mse(x, w1, mask)
        term2 = masked_mse(x, w2, mask)
        term_adv = masked_mse(x, w3, mask)
        inv_n = 1.0 / epoch_n
        loss1 = inv_n * term1 + (1.0 - inv_n) * term_adv
        loss2 = inv_n * term2 - (1.0 - inv_n) * term_adv
        return loss1, loss2

    def validation_reconstruction_quality(self, x: torch.Tensor, mask: torch.Tensor) -> float:
        """Sum of both decoders' *direct* (unweighted, non-adversarial) masked
        MSE against true normal windows. Used as the early-stopping/model-
        selection criterion instead of the signed training loss: L_AE2 has a
        subtracted adversarial term and is not a monotonic "smaller = better
        reconstruction" quantity by itself, so it is unsuitable for picking
        the best epoch. This is a documented implementation choice beyond the
        literal paper (which trains for a fixed epoch budget and does not
        specify a validation/early-stopping criterion)."""
        w1 = self.ae1(x)
        w2 = self.ae2(x)
        return float(masked_mse(x, w1, mask) + masked_mse(x, w2, mask))


def train_usad(model: USAD, x_train, m_train, x_val, m_val, *,
               seed: int, epochs: int = 40, batch_size: int = 64,
               learning_rate: float = 1e-3, patience: int = 5):
    """Two-optimizer training loop (opt1 over encoder+decoder1, opt2 over
    encoder+decoder2), matching the canonical USAD reference implementation's
    per-batch pattern: recompute the forward pass once per optimizer step
    (rather than calling .backward() twice on one shared graph) so each
    phase's gradients are independent, exactly as in the reference training
    loop this module cites in its module docstring."""
    torch.manual_seed(seed)
    opt1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=learning_rate)
    opt2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=learning_rate)
    xt, mt = torch.tensor(x_train), torch.tensor(m_train)
    xv, mv = torch.tensor(x_val), torch.tensor(m_val)
    best_loss, best_state, bad = np.inf, None, 0
    history: list[dict] = []
    for epoch in range(epochs):
        epoch_n = epoch + 1
        model.train()
        permutation = torch.randperm(len(xt))
        losses1, losses2 = [], []
        for start in range(0, len(permutation), batch_size):
            idx = permutation[start:start + batch_size]
            xb, mb = xt[idx], mt[idx]

            loss1, _ = model.training_losses(xb, mb, epoch_n)
            opt1.zero_grad()
            loss1.backward()
            opt1.step()

            _, loss2 = model.training_losses(xb, mb, epoch_n)
            opt2.zero_grad()
            loss2.backward()
            opt2.step()

            losses1.append(float(loss1.detach()))
            losses2.append(float(loss2.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = model.validation_reconstruction_quality(xv, mv)
        history.append({"epoch": epoch_n,
                        "train_loss": float(np.mean(losses1) + np.mean(losses2)),
                        "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("USAD training en iyi state uretmedi")
    model.load_state_dict(best_state)
    return model, {"best_val_loss": best_loss, "epochs": epoch_n,
                   "history": history}


def usad_reconstruction_scores(model: USAD, x, mask, *, alpha: float = 0.5, beta: float = 0.5,
                               batch_size: int = 512) -> np.ndarray:
    """Paper's standard inference score (Section 3.4): alpha*||w-AE1(w)||^2 +
    beta*||w-AE2(AE1(w))||^2, masked-MSE per sample. alpha=beta=0.5 is the
    paper's/reference implementation's default (equal weighting; alpha+beta=1
    with the ratio tunable as an operating-point knob, unused here -- no
    result-dependent tuning per this phase's pre-registration)."""
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start:start + batch_size])
            mb = torch.tensor(mask[start:start + batch_size])
            w1 = model.ae1(xb)
            w3 = model.ae2(w1)
            term1 = masked_mse(xb, w1, mb, per_sample=True)
            term3 = masked_mse(xb, w3, mb, per_sample=True)
            output.append((alpha * term1 + beta * term3).numpy())
    return np.concatenate(output) if output else np.array([])
