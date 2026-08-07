"""Normal-only, maskeli DUZ (recurrent olmayan) autoencoder -- ML-16 Kol D.

`src/ml/models/lstm_autoencoder.py`'nin duz-feedforward karsiligi. Ayni
`AE_FEATURES["uav_sead"]`/`WINDOW["uav_sead"]`/`STRIDE["uav_sead"]` sabitlerini
ve ayni `src/ml/data/windowing.py::build_windows` cikisini (X: (n, window, f)
NaN->0, M: (n, window, f) 1=var/0=eksik) kullanir. TEK mimari fark: pencere
LSTM'e sequence olarak degil, tek bir duz vektore (window*f) flatten edilip
simetrik fully-connected encoder/decoder'a verilir -- boylece zaman sirasi
bilgisi modele hic girmiyor (bkz. docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md SS0).

`masked_mse` ve `reconstruction_scores` `lstm_autoencoder.py`'den DEGISTIRILMEDEN
import edilir: ikisi de yalniz tensor sekli/`model(x)` cagrisina bagli, mimariye
bagimli degil -- ayni maskeli-kayip semantigini iki model ailesinde birebir
tutmak icin yeniden yazilmadi (adil karsilastirma sarti).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from src.ml.models.lstm_autoencoder import masked_mse, reconstruction_scores

__all__ = ["DenseAutoencoder", "masked_mse", "reconstruction_scores", "train_dense_autoencoder"]


class DenseAutoencoder(nn.Module):
    """Pencereyi (window, n_features) -> duz vektor -> simetrik FC AE.

    `hidden`/`latent` varsayilanlari `LSTMAutoencoder(n_features=22, hidden=32,
    latent=16)`nin toplam parametre sayisiyla (17414) kabaca eslesecek sekilde
    secildi (docs/ML16_KOL_D_DENSE_AE_SEAD_PLAN.md SS2): window=50, n_features=22
    icin flat=1100, hidden=7, latent=4 -> toplam 16574 parametre (%4.8 daha az,
    dogrulama: `python -c "from src.ml.models.lstm_autoencoder import
    LSTMAutoencoder; from src.ml.models.dense_autoencoder import
    DenseAutoencoder; ..."`).
    Amac: performans farki varsa bunun kapasiteden degil mimariden (sequence
    modelleme yoklugu) geldigini gosterebilmek.
    """

    def __init__(self, window: int, n_features: int, hidden: int = 7, latent: int = 4):
        super().__init__()
        self.window = window
        self.n_features = n_features
        flat = window * n_features
        self.encoder = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU())
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU())
        self.decoder = nn.Linear(hidden, flat)

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    def _unflatten(self, w: torch.Tensor) -> torch.Tensor:
        return w.reshape(w.shape[0], self.window, self.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.to_latent(self.encoder(self._flatten(x)))
        recon = self.decoder(self.from_latent(z))
        return self._unflatten(recon)


def train_dense_autoencoder(model, x_train, m_train, x_val, m_val, *,
                            seed: int, epochs: int = 40, batch_size: int = 64,
                            learning_rate: float = 1e-3, patience: int = 5):
    """`train_lstm_autoencoder` ile ayni egitim dongusu/erken-durdurma mantigi
    (ayni epoch/batch/optimizer/patience semantigi) -- LSTM'e ozgu hicbir sey
    yok, ama her model ailesi kendi dosyasinda kendi egitim girisini tasir
    (lstm_autoencoder.py de ayni deseni izliyor: paylasilan sadece maskeli-kayip
    fonksiyonu, egitim dongusu degil)."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    xt, mt = torch.tensor(x_train), torch.tensor(m_train)
    xv, mv = torch.tensor(x_val), torch.tensor(m_val)
    best_loss, best_state, bad = np.inf, None, 0
    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(xt))
        epoch_losses = []
        for start in range(0, len(permutation), batch_size):
            idx = permutation[start:start + batch_size]
            loss = masked_mse(xt[idx], model(xt[idx]), mt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(masked_mse(xv, model(xv), mv))
        history.append({"epoch": epoch + 1,
                        "train_loss": float(np.mean(epoch_losses)),
                        "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("Dense-AE training en iyi state uretmedi")
    model.load_state_dict(best_state)
    return model, {"best_val_loss": best_loss, "epochs": epoch + 1,
                   "history": history}
