"""Item tower prototip: ham havalimani feature'lari -> projeksiyon (h^0) ->
GCN mesaj-gecisi (h^final). Henuz egitim/loss YOK -- sadece mimarinin uctan
uca (forward pass) gercek havalimani graf/feature verisiyle calistigini
dogruluyor.

Girdi dosyalari (bu klasorde):
  - airport_features.parquet     (build_airport_features.py ciktisi)
  - route_scan_full_archive.json (check_full_archive_incremental.py ciktisi,
    route_counts iceriyor -- rota grafiginin kenarlari)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).parent
FEATURES_PATH = HERE / "airport_features.parquet"
CHECKPOINT_PATH = HERE / "route_scan_full_archive.json"


class AirportProjection(nn.Module):
    """Ham feature vektorunu (x_a) baslangic embedding'ine (h_a^0) cevirir.
    Her havalimanini BAGIMSIZ isler -- graf yapisini KULLANMAZ, o GCN'in isi."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GCNLayer(nn.Module):
    """Standart Kipf&Welling GCN katmani: H' = A_norm @ (H @ W)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ self.linear(h)


class AirportGCN(nn.Module):
    """Tam item tower: projeksiyon + N katman GCN mesaj-gecisi."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_gcn_layers: int = 2):
        super().__init__()
        self.proj = AirportProjection(in_dim, hidden_dim, hidden_dim)
        dims = [hidden_dim] + [out_dim] * n_gcn_layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(dims[i], dims[i + 1]) for i in range(n_gcn_layers)
        ])

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for i, layer in enumerate(self.gcn_layers):
            h = layer(h, a_norm)
            if i < len(self.gcn_layers) - 1:
                h = torch.relu(h)
        return h


def build_normalized_adjacency(route_counts: dict[tuple, int], idents: list[str]) -> torch.Tensor:
    """Rota sayaclarindan simetrik-normalize edilmis (Kipf&Welling) komsuluk
    matrisini kurar. Yon simdilik goz ardi ediliyor (A->B ve B->A agirliklari
    toplanip simetrik kenar yapiliyor) -- ilk prototip icin basitlestirme."""
    n = len(idents)
    idx = {a: i for i, a in enumerate(idents)}
    A = np.zeros((n, n), dtype=np.float32)
    for (o, d), w in route_counts.items():
        if o in idx and d in idx:
            i, j = idx[o], idx[d]
            A[i, j] += w
            A[j, i] += w  # simetriklestir

    A_hat = A + np.eye(n, dtype=np.float32)  # self-loop
    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


def load_feature_tensor(df: pd.DataFrame) -> tuple[torch.Tensor, list[str]]:
    idents = df["ident"].tolist()
    numeric = df.drop(columns=["ident", "iso_country"]).copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.fillna(0.0)  # kolon tamamen NaN ise median de NaN kalir
    x = torch.tensor(numeric.to_numpy(dtype=np.float32))
    return x, idents


def main():
    df = pd.read_parquet(FEATURES_PATH)
    x, idents = load_feature_tensor(df)
    print(f"Feature tensoru: {tuple(x.shape)} (havalimani x feature)")

    raw = json.loads(CHECKPOINT_PATH.read_text())
    route_counts = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})
    a_norm = build_normalized_adjacency(route_counts, idents)
    print(f"Normalize komsuluk matrisi: {tuple(a_norm.shape)}")

    torch.manual_seed(0)
    model = AirportGCN(in_dim=x.shape[1], hidden_dim=32, out_dim=16, n_gcn_layers=2)
    model.eval()
    with torch.no_grad():
        h_final = model(x, a_norm)
    print(f"Final embedding: {tuple(h_final.shape)}  ({len(idents)} havalimani x 16-boyutlu vektor)")
    print("Ornek (ilk 3 havalimani, ilk 5 boyut):")
    for i in range(3):
        print(idents[i], h_final[i, :5].tolist())

    assert torch.isfinite(h_final).all(), "NaN/Inf embedding bulundu!"
    print("\nOK: forward pass NaN/Inf icermiyor, mimari uctan uca calisiyor.")


if __name__ == "__main__":
    main()
