"""Item tower alternatifi: GCN yerine GAT (Graph Attention Network).

GCN'den farki: komsuluk agirliklari (A_norm) ONCEDEN SABIT (rota-sayisina
gore hesaplanmis) -- GAT'ta ise her komsunun agirligi (attention katsayisi)
MODELIN KENDISI tarafindan, veriden OGRENILIR. Fikir: "bazi komsular tahmin
icin digerlerinden daha onemli olabilir" -- GCN bunu varsayamaz, GAT
varsayabilir.

852 dugumluk bu olcekte YOGUN (dense) matris implementasyonu yeterince
hizli -- torch_geometric gerekmiyor (Windows'ta C-extension kurulum riski,
bkz. airport_gcn.py'nin ayni gerekcesi).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from individual.metehan_varis_tahmini.airport_gcn import AirportProjection


def build_adjacency_mask(route_counts: dict[tuple, int], idents: list[str]) -> torch.Tensor:
    """GAT icin: A_norm gibi AGIRLIKLI degil, sadece "bu kenar var mi" (Bool)
    -- agirligi model kendi ogrenecegi icin burada sadece hangi cift
    komsu sayilir bilgisi yeterli. Kendine-donguler (self-loop) dahil
    (bir havalimani kendi eski temsiline de "dikkat" edebilsin diye)."""
    n = len(idents)
    idx = {a: i for i, a in enumerate(idents)}
    mask = torch.eye(n, dtype=torch.bool)  # self-loop
    for (o, d), w in route_counts.items():
        if o in idx and d in idx:
            i, j = idx[o], idx[d]
            mask[i, j] = True
            mask[j, i] = True  # GCN ile tutarli: yon simdilik goz ardi ediliyor
    return mask


class GATLayer(nn.Module):
    """Standart GAT (Velickovic ve ark. 2018) katmani, coklu-head, YOGUN
    (dense NxN) implementasyon. Sadece GERCEK komsular arasinda dikkat
    hesaplanir (adj_mask ile maskelenerek) -- 852 dugum olceginde N^2
    hesabı ucuz (~726K cift), Transformer'in trajectory-batch hesabından
    cok daha hafif."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, leaky_slope: float = 0.2, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim, n_heads'e tam bolunmeli"
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # GAT makalesindeki "a" vektoru: [Wh_i || Wh_j] ile carpilan tek bir
        # vektor yerine, hesaplama kolayligi icin src/dst parcalarina ayrildi
        # (matematiksel olarak es-degerdir: a^T[Wh_i||Wh_j] = a_src.Wh_i + a_dst.Wh_j).
        self.a_src = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky = nn.LeakyReLU(leaky_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        Wh = self.W(h).view(n, self.n_heads, self.head_dim)  # (N, heads, head_dim)

        # e_ij = LeakyReLU(a_src . Wh_i + a_dst . Wh_j) -- her head icin ayri
        src_scores = (Wh * self.a_src).sum(-1)  # (N, heads) -- dugum i'nin "kaynak" skoru
        dst_scores = (Wh * self.a_dst).sum(-1)  # (N, heads) -- dugum j'nin "hedef" skoru
        e = src_scores.unsqueeze(1) + dst_scores.unsqueeze(0)  # (N, N, heads), e[i,j,h]
        e = self.leaky(e)

        # SADECE gercek komsularda (adj_mask) attention hesaplanir -- geri
        # kalani -inf'e cekilip softmax'ta sifir agirlik alir.
        e = e.masked_fill(~adj_mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=1)  # her i icin, TUM j komsulari uzerinde normalize
        alpha = self.dropout(alpha)

        # h_i' = sum_j alpha_ij * Wh_j (head basina), sonra head'ler concat edilir
        out = torch.einsum("ijh,jhd->ihd", alpha, Wh)  # (N, heads, head_dim)
        return out.reshape(n, self.out_dim)


class AirportGAT(nn.Module):
    """AirportGCN ile AYNI dis arayuz (forward(x, adj) -> (N, out_dim)) --
    train_full.py'de tek bir --item-tower bayragiyla ikisi arasinda
    gecis yapilabilsin diye."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_gat_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.proj = AirportProjection(in_dim, hidden_dim, hidden_dim)
        dims = [hidden_dim] + [out_dim] * n_gat_layers
        self.gat_layers = nn.ModuleList([
            GATLayer(dims[i], dims[i + 1], n_heads=n_heads) for i in range(n_gat_layers)
        ])

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for i, layer in enumerate(self.gat_layers):
            h = layer(h, adj_mask)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)  # orijinal GAT makalesi ELU kullanir (ReLU degil)
        return h


def main():
    """Kucuk, hizli bir dogrulama: gercek 852 havalimani + gercek rota
    grafigiyle forward pass calisiyor mu, NaN/Inf var mi, gradyan akiyor mu."""
    import json
    from collections import Counter
    import pandas as pd
    from individual.metehan_varis_tahmini.airport_gcn import load_feature_tensor, FEATURES_PATH, CHECKPOINT_PATH

    feat_df = pd.read_parquet(FEATURES_PATH)
    x, idents = load_feature_tensor(feat_df)

    raw = json.loads(CHECKPOINT_PATH.read_text())
    route_counts = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})
    adj_mask = build_adjacency_mask(route_counts, idents)
    print(f"adj_mask: {tuple(adj_mask.shape)}, toplam kenar (self-loop dahil): {adj_mask.sum().item()}")

    torch.manual_seed(0)
    model = AirportGAT(in_dim=x.shape[1], hidden_dim=32, out_dim=16, n_gat_layers=2, n_heads=4)
    out = model(x, adj_mask)
    print(f"Cikti: {tuple(out.shape)}, finite: {torch.isfinite(out).all().item()}")

    loss = out.sum()
    loss.backward()
    grad_ok = model.proj.net[0].weight.grad is not None and torch.isfinite(model.proj.net[0].weight.grad).all()
    print(f"Backward basarili, gradyan finite: {grad_ok}")


if __name__ == "__main__":
    main()
