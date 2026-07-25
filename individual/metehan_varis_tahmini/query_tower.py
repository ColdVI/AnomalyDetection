"""Query tower prototip: ucagin gozlemlenmis (kismi) izini kodlayip item
tower (airport_gcn.py) ile AYNI boyutlu bir vektore ceviren Transformer
encoder. Henuz EGITIM/loss YOK -- sadece mimarinin uctan uca calistigini
(rastgele/sahte bir batch ile) dogruluyor.

Girdi noktasi feature'lari: [lat, lon, altitude_m, velocity_mps,
sin(heading), cos(heading), vertical_rate_mps, dt_since_start_norm]
-- heading DAIRESEL oldugu icin (0/360 sinirinda kopukluk olmasin diye)
sin/cos ciftine cevrilir. Zaman damgasi sabit pozisyonel-encoding yerine
DOGRUDAN feature olarak veriliyor -- ADS-B pingleri DUZENSIZ araliklarla
geldigi icin (klasik Transformer PE'nin varsaydigi esit-aralikli sira degil).

Havuzlama (pooling): SON (en guncel) zaman adiminin temsili kullanilir --
self-attention sayesinde bu temsil zaten TUM gecmis noktalari icerir, ve
"anlik guncellenen tahmin" ihtiyaciyla dogal olarak ortusur (yeni nokta
geldikce son-adim temsili guncellenir).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

POINT_DIM = 8  # lat, lon, alt, velocity, sin(heading), cos(heading), vrate, dt_norm


class QueryTower(nn.Module):
    def __init__(self, d_model: int = 32, nhead: int = 4, num_layers: int = 2, out_dim: int = 16):
        super().__init__()
        self.input_proj = nn.Linear(POINT_DIM, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, points: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """points: (batch, max_len, POINT_DIM) -- kisa izler 0 ile pad edilmis.
        lengths: (batch,) -- her ornegin GERCEK (pad'siz) nokta sayisi."""
        batch, max_len, _ = points.shape
        pad_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)  # True = pad

        h = self.input_proj(points)
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        # her ornegin SON gercek (pad-olmayan) adimini sec
        last_idx = (lengths - 1).clamp(min=0)
        last_h = h[torch.arange(batch), last_idx]
        return self.output_proj(last_h)


def normalize_heading(heading_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rad = heading_deg * (math.pi / 180.0)
    return torch.sin(rad), torch.cos(rad)


def match_query_to_airports(query_emb: torch.Tensor, airport_emb: torch.Tensor) -> torch.Tensor:
    """query_emb: (batch, dim), airport_emb: (n_airports, dim) -> (batch, n_airports)
    softmax olasilik dagilimi (origin-kosullu adaylarla kisitlama burada YOK,
    ilk prototip -- TUM havalimanlarina karsi skorluyor)."""
    scores = query_emb @ airport_emb.T
    return torch.softmax(scores, dim=-1)


def _synthetic_batch(batch_size: int = 4, max_len: int = 20, seed: int = 0):
    """Gercek session verisi henuz cikarilmadigi icin, mimariyi test etmek
    icin rastgele ama gercekci ARALIKTA sahte bir batch uretir."""
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(3, max_len + 1, (batch_size,), generator=g)
    points = torch.zeros(batch_size, max_len, POINT_DIM)
    for i in range(batch_size):
        n = lengths[i].item()
        lat = torch.linspace(35.0, 45.0, n) + torch.randn(n, generator=g) * 0.1
        lon = torch.linspace(10.0, 30.0, n) + torch.randn(n, generator=g) * 0.1
        alt = torch.linspace(500.0, 9000.0, n)
        vel = torch.full((n,), 220.0) + torch.randn(n, generator=g) * 5
        heading = torch.full((n,), 75.0)
        sin_h, cos_h = normalize_heading(heading)
        vrate = torch.randn(n, generator=g) * 2
        dt_norm = torch.linspace(0, 1, n)
        points[i, :n, 0] = lat
        points[i, :n, 1] = lon
        points[i, :n, 2] = alt
        points[i, :n, 3] = vel
        points[i, :n, 4] = sin_h
        points[i, :n, 5] = cos_h
        points[i, :n, 6] = vrate
        points[i, :n, 7] = dt_norm
    return points, lengths


def main():
    points, lengths = _synthetic_batch()
    print(f"Sahte batch: points={tuple(points.shape)}, lengths={lengths.tolist()}")

    torch.manual_seed(0)
    tower = QueryTower(out_dim=16)
    tower.eval()
    with torch.no_grad():
        query_emb = tower(points, lengths)
    print(f"Query embedding: {tuple(query_emb.shape)}")
    assert torch.isfinite(query_emb).all(), "NaN/Inf query embedding!"

    # item tower ile eslesme testi -- gercek 852 havalimanlik embedding kullan
    from individual.metehan_varis_tahmini.airport_gcn import (
        AirportGCN, build_normalized_adjacency, load_feature_tensor, FEATURES_PATH, CHECKPOINT_PATH,
    )
    import json
    from collections import Counter
    import pandas as pd

    df = pd.read_parquet(FEATURES_PATH)
    x, idents = load_feature_tensor(df)
    raw = json.loads(CHECKPOINT_PATH.read_text())
    route_counts = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})
    a_norm = build_normalized_adjacency(route_counts, idents)

    item_model = AirportGCN(in_dim=x.shape[1], hidden_dim=32, out_dim=16, n_gcn_layers=2)
    item_model.eval()
    with torch.no_grad():
        airport_emb = item_model(x, a_norm)

    probs = match_query_to_airports(query_emb, airport_emb)
    print(f"\nEslesme olasilik matrisi: {tuple(probs.shape)} (batch x havalimani)")
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), atol=1e-4)

    top5 = probs[0].topk(5)
    print("Ilk ornek icin (egitimsiz, rastgele agirliklarla) en olasi 5 havalimani:")
    for score, i in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {idents[i]}: {score:.4f}")

    print("\nOK: query tower + item tower UCTAN UCA baglaniyor, olasilik dagilimi gecerli (toplam=1).")
    print("NOT: agirliklar HENUZ EGITILMEDI -- bu sadece mimari/boyut dogrulamasi.")


if __name__ == "__main__":
    main()
