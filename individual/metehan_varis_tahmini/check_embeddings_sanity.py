"""Epoch-0 checkpoint'teki (henuz 1 epoch egitilmis) havalimani embedding'lerine
hizli bir akil-sagligi testi: her havalimani icin en yakin (cosine benzerlik)
komsularina bakip, bunlarin GERCEK rota-komsulari (route_counts) ile ne kadar
ortustugunu olcer. Egitim surerken paralel calistirilabilir (checkpoint'i
sadece OKUYOR, egitim surecine dokunmuyor).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from individual.metehan_varis_tahmini.airport_gcn import (
    AirportGCN, build_normalized_adjacency, load_feature_tensor, FEATURES_PATH, CHECKPOINT_PATH,
)

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model_checkpoint.pt"


def main():
    ckpt = torch.load(MODEL_PATH, weights_only=False)
    epoch = ckpt.get("epoch", "eski-format/bilinmiyor")
    val_loss = ckpt.get("val_loss")
    val_acc = ckpt.get("val_acc")
    val_loss_s = f"{val_loss:.4f}" if val_loss is not None else "yok"
    val_acc_s = f"{val_acc:.4f}" if val_acc is not None else "yok"
    print(f"Checkpoint: epoch={epoch}, val_loss={val_loss_s}, val_acc={val_acc_s}")

    feat_df = pd.read_parquet(FEATURES_PATH)
    x, idents = load_feature_tensor(feat_df)
    airport_to_idx = {a: i for i, a in enumerate(idents)}

    raw = json.loads(CHECKPOINT_PATH.read_text())
    route_counts = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})
    a_norm = build_normalized_adjacency(route_counts, idents)

    model = AirportGCN(in_dim=x.shape[1], hidden_dim=32, out_dim=16, n_gcn_layers=2)
    model.load_state_dict(ckpt["item_tower"])
    model.eval()
    with torch.no_grad():
        emb = model(x, a_norm)
    emb_norm = emb / emb.norm(dim=-1, keepdim=True)
    sims = emb_norm @ emb_norm.T

    # her havalimaninin GERCEK rota-komsulari (route_counts'tan)
    real_neighbors = {a: set() for a in idents}
    for (o, d), w in route_counts.items():
        if o in airport_to_idx and d in airport_to_idx:
            real_neighbors[o].add(d)
            real_neighbors[d].add(o)

    # test edilecek birkaca taninan hub
    test_airports = ["LTFM", "EGLL", "LFPG", "EDDF", "RJTT"]
    print("\nHavalimani | Embedding'e gore en yakin 5 komsu | Bunlardan kacı GERCEK rota-komsusu")
    for a in test_airports:
        if a not in airport_to_idx:
            print(f"{a}: feature dosyasinda yok, atlaniyor")
            continue
        i = airport_to_idx[a]
        top = sims[i].topk(6)  # ilk sira kendisi olacak, 6 alip cikaracagiz
        neighbors = [idents[j] for j in top.indices.tolist() if idents[j] != a][:5]
        real = real_neighbors[a]
        overlap = sum(1 for n in neighbors if n in real)
        print(f"{a} (gercek rota-komsu sayisi={len(real)}): {neighbors} -> ortusme {overlap}/5")


if __name__ == "__main__":
    main()
