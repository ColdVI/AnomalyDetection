"""Nihai sunum/rapor politikasinin gercek etkisini TEK bir sayida olcer:
nadir-rota deneyi (min_route_count ile yeniden egitim) beklenenin AKSINE
kotulesme gosterdi (bkz. gunluk 2026-08-01, %47.34 -> %42.87) -- bu yuzden
YENIDEN EGITIM YOK, orijinal (route-filtresiz) en iyi checkpoint kullanilir.
Bunun yerine SADECE raporlama/serving asamasinda iki filtre AYNI ANDA
uygulanir:
  (1) rota train+val+test'te (split_df uzerinden) en az min_route_count
      kez uculmus olmali (nadir rotalar RAPORDAN cikarilir, modelden degil),
  (2) ornek en az min_truncate_min (veya "full") kadar iz icermeli (ilk
      birkac dakikada tahmin verilmez).
Ikisi TEK BIR alt-kumede (kesisim) birlikte uygulanip top-1/3/5 raporlanir --
evaluate_by_route_frequency.py ve evaluate_by_truncation.py'nin AYRI AYRI
gosterdigi iki boyutun gercek KESISIMINI gormek icin (ikisini toplayip
tahmin etmek yerine).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader

from individual.metehan_varis_tahmini.airport_gcn import (
    AirportGCN, build_normalized_adjacency, load_feature_tensor, FEATURES_PATH, CHECKPOINT_PATH,
)
from individual.metehan_varis_tahmini.airport_gat import AirportGAT, build_adjacency_mask
from individual.metehan_varis_tahmini.query_tower import QueryTower
from individual.metehan_varis_tahmini.train_full import (
    TrajectoryDataset, collate, load_all_sessions, SPLIT_PATH, checkpoint_paths,
)

TRUNCATE_ORDER = ["5", "10", "15", "30", "full"]


@torch.no_grad()
def main(item_tower_kind: str = "gat", loss_kind: str = "ce", extra: str = "aug_opensky_h64o32",
         checkpoint: str = "best", hidden_dim: int = 64, out_dim: int = 32,
         min_route_count: int = 10, min_truncate_min: str = "15",
         top_k: tuple[int, ...] = (1, 3, 5)):
    feat_df = pd.read_parquet(FEATURES_PATH)
    x, idents = load_feature_tensor(feat_df)
    airport_to_idx = {a: i for i, a in enumerate(idents)}

    raw = json.loads(CHECKPOINT_PATH.read_text())
    route_counts_archive = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})
    adj = build_normalized_adjacency(route_counts_archive, idents) if item_tower_kind == "gcn" \
        else build_adjacency_mask(route_counts_archive, idents)

    split_df = pd.read_parquet(SPLIT_PATH)
    split_map = split_df.set_index("session_id")["split"]

    # test-setindeki rota-tekrar sikligini split_df uzerinden hesapla
    # (evaluate_by_route_frequency.py ile AYNI mantik -- TUM kept session'lar).
    route_counts = split_df.groupby(["origin_airport", "dest_airport"]).size()
    route_count_map = route_counts.to_dict()

    all_df = load_all_sessions()
    all_df["split"] = all_df["session_id"].map(split_map)
    all_df = all_df.dropna(subset=["split"])
    test_df = all_df[all_df["split"] == "test"].copy()
    test_df["route_count"] = test_df.apply(
        lambda r: route_count_map.get((r["origin_airport"], r["dest_airport"]), 1), axis=1)
    test_df["truncate_min"] = test_df["truncate_min"].astype(str)

    # TRUNCATE_ORDER'daki indeksi kullanarak "en az min_truncate_min kadar iz"
    # kosulunu kur -- "full" her zaman en buyuk/en bilgili kesim sayilir.
    min_idx = TRUNCATE_ORDER.index(min_truncate_min)
    allowed_truncates = set(TRUNCATE_ORDER[min_idx:])

    n_before = len(test_df)
    combined_df = test_df[
        (test_df["route_count"] >= min_route_count) & (test_df["truncate_min"].isin(allowed_truncates))
    ].copy()
    print(f"Politika: rota >= {min_route_count} kez (train+val+test) VE kesme >= {min_truncate_min}dk "
          f"({sorted(allowed_truncates)}) -- {n_before} ornekten {len(combined_df)} tanesi kaldi "
          f"({100*len(combined_df)/n_before:.1f}%)", flush=True)

    MODEL_OUT, BEST_MODEL_OUT = checkpoint_paths(item_tower_kind, loss_kind, extra=extra)
    ckpt_path = BEST_MODEL_OUT if (checkpoint == "best" and BEST_MODEL_OUT.exists()) else MODEL_OUT
    ckpt = torch.load(ckpt_path, weights_only=False)
    print(f"Checkpoint: {ckpt_path.name} (epoch={ckpt.get('epoch', 'eski-format')}, "
          f"val_acc={ckpt.get('val_acc', float('nan')):.4f})")

    if item_tower_kind == "gcn":
        item_tower = AirportGCN(in_dim=x.shape[1], hidden_dim=hidden_dim, out_dim=out_dim, n_gcn_layers=2)
    else:
        item_tower = AirportGAT(in_dim=x.shape[1], hidden_dim=hidden_dim, out_dim=out_dim, n_gat_layers=2, n_heads=4)
    query_tower = QueryTower(d_model=hidden_dim, nhead=4, num_layers=2, out_dim=out_dim)
    item_tower.load_state_dict(ckpt["item_tower"])
    query_tower.load_state_dict(ckpt["query_tower"])
    item_tower.eval()
    query_tower.eval()

    airport_emb = item_tower(x, adj)

    ds = TrajectoryDataset(combined_df, airport_to_idx)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
    max_k = max(top_k)
    correct = {k: 0 for k in top_k}
    n = 0
    for points, lengths, labels, origins in loader:
        query_emb = query_tower(points, lengths)
        logits = query_emb @ airport_emb.T
        topk_idx = logits.topk(min(max_k, logits.shape[-1]), dim=-1).indices
        hit = topk_idx == labels.unsqueeze(1)
        for k in top_k:
            correct[k] += hit[:, :k].any(dim=1).sum().item()
        n += len(labels)

    print(f"\n{'Ornek':<10} " + " ".join(f"{'top'+str(k):<10}" for k in top_k))
    row = " ".join(f"{(correct[k]/n if n else float('nan')):<10.4f}" for k in top_k)
    print(f"{n:<10} {row}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-tower", choices=["gcn", "gat"], default="gat")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--extra", default="aug_opensky_h64o32",
                         help="checkpoint_paths()'in ucuncu parametresi -- varsayilan, orijinal "
                              "(route-filtresiz) en iyi model")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--out-dim", type=int, default=32)
    parser.add_argument("--min-route-count", type=int, default=10,
                         help="Bu sayidan az uculmus rotalar RAPORDAN cikarilir (train'den degil)")
    parser.add_argument("--min-truncate-min", choices=TRUNCATE_ORDER, default="15",
                         help="Bu kesimden ONCEKI (daha az bilgili) ornekler RAPORDAN cikarilir")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()
    main(item_tower_kind=args.item_tower, loss_kind=args.loss, extra=args.extra, checkpoint=args.checkpoint,
         hidden_dim=args.hidden_dim, out_dim=args.out_dim, min_route_count=args.min_route_count,
         min_truncate_min=args.min_truncate_min, top_k=tuple(args.top_k))
