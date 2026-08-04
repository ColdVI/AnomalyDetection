"""Test-seti dogrulugunu kesme-noktasina (truncate_min: 5/10/15/30/full)
gore AYRI raporlar -- rota-sikligina gore kirilim (evaluate_by_route_frequency.py)
YETERLI DEGIL cunku aggregate rakam, kalkistan hemen sonraki (5dk) orneklerle
inise yakin (tam iz) ornekleri karistiriyor. Kalkistan 5dk sonra dogru varis
GERCEKTEN belirsiz olabilir (mukemmel bir model bile bilemez -- onlarca
havalimani hala olasi); inise yakin ornekte cok daha yuksek dogruluk beklenir.
Bu script, modelin gercek tavaninin nerede oldugunu kesme-noktasi bazinda
gosterir (bkz. gunluk 2026-07-27).

Herhangi bir checkpoint'e karsi calisir -- egitim surerken bile (checkpoint
VARSA) calistirilip kod dogrulugu onceden test edilebilir; nihai/anlamli
sonuc icin egitim TAMAMLANMIS checkpoint'e karsi calistirilmali.
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

# build_training_dataset.py'deki TRUNCATE_MINUTES sirasiyla (None -> "full")
TRUNCATE_ORDER = ["5", "10", "15", "30", "full"]


@torch.no_grad()
def main(item_tower_kind: str = "gcn", checkpoint: str = "best", extra: str = "",
         hidden_dim: int = 32, out_dim: int = 16, top_k: tuple[int, ...] = (1, 3, 5)):
    feat_df = pd.read_parquet(FEATURES_PATH)
    x, idents = load_feature_tensor(feat_df)
    airport_to_idx = {a: i for i, a in enumerate(idents)}

    raw = json.loads(CHECKPOINT_PATH.read_text())
    route_counts = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})

    split_df = pd.read_parquet(SPLIT_PATH)
    split_map = split_df.set_index("session_id")["split"]

    all_df = load_all_sessions()
    all_df["split"] = all_df["session_id"].map(split_map)
    all_df = all_df.dropna(subset=["split"])
    test_df = all_df[all_df["split"] == "test"].copy()
    test_df["truncate_min"] = test_df["truncate_min"].astype(str)

    MODEL_OUT, BEST_MODEL_OUT = checkpoint_paths(item_tower_kind, "ce", extra=extra)
    ckpt_path = BEST_MODEL_OUT if (checkpoint == "best" and BEST_MODEL_OUT.exists()) else MODEL_OUT
    ckpt = torch.load(ckpt_path, weights_only=False)
    print(f"Checkpoint: {ckpt_path.name} (epoch={ckpt.get('epoch', 'eski-format')}, "
          f"val_acc={ckpt.get('val_acc', float('nan')):.4f})")

    if item_tower_kind == "gcn":
        adj = build_normalized_adjacency(route_counts, idents)
        item_tower = AirportGCN(in_dim=x.shape[1], hidden_dim=hidden_dim, out_dim=out_dim, n_gcn_layers=2)
    else:
        adj = build_adjacency_mask(route_counts, idents)
        item_tower = AirportGAT(in_dim=x.shape[1], hidden_dim=hidden_dim, out_dim=out_dim, n_gat_layers=2, n_heads=4)
    query_tower = QueryTower(d_model=hidden_dim, nhead=4, num_layers=2, out_dim=out_dim)
    item_tower.load_state_dict(ckpt["item_tower"])
    query_tower.load_state_dict(ckpt["query_tower"])
    item_tower.eval()
    query_tower.eval()

    airport_emb = item_tower(x, adj)

    # top_k: bkz. evaluate_by_route_frequency.py -- kismi/erken izlerde tek
    # bir kesin cevap yerine "makul adaylar" icinde yakalama oranini gosterir.
    max_k = max(top_k)
    header = f"{'Kesme':<10} {'Ornek':<10} " + " ".join(f"{'top'+str(k):<10}" for k in top_k)
    print(f"\n{header}")
    overall_correct = {k: 0 for k in top_k}
    overall_n = 0
    for label in TRUNCATE_ORDER:
        sub = test_df[test_df["truncate_min"] == label]
        if len(sub) == 0:
            print(f"{label:<10} {'0':<10} -")
            continue
        ds = TrajectoryDataset(sub, airport_to_idx)
        if len(ds) == 0:
            print(f"{label:<10} {'0':<10} -")
            continue
        loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
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
        row = " ".join(f"{(correct[k]/n if n else float('nan')):<10.4f}" for k in top_k)
        print(f"{label:<10} {n:<10} {row}")
        for k in top_k:
            overall_correct[k] += correct[k]
        overall_n += n

    row = " ".join(f"{(overall_correct[k]/overall_n):<10.4f}" for k in top_k)
    print(f"\n{'GENEL':<10} {overall_n:<10} {row}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-tower", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best",
                         help="'best' = en yuksek val_acc'li checkpoint, 'last' = son epoch")
    parser.add_argument("--extra", default="", help="checkpoint_paths()'in ucuncu parametresi (ör. 'aug_opensky_h64o32')")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Egitimde kullanilan hidden_dim ile AYNI olmali")
    parser.add_argument("--out-dim", type=int, default=16, help="Egitimde kullanilan out_dim ile AYNI olmali")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5], help="Raporlanacak top-k degerleri")
    args = parser.parse_args()
    main(item_tower_kind=args.item_tower, checkpoint=args.checkpoint, extra=args.extra,
         hidden_dim=args.hidden_dim, out_dim=args.out_dim, top_k=tuple(args.top_k))
