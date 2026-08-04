"""Egitim bitince kullanilacak nihai degerlendirme: genel test-seti dogrulugu
YETERLI DEGIL cunku rota dagilimi ciddi dengesiz (en ust 227 rota hacmin
%21,6'sini olusturuyor -- bkz. gunluk 2026-07-23). Bu yuzden test-seti
dogrulugu rota-tekrar-sikligi dilimlerine gore AYRI raporlanir.

Herhangi bir checkpoint'e karsi calisir (train_full_v6.py'nin MODEL_OUT'u) --
egitim surerken bile (checkpoint dosyasi VARSA) calistirilip kod dogrulugu
onceden test edilebilir; nihai/anlamli sonuc icin egitim TAMAMLANMIS
checkpoint'e karsi calistirilmali.

v5: v4'un import'unu train_full_v6'ya guncelledi (Colab'daki guncel script).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from individual.metehan_varis_tahmini.airport_gcn import (
    AirportGCN, build_normalized_adjacency, load_feature_tensor, FEATURES_PATH, CHECKPOINT_PATH,
)
from individual.metehan_varis_tahmini.airport_gat import AirportGAT, build_adjacency_mask
from individual.metehan_varis_tahmini.query_tower import QueryTower
from individual.metehan_varis_tahmini.train_full_v6 import (
    TrajectoryDataset, collate, load_all_sessions, SPLIT_PATH, checkpoint_paths,
)
import json
from collections import Counter

BUCKETS = [(2, 4), (5, 9), (10, 19), (20, 49), (50, None)]


def bucket_label(n):
    for lo, hi in BUCKETS:
        if hi is None and n >= lo:
            return f"{lo}+"
        if hi is not None and lo <= n <= hi:
            return f"{lo}-{hi}"
    return "?"


@torch.no_grad()
def main(checkpoint: str = "best", item_tower_kind: str = "gcn", loss_kind: str = "ce", extra: str = ""):
    MODEL_OUT, BEST_MODEL_OUT = checkpoint_paths(item_tower_kind, loss_kind, extra=extra)

    feat_df = pd.read_parquet(FEATURES_PATH)
    x, idents = load_feature_tensor(feat_df)
    airport_to_idx = {a: i for i, a in enumerate(idents)}

    raw = json.loads(CHECKPOINT_PATH.read_text())
    route_counts_archive = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})
    a_norm = build_normalized_adjacency(route_counts_archive, idents) if item_tower_kind == "gcn" \
        else build_adjacency_mask(route_counts_archive, idents)

    split_df = pd.read_parquet(SPLIT_PATH)
    split_map = split_df.set_index("session_id")["split"]

    all_df = load_all_sessions()
    all_df["split"] = all_df["session_id"].map(split_map)
    all_df = all_df.dropna(subset=["split"])

    # test-setindeki rota-tekrar sikligini split_df uzerinden hesapla
    # (build_split.py'nin ayni mantigi -- TUM kept session'lar uzerinden rota sayimi)
    route_counts = split_df.groupby(["origin_airport", "dest_airport"]).size()
    route_count_map = route_counts.to_dict()

    test_df = all_df[all_df["split"] == "test"].copy()
    test_df["route_count"] = test_df.apply(
        lambda r: route_count_map.get((r["origin_airport"], r["dest_airport"]), 1), axis=1)
    test_df["bucket"] = test_df["route_count"].apply(bucket_label)

    # Varsayilan: EN IYI (val_acc'e gore) checkpoint -- val_acc epoch'tan
    # epoch'a dalgalanabiliyor (2026-07-24: epoch 5, epoch 4'ten kotu cikti),
    # bu yuzden "son epoch" ile "en iyi epoch" AYNI SEY DEGIL.
    ckpt_path = BEST_MODEL_OUT if (checkpoint == "best" and BEST_MODEL_OUT.exists()) else MODEL_OUT
    ckpt = torch.load(ckpt_path, weights_only=False)
    print(f"Checkpoint: {ckpt_path.name} (epoch={ckpt.get('epoch', 'eski-format')}, "
          f"val_acc={ckpt.get('val_acc', float('nan')):.4f})")

    if item_tower_kind == "gcn":
        item_tower = AirportGCN(in_dim=x.shape[1], hidden_dim=32, out_dim=16, n_gcn_layers=2)
    else:
        item_tower = AirportGAT(in_dim=x.shape[1], hidden_dim=32, out_dim=16, n_gat_layers=2, n_heads=4)
    query_tower = QueryTower(d_model=32, nhead=4, num_layers=2, out_dim=16)
    item_tower.load_state_dict(ckpt["item_tower"])
    query_tower.load_state_dict(ckpt["query_tower"])
    item_tower.eval()
    query_tower.eval()

    airport_emb = item_tower(x, a_norm)

    print(f"\n{'Dilim':<10} {'Ornek':<10} {'Dogruluk':<10}")
    overall_correct, overall_n = 0, 0
    for label in [bucket_label(lo) for lo, _ in BUCKETS]:
        sub = test_df[test_df["bucket"] == label]
        if len(sub) == 0:
            print(f"{label:<10} {'0':<10} -")
            continue
        ds = TrajectoryDataset(sub, airport_to_idx)
        if len(ds) == 0:
            print(f"{label:<10} {'0':<10} -")
            continue
        loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
        correct, n = 0, 0
        for points, lengths, labels, origins in loader:  # collate artik origin da donduruyor (mask_routes icin)
            query_emb = query_tower(points, lengths)
            logits = query_emb @ airport_emb.T
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            n += len(labels)
        acc = correct / n if n else float("nan")
        print(f"{label:<10} {n:<10} {acc:.4f}")
        overall_correct += correct
        overall_n += n

    print(f"\n{'GENEL':<10} {overall_n:<10} {overall_correct/overall_n:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best",
                         help="'best' = en yuksek val_acc'li checkpoint, 'last' = son epoch")
    parser.add_argument("--item-tower", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--extra", default="", help="checkpoint_paths()'in ucuncu parametresi (ör. 'aug', 'opensky', 'aug_opensky')")
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, item_tower_kind=args.item_tower, loss_kind=args.loss, extra=args.extra)
