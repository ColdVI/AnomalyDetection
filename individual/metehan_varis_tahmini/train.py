"""Egitim dongusu prototipi: query tower (Transformer) + item tower (GCN)
birlikte egitiliyor, origin-kosullu DEGIL tam-softmax cross-entropy ile
(852 havalimanindan dogru varisi tahmin et). Bu ilk versiyon -- origin'e gore
aday kisitlama sonraki bir iyilestirme.

Veri: training_sessions/*.parquet (build_training_dataset.py ciktisi).
Henuz TAM veri seti bitmedi (arka planda calisiyor) -- bu script su ana kadar
YAZILMIS parca dosyalarindan bir ornekle mekanigi (ileri+geri yayilim, loss
azalisi) dogrular. Tam veri bitince ayni script full dataset + gercek
train/val/test ayrimiyla yeniden calistirilacak.
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"c:\Users\PC_6276\Desktop\github\AnomalyDetection")
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from individual.metehan_varis_tahmini.airport_gcn import (
    AirportGCN, build_normalized_adjacency, load_feature_tensor, FEATURES_PATH, CHECKPOINT_PATH,
)
from individual.metehan_varis_tahmini.query_tower import QueryTower, POINT_DIM

SESSIONS_DIR = Path(__file__).parent / "training_sessions"


MAX_SEQ_LEN = 300  # Transformer self-attention O(n^2) -- bazi "tam iz" ornekleri
                    # binlerce nokta iceriyor (uzun ucuslar, sik ornekleme),
                    # sinirlamazsak bellek patluyor. Esit araliklarla downsample edilir.


def _downsample(arr, n, max_len):
    if n <= max_len:
        return arr
    idx = np.linspace(0, n - 1, max_len).round().astype(int)
    return [arr[i] for i in idx]


class TrajectoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, airport_to_idx: dict[str, int]):
        df = df[df["dest_airport"].isin(airport_to_idx)].reset_index(drop=True)
        self.df = df
        self.airport_to_idx = airport_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        n = min(row["n_points"], MAX_SEQ_LEN)
        lats = _downsample(row["lats"], row["n_points"], MAX_SEQ_LEN)
        lons = _downsample(row["lons"], row["n_points"], MAX_SEQ_LEN)
        alts = _downsample(row["alts"], row["n_points"], MAX_SEQ_LEN)
        vels = _downsample(row["velocities"], row["n_points"], MAX_SEQ_LEN)
        heads = _downsample(row["headings"], row["n_points"], MAX_SEQ_LEN)
        vrates = _downsample(row["vrates"], row["n_points"], MAX_SEQ_LEN)
        dt_norms = _downsample(row["dt_norms"], row["n_points"], MAX_SEQ_LEN)

        points = np.zeros((n, POINT_DIM), dtype=np.float32)
        points[:, 0] = lats
        points[:, 1] = lons
        points[:, 2] = np.array(alts, dtype=np.float32) / 1000.0  # km'ye olcekle
        points[:, 3] = np.array(vels, dtype=np.float32) / 100.0  # kaba olcekleme
        heading_rad = np.radians(heads)
        points[:, 4] = np.sin(heading_rad)
        points[:, 5] = np.cos(heading_rad)
        points[:, 6] = np.array(vrates, dtype=np.float32) / 10.0
        points[:, 7] = dt_norms
        label = self.airport_to_idx[row["dest_airport"]]
        return torch.from_numpy(points), label


def collate(batch):
    points_list, labels = zip(*batch)
    lengths = torch.tensor([p.shape[0] for p in points_list], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.zeros(len(points_list), max_len, POINT_DIM, dtype=torch.float32)
    for i, p in enumerate(points_list):
        padded[i, :p.shape[0]] = p
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def load_sample_sessions(n_files: int = 30) -> pd.DataFrame:
    files = sorted(glob.glob(str(SESSIONS_DIR / "*.parquet")))[:n_files]
    print(f"{len(files)} parquet dosyasi okunuyor (mevcut arsivden ornek)...")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"-> {len(df)} satir (tum kesme noktalari dahil)")
    return df


def main():
    # --- item tower girdisi: TUM havalimanlarinin feature'lari + graf ---
    feat_df = pd.read_parquet(FEATURES_PATH)
    x, idents = load_feature_tensor(feat_df)
    airport_to_idx = {a: i for i, a in enumerate(idents)}

    raw = json.loads(CHECKPOINT_PATH.read_text())
    route_counts = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})
    a_norm = build_normalized_adjacency(route_counts, idents)

    # --- veri: su ana kadar cikarilmis session'lardan bir ornek ---
    df = load_sample_sessions(n_files=30)
    dataset = TrajectoryDataset(df, airport_to_idx)
    print(f"Kullanilabilir (havalimani sozlugunde eslesen) ornek: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)

    torch.manual_seed(0)
    item_tower = AirportGCN(in_dim=x.shape[1], hidden_dim=32, out_dim=16, n_gcn_layers=2)
    query_tower = QueryTower(d_model=32, nhead=4, num_layers=2, out_dim=16)

    optimizer = torch.optim.Adam(
        list(item_tower.parameters()) + list(query_tower.parameters()), lr=1e-3
    )
    loss_fn = nn.CrossEntropyLoss()

    item_tower.train()
    query_tower.train()

    print("\n--- Egitim adimlari (mekanik/sanity dogrulamasi) ---")
    step = 0
    for epoch in range(5):
        for points, lengths, labels in loader:
            optimizer.zero_grad()
            airport_emb = item_tower(x, a_norm)          # (852, 16) -- her adimda yeniden
            query_emb = query_tower(points, lengths)     # (batch, 16)
            logits = query_emb @ airport_emb.T           # (batch, 852)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                acc = (logits.argmax(dim=-1) == labels).float().mean().item()
                print(f"epoch={epoch} step={step} loss={loss.item():.4f} batch_acc={acc:.2f}")
            step += 1

    print("\nOK: egitim dongusu calisti, geri-yayilim + optimizer adimlari basarili.")
    print("NOT: bu KUCUK bir ornek uzerinde mekanik dogrulama -- tam veri + gercek")
    print("train/val/test ayrimi bitince asil egitim calistirilacak.")


if __name__ == "__main__":
    main()
