"""Feature ablation'in ucuncu ayagi (mentor sorusu, 2026-07-27): mevcut 20
sayisal ozelligin (8 traffic/graf-turevli + 12 statik) uzerine ~9 YENI
graf-topoloji ozelligi ekleyip ~29-30 ozellikli genisletilmis bir feature
seti uretir -- gercek katki olcumu icin.

Kaynak: route_scan_full_archive.json'daki route_counts (mevcut 8 ozelligin
de kaynagi, AYNI graf) -- yonlu (directed), agirlik=rota-sayisi.

Yeni 9 ozellik:
  - betweenness_centrality, closeness_centrality: yapisal pozisyon
    (agirliksiz/topolojik -- trafik hacmini out_degree/total_volume zaten
    tasiyor, bunlar SADECE graf yapisini olcsun diye kasitli agirliksiz).
  - reciprocity: giden komsularin kacinin GERI DONUS rotasi da var (o->d
    VE d->o ikisi de mevcut) -- oran.
  - origin_entropy: gelen rotalarin kaynak-cesitliligi (dest_entropy'nin
    "gelen" yonundeki aynisi).
  - max_out_share: en yogun tek destinasyonun toplam giden hacimdeki payi
    (hub-bagimliligi -- 1.0'a yakinsa "tek rotaya bagimli" havalimani).
  - avg_neighbor_out_degree: komsularinin (gittigi havalimanlarinin)
    ortalama out_degree'si -- "kucuk feeder havalimanlarina mi, buyuk
    hub'lara mi bagli" (assortativity benzeri).
  - n_countries_connected: dogrudan bagli oldugu benzersiz ulke sayisi.
  - hub_score, authority_score: HITS algoritmasi (pagerank'in yonlu
    tamamlayicisi -- "iyi bir hub'a mi isaret ediyor" vs "iyi hub'lardan mi
    isaret aliyor").
"""
from __future__ import annotations

import json
from collections import Counter
from math import log
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

FEATURES_PATH = Path(__file__).parent / "airport_features.parquet"
CHECKPOINT_PATH = Path(__file__).parent / "route_scan_full_archive.json"
OUT_PATH = Path(__file__).parent / "airport_features_expanded30.parquet"


def _entropy(counts: list[float]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    ps = [c / total for c in counts if c > 0]
    return -sum(p * log(p) for p in ps)


def main():
    base = pd.read_parquet(FEATURES_PATH)
    idents = base["ident"].tolist()
    ident_set = set(idents)
    ident_to_country = dict(zip(base["ident"], base["iso_country"]))

    raw = json.loads(CHECKPOINT_PATH.read_text())
    route_counts = Counter({tuple(k.split("||")): v for k, v in raw["route_counts"].items()})

    G = nx.DiGraph()
    G.add_nodes_from(idents)
    for (o, d), w in route_counts.items():
        if o in ident_set and d in ident_set and o != d:
            G.add_edge(o, d, weight=w)

    print(f"Graf: {G.number_of_nodes()} node, {G.number_of_edges()} kenar", flush=True)

    print("betweenness_centrality hesaplaniyor...", flush=True)
    betweenness = nx.betweenness_centrality(G, weight=None)  # kasitli agirliksiz -- yapisal pozisyon
    print("closeness_centrality hesaplaniyor...", flush=True)
    closeness = nx.closeness_centrality(G)
    print("HITS (hub/authority) hesaplaniyor...", flush=True)
    try:
        hubs, authorities = nx.hits(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        hubs = {n: 0.0 for n in idents}
        authorities = {n: 0.0 for n in idents}

    reciprocity = {}
    origin_entropy = {}
    max_out_share = {}
    avg_neighbor_out_degree = {}
    n_countries_connected = {}

    out_degree = dict(G.out_degree())
    for node in idents:
        out_neighbors = list(G.successors(node))
        in_neighbors = list(G.predecessors(node))

        # reciprocity: giden komsularin kacinda geri-donus kenari da var
        if out_neighbors:
            recip = sum(1 for d in out_neighbors if G.has_edge(d, node)) / len(out_neighbors)
        else:
            recip = 0.0
        reciprocity[node] = recip

        # origin_entropy: gelen rotalarin kaynak-cesitliligi
        in_weights = [G[o][node]["weight"] for o in in_neighbors]
        origin_entropy[node] = _entropy(in_weights)

        # max_out_share: en yogun tek destinasyonun payi
        out_weights = [G[node][d]["weight"] for d in out_neighbors]
        max_out_share[node] = (max(out_weights) / sum(out_weights)) if out_weights else 0.0

        # avg_neighbor_out_degree: gittigi havalimanlarinin ortalama out_degree'si
        if out_neighbors:
            avg_neighbor_out_degree[node] = float(np.mean([out_degree.get(d, 0) for d in out_neighbors]))
        else:
            avg_neighbor_out_degree[node] = 0.0

        # n_countries_connected: dogrudan bagli benzersiz ulke sayisi
        neighbor_countries = {ident_to_country.get(n) for n in set(out_neighbors) | set(in_neighbors)}
        neighbor_countries.discard(None)
        n_countries_connected[node] = len(neighbor_countries)

    new_cols = pd.DataFrame({
        "ident": idents,
        "betweenness_centrality": [betweenness[i] for i in idents],
        "closeness_centrality": [closeness[i] for i in idents],
        "hub_score": [hubs[i] for i in idents],
        "authority_score": [authorities[i] for i in idents],
        "reciprocity": [reciprocity[i] for i in idents],
        "origin_entropy": [origin_entropy[i] for i in idents],
        "max_out_share": [max_out_share[i] for i in idents],
        "avg_neighbor_out_degree": [avg_neighbor_out_degree[i] for i in idents],
        "n_countries_connected": [n_countries_connected[i] for i in idents],
    })

    expanded = base.merge(new_cols, on="ident", how="left")
    n_numeric = len(expanded.columns) - 2  # ident, iso_country haric
    print(f"Genisletilmis feature seti: {len(expanded.columns)} kolon ({n_numeric} sayisal ozellik)", flush=True)
    expanded.to_parquet(OUT_PATH, index=False)
    print(f"Yazildi: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
