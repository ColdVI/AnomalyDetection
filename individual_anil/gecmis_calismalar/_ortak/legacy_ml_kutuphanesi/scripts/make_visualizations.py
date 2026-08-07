"""ML-11 gorselleme ve veri-kesif fazi: read-only analiz ciktilari.

Hicbir model/esik/split/scaler/CUSUM artifact'i DEGISMEZ; yalnizca mevcut
artifact'lar yuklenip figur/CSV uretilir (docs/ML11_GORSELLESTIRME_PLAN.md).
Blind holdout (131 SEAD ucusu) hicbir figur/istatistikte kullanilmaz; bu
dosyadaki her veri okumasi development filtresiyle yapilir ve assert edilir.

Kullanim:
    python scripts/make_visualizations.py --dataset uav_sead --sections 1,2,3,4
Cikti: artifacts/viz/<dataset>/s{1..4}_*/*.png|csv + viz_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.artifacts import load_lstm_bundle, load_modular_iforest_bundle
from src.ml.data.scaling import apply_scaler_params
from src.ml.data.splits import NORMAL_LABELS, session_of
from src.ml.data.windowing import build_windows
from src.ml.decision.decision_layers import policy_from_dict
from src.ml.evaluation.events import (
    load_uav_sead_ranges_by_category,
    range_mask,
    uav_sead_absolute_us,
)
from src.ml.evaluation.score_fusion import (
    empirical_probability,
    last_causal_per_bucket,
    max_score_fusion,
)
from src.ml.models.modular_iforest import PX4_ML7_CANDIDATE_MODULES, anomaly_scores
from src.ml.models.lstm_autoencoder import reconstruction_scores

SOURCES = ("alfa", "uav_attack", "uav_sead")
ID_COLUMNS = ("source_id", "label", "t_rel_s")
SEED = 42
PER_FLIGHT_CAP = 200
# t-SNE maliyeti nedeniyle SEAD 30k'ya indirildi (plan siniri <=50k'nin altinda).
MAX_POINTS = {"alfa": 50_000, "uav_attack": 50_000, "uav_sead": 30_000}
CORR_MAX_POINTS = 30_000
AUC_MAX_NEGATIVES = 200_000
REDUNDANT_RHO = 0.9
# Yalniz projeksiyon figurlerinde: RobustScaler ciktisi IQR birimindedir; tek
# bir uc deger PCA/t-SNE'yi tek noktaya sikistirir. +-10 IQR kirpma gorsel
# amaclidir, hicbir model/skor hesabina girmez.
EMBED_CLIP = 10.0

SPLIT_PATH = ROOT / "data/gold/ml_features/split_manifest.json"
FEATURE_PATHS = {s: ROOT / f"data/gold/ml_features/{s}/{s}_ml_features.parquet" for s in SOURCES}
SEAD_SILVER = ROOT / "data/silver/uav_sead_silver.parquet"
SEAD_LABELS = ROOT / "data/objectstore/bronze/uav_sead/labels.json"
ML9_RUN = ROOT / "artifacts/ml9/uav_sead/full_matrix"
IF_BUNDLES = {s: ROOT / f"artifacts/models/{s}/ml6_modular_iforest" for s in SOURCES}
ALFA_LSTM_BUNDLE = ROOT / "artifacts/models/alfa/ml6_lstm_ae"
OUT_ROOT = ROOT / "artifacts/viz"

TITLES = {
    "alfa": "ALFA (sabit kanat, mekanik ariza)",
    "uav_attack": "UAV Attack (PX4, siber saldiri)",
    "uav_sead": "UAV-SEAD (PX4)",
}
TSNE_NOTE = ("t-SNE yalniz lokal komsulugu korur; kumeler arasi mesafe ve "
             "kume boyutu yorumlanmaz.")

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8})


# --------------------------------------------------------------------------
# veri yukleme ve ortak yardimcilar
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def load_development(source: str, *, columns: list[str] | None = None):
    """Development ucuslarini yukle; holdout'un okunmadigini assert et."""
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    config = manifest["sources"][source]
    flight_labels: dict[str, str] = config["flight_labels"]
    holdout = set(config["splits"]["split_00"].get("final_holdout", []))
    development = sorted(set(flight_labels) - holdout)
    frame = pd.read_parquet(
        FEATURE_PATHS[source], columns=columns,
        filters=[("source_id", "in", development)],
    )
    seen = set(frame["source_id"].unique())
    if seen & holdout:
        raise AssertionError(f"{source}: blind holdout satirlari okundu")
    return frame, development, sorted(holdout), flight_labels, config


def feature_columns_of(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ID_COLUMNS]


def deterministic_subsample_index(frame: pd.DataFrame, *, per_flight_cap: int,
                                  max_points: int, seed: int = SEED) -> np.ndarray:
    """Ucus basina cap + dataset tavani; sabit seed'le deterministik."""
    rng = np.random.default_rng(seed)
    picks = []
    for _, group in frame.groupby("source_id", sort=True):
        idx = group.index.to_numpy()
        if len(idx) > per_flight_cap:
            idx = np.sort(rng.choice(idx, size=per_flight_cap, replace=False))
        picks.append(idx)
    chosen = np.concatenate(picks) if picks else np.array([], dtype=int)
    if len(chosen) > max_points:
        chosen = np.sort(rng.choice(chosen, size=max_points, replace=False))
    return chosen


def _save(fig, path: Path, files: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    files[str(path.relative_to(OUT_ROOT))] = None  # checksum sonda hesaplanir


def _write_csv(df: pd.DataFrame, path: Path, files: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    files[str(path.relative_to(OUT_ROOT))] = None


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC = P(pos > neg); bag ortalama-rank ile."""
    from scipy.stats import rankdata

    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def _combined_categories(categories: dict[str, list]) -> dict[str, list]:
    result = {name: list(spans) for name, spans in categories.items()}
    actuator = result.get("Actuator Outputs", []) + result.get("Actuator Controls", [])
    if actuator:
        result["Actuator Outputs+Controls"] = actuator
    return result


def sead_category_masks(frame: pd.DataFrame, dev_ids: list[str]):
    """Satir bazinda kategori maskeleri (yalniz development ucuslari).

    Donen: {kategori: bool maske}, t0 sozlugu (silver'dan; holdout okunmaz).
    """
    silver = pd.read_parquet(
        SEAD_SILVER, columns=["source_id", "timestamp"],
        filters=[("source_id", "in", dev_ids)],
    )
    if set(silver["source_id"].unique()) - set(dev_ids):
        raise AssertionError("uav_sead: silver okuma development disina cikti")
    t0 = silver.groupby("source_id")["timestamp"].min().to_dict()
    by_flight = load_uav_sead_ranges_by_category(SEAD_LABELS)

    masks: dict[str, np.ndarray] = {}
    grouped = frame.groupby("source_id", sort=False).indices
    for source_id, row_idx in grouped.items():
        categories = _combined_categories(by_flight.get(source_id, {}))
        if not categories:
            continue
        absolute = uav_sead_absolute_us(
            frame["t_rel_s"].to_numpy()[row_idx], t0[source_id])
        for category, spans in categories.items():
            hit = range_mask(absolute, spans)
            if not hit.any():
                continue
            mask = masks.setdefault(category, np.zeros(len(frame), dtype=bool))
            mask[row_idx[hit]] = True
    return masks, t0


def row_category_labels(source: str, frame: pd.DataFrame,
                        flight_labels: dict[str, str],
                        sead_masks: dict[str, np.ndarray] | None) -> pd.Series:
    """Boyama etiketi: SEAD'de aralik-bazli kategori, digerlerinde satir etiketi."""
    if source != "uav_sead":
        return frame["label"].astype(str)
    labels = np.array(["aralik disi (anomalili ucus)"] * len(frame), dtype=object)
    normal_flight = frame["source_id"].map(
        lambda s: flight_labels[s] in NORMAL_LABELS).to_numpy()
    labels[normal_flight] = "normal"
    # combined aktuator kategorisi tekil kategorilerden sonra yazilsin diye sirali
    for category in sorted(sead_masks or {}):
        labels[sead_masks[category]] = category
    return pd.Series(labels, index=frame.index)


# --------------------------------------------------------------------------
# Bolum 1 -- veri karnesi
# --------------------------------------------------------------------------

def section_portfolio(source, frame, dev_ids, holdout_ids, flight_labels,
                      out_dir: Path, files: dict, sead_masks=None) -> dict:
    stats: dict = {}
    feature_cols = feature_columns_of(frame)
    title = TITLES[source]
    holdout_note = (f" — {len(holdout_ids)} ucusluk kor holdout haric"
                    if holdout_ids else "")

    # 1. sinif dagilimi (ucus duzeyi)
    dev_labels = pd.Series({sid: flight_labels[sid] for sid in dev_ids})
    counts = dev_labels.value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color="tab:blue")
    for bar, n in zip(bars, counts.values):
        ax.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{title} — Sinif Dagilimi (ucus duzeyi, "
                 f"toplam n={len(dev_labels)}){holdout_note}")
    ax.set_ylabel("ucus sayisi")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, out_dir / "class_counts.png", files)
    stats["flight_label_counts"] = counts.to_dict()

    # 2. SEAD: anotasyon-kategorisi event sayilari + oturum histogrami
    if source == "uav_sead":
        by_flight = load_uav_sead_ranges_by_category(SEAD_LABELS)
        event_counts: dict[str, int] = {}
        for sid in dev_ids:
            for category, spans in _combined_categories(by_flight.get(sid, {})).items():
                event_counts[category] = event_counts.get(category, 0) + len(spans)
        order = sorted(event_counts, key=event_counts.get, reverse=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(order, [event_counts[c] for c in order], color="tab:orange")
        for bar, c in zip(bars, order):
            ax.annotate(f"n={event_counts[c]}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{title} — Anotasyon Kategorisi Basina Aralik Sayisi "
                     f"(gelistirme kumesi){holdout_note}")
        ax.set_ylabel("etiketli aralik (event) sayisi")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        _save(fig, out_dir / "annotation_event_counts.png", files)
        stats["annotation_event_counts"] = event_counts

        normals = [sid for sid in dev_ids if flight_labels[sid] in NORMAL_LABELS]
        session_sizes = pd.Series([session_of(s) for s in normals]).value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(session_sizes)), session_sizes.values, color="tab:green")
        ax.set_title(f"{title} — Normal Ucuslarin Oturum Dagilimi: "
                     f"{len(normals)} ucus / {len(session_sizes)} oturum "
                     f"(gelistirme kumesi){holdout_note}")
        ax.set_xlabel("oturum (buyukten kucuge sirali)")
        ax.set_ylabel("ucus sayisi")
        _save(fig, out_dir / "session_histogram.png", files)
        stats["normal_sessions"] = {"n_normal_flights": len(normals),
                                    "n_sessions": int(len(session_sizes)),
                                    "max_session_size": int(session_sizes.max())}

    # 3. feature x label doluluk haritasi
    label_groups = frame.groupby("label")
    completeness = label_groups[feature_cols].apply(lambda g: g.notna().mean()).T
    label_ns = label_groups.size()
    col_labels = [f"{lab}\n(n={label_ns[lab]:,} satir)" for lab in completeness.columns]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(completeness.columns)),
                                    max(6, 0.16 * len(feature_cols))))
    im = ax.imshow(completeness.to_numpy(), aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(feature_cols)), completeness.index, fontsize=5)
    ax.set_title(f"{title} — Feature x Etiket Doluluk (non-null orani)"
                 f"{holdout_note}")
    fig.colorbar(im, ax=ax, label="non-null orani")
    _save(fig, out_dir / "completeness_heatmap.png", files)
    _write_csv(completeness.reset_index().rename(columns={"index": "feature"}),
               out_dir / "completeness_matrix.csv", files)

    # 4. ucus suresi histogrami (ucus etiketine gore renkli)
    duration_min = frame.groupby("source_id")["t_rel_s"].max() / 60.0
    flight_of = pd.Series({sid: flight_labels[sid] for sid in duration_min.index})
    fig, ax = plt.subplots(figsize=(7, 4))
    labels_sorted = sorted(flight_of.unique())
    ax.hist([duration_min[flight_of == lab].to_numpy() for lab in labels_sorted],
            bins=20, stacked=True,
            label=[f"{lab} (n={int((flight_of == lab).sum())})" for lab in labels_sorted])
    ax.set_title(f"{title} — Ucus Suresi Dagilimi (n={len(duration_min)} ucus)"
                 f"{holdout_note}")
    ax.set_xlabel("sure (dakika)")
    ax.set_ylabel("ucus sayisi")
    ax.legend()
    _save(fig, out_dir / "flight_duration_hist.png", files)
    stats["duration_minutes"] = {"median": float(duration_min.median()),
                                 "max": float(duration_min.max())}
    return stats


# --------------------------------------------------------------------------
# Bolum 3 -- feature analitikleri (ANA CIKTI)
# --------------------------------------------------------------------------

def section_features(source, frame, flight_labels, out_dir: Path, files: dict,
                     sead_masks=None) -> dict:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    stats: dict = {}
    feature_cols = feature_columns_of(frame)
    title = TITLES[source]
    normal_mask = frame["source_id"].map(
        lambda s: flight_labels[s] in NORMAL_LABELS).to_numpy()

    # kategori -> pozitif satir maskesi
    if source == "uav_sead":
        cat_masks = dict(sead_masks or {})
    else:
        cat_masks = {}
        for label in sorted(frame["label"].unique()):
            if label in NORMAL_LABELS or label == "unknown":
                continue
            cat_masks[label] = (frame["label"] == label).to_numpy()

    rng = np.random.default_rng(SEED)
    neg_idx = np.flatnonzero(normal_mask)
    if len(neg_idx) > AUC_MAX_NEGATIVES:
        neg_idx = np.sort(rng.choice(neg_idx, size=AUC_MAX_NEGATIVES, replace=False))

    # 1. tek-feature AUC + q99 orani matrisi
    rows = []
    eps = np.finfo(float).eps
    for feature in feature_cols:
        values = frame[feature].to_numpy(dtype=float)
        neg = values[neg_idx]
        neg_abs_q99 = float(np.nanquantile(np.abs(neg), 0.99)) if np.isfinite(neg).any() else np.nan
        for category, mask in cat_masks.items():
            pos = values[mask]
            auc = rank_auc(pos, neg)
            pos_abs_q99 = (float(np.nanquantile(np.abs(pos), 0.99))
                           if np.isfinite(pos).any() else np.nan)
            q99_ratio = (pos_abs_q99 / max(neg_abs_q99, eps)
                         if np.isfinite(pos_abs_q99) and np.isfinite(neg_abs_q99)
                         else np.nan)
            rows.append({
                "feature": feature, "category": category, "auc": auc,
                "separation": max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
                "q99_abs_ratio": q99_ratio,
                "n_pos": int(np.isfinite(pos).sum()),
                "n_neg": int(np.isfinite(neg).sum()),
                "pos_nonnull_frac": float(np.isfinite(pos).mean()) if len(pos) else np.nan,
                "neg_nonnull_frac": float(np.isfinite(neg).mean()) if len(neg) else np.nan,
            })
    auc_table = pd.DataFrame(rows)
    _write_csv(auc_table, out_dir / "feature_auc_matrix.csv", files)

    pivot = auc_table.pivot(index="feature", columns="category", values="auc")
    order = pivot.max(axis=1).combine(1.0 - pivot.min(axis=1), max)
    pivot = pivot.loc[order.sort_values(ascending=False).index]
    n_pos_of = {c: int(auc_table[auc_table["category"] == c]["n_pos"].max())
                for c in pivot.columns}
    col_labels = [f"{c}\n(n_satir={n_pos_of[c]:,})" for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(pivot.columns)),
                                    max(6, 0.16 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r",
                   vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=25,
                  ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot)), pivot.index, fontsize=5)
    ax.set_title(f"{title} — Feature-Kategori Ayristirma (tek-feature satir AUC; "
                 f"normal referans n={len(neg_idx):,} satir)")
    fig.colorbar(im, ax=ax, label="AUC (0.5=ayrisma yok)")
    _save(fig, out_dir / "feature_auc_heatmap.png", files)

    q_pivot = auc_table.pivot(index="feature", columns="category",
                              values="q99_abs_ratio").loc[pivot.index]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(pivot.columns)),
                                    max(6, 0.16 * len(pivot))))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2(q_pivot.to_numpy())
    im = ax.imshow(np.clip(log_ratio, -3, 3), aspect="auto", cmap="PuOr_r",
                   vmin=-3, vmax=3)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=25,
                  ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot)), pivot.index, fontsize=5)
    ax.set_title(f"{title} — Seyrek Imza Gostergesi: log2(kategori q99 / normal q99), "
                 "|x| uzerinden")
    fig.colorbar(im, ax=ax, label="log2 q99 orani (0=fark yok)")
    _save(fig, out_dir / "feature_q99_heatmap.png", files)

    best = (auc_table.sort_values("separation", ascending=False)
            .groupby("category").head(5))
    stats["top_by_category"] = {
        category: [{"feature": r["feature"], "auc": round(r["auc"], 3),
                    "q99_abs_ratio": (round(r["q99_abs_ratio"], 2)
                                      if np.isfinite(r["q99_abs_ratio"]) else None)}
                   for _, r in group.iterrows()]
        for category, group in best.groupby("category")
    }

    # 2. Spearman korelasyon (hiyerarsik siralama) + gereksiz ciftler
    sub_idx = deterministic_subsample_index(
        frame, per_flight_cap=PER_FLIGHT_CAP, max_points=CORR_MAX_POINTS)
    sub = frame.loc[sub_idx, feature_cols]
    usable = [c for c in feature_cols if sub[c].notna().sum() >= 100
              and sub[c].nunique(dropna=True) > 1]
    rho = sub[usable].rank().corr()
    rho_filled = rho.fillna(0.0).to_numpy().copy()
    np.fill_diagonal(rho_filled, 1.0)
    distance = 1.0 - np.abs(rho_filled)
    distance = (distance + distance.T) / 2.0
    order_idx = leaves_list(linkage(squareform(distance, checks=False),
                                    method="average"))
    ordered = rho.iloc[order_idx, order_idx]
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(ordered.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered)), ordered.columns, rotation=90, fontsize=4)
    ax.set_yticks(range(len(ordered)), ordered.index, fontsize=4)
    ax.set_title(f"{title} — Feature Spearman Korelasyonu "
                 f"(hiyerarsik sirali, n={len(sub):,} satir orneklem)")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    _save(fig, out_dir / "spearman_heatmap.png", files)

    pairs = []
    cols = list(rho.columns)
    rho_np = rho.to_numpy()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if np.isfinite(rho_np[i, j]) and abs(rho_np[i, j]) > REDUNDANT_RHO:
                pairs.append({"feature_a": cols[i], "feature_b": cols[j],
                              "spearman_rho": float(rho_np[i, j])})
    redundant = pd.DataFrame(pairs).sort_values(
        "spearman_rho", key=np.abs, ascending=False) if pairs else pd.DataFrame(
        columns=["feature_a", "feature_b", "spearman_rho"])
    _write_csv(redundant, out_dir / "redundant_pairs.csv", files)
    stats["n_redundant_pairs"] = int(len(redundant))
    stats["n_features_in_corr"] = len(usable)
    stats["categories"] = sorted(cat_masks)
    return stats


# --------------------------------------------------------------------------
# IF-fuzyon satir skorlari (embedding boyamasi + SEAD tanilama icin)
# --------------------------------------------------------------------------

def sead_split_scores(frame: pd.DataFrame, split_name: str, config: dict,
                      *, restrict_ids: set[str] | None = None) -> pd.DataFrame:
    """ML-9 full_matrix split modelleriyle existing_fusion satir skorlari."""
    split_dir = ML9_RUN / split_name
    scaler = json.loads((split_dir / "scaler.json").read_text(encoding="utf-8"))
    calibration = json.loads((split_dir / "calibration.json").read_text(encoding="utf-8"))
    split = config["splits"][split_name]
    val_ids = set(split["val"])

    work = frame
    if restrict_ids is not None:
        work = frame[frame["source_id"].isin(restrict_ids | val_ids)]
    needed = sorted({c for name in PX4_ML7_CANDIDATE_MODULES
                     for c in calibration.get(name, {}).get("feature_columns", [])})
    scaled = apply_scaler_params(
        work[["source_id", "t_rel_s", "label", *needed]], scaler)
    out = scaled[["source_id", "t_rel_s", "label"]].copy()
    val_mask = out["source_id"].isin(val_ids).to_numpy()
    modules = []
    for name in PX4_ML7_CANDIDATE_MODULES:
        model_path = split_dir / "models" / f"{name}.joblib"
        if not model_path.exists() or name not in calibration:
            continue
        model = joblib.load(model_path)
        raw = anomaly_scores(model, scaled[calibration[name]["feature_columns"]])
        out[name] = empirical_probability(raw[val_mask], raw)
        modules.append(name)
    out["existing_fusion"] = max_score_fusion(out, modules)
    return out


def bundle_row_fusion(source: str, frame: pd.DataFrame) -> pd.Series:
    """ML-6 paket IF modulleriyle satir fuzyon skoru (val'e gore kalibre)."""
    fitted, manifest = load_modular_iforest_bundle(IF_BUNDLES[source])
    scaler = json.loads((IF_BUNDLES[source] / "scaler.json").read_text(encoding="utf-8"))
    needed = sorted({c for item in fitted.values() for c in item["feature_columns"]})
    scaled = apply_scaler_params(
        frame[["source_id", "t_rel_s", "label", *needed]], scaler)
    val_mask = scaled["source_id"].isin(set(manifest["validation_flights"])).to_numpy()
    probs = {}
    for name, item in fitted.items():
        raw = anomaly_scores(item["model"], scaled[item["feature_columns"]])
        probs[name] = empirical_probability(raw[val_mask], raw)
    return pd.DataFrame(probs, index=frame.index).max(axis=1)


# --------------------------------------------------------------------------
# Bolum 2 -- embedding projeksiyonlari (PCA + t-SNE)
# --------------------------------------------------------------------------

def _scatter(ax, xy, values, *, categorical: bool, title: str, cmap="tab10"):
    if categorical:
        levels = sorted(pd.Series(values).unique())
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(len(levels), 2)))
        for level, color in zip(levels, colors):
            mask = np.asarray(values) == level
            ax.scatter(xy[mask, 0], xy[mask, 1], s=3, alpha=0.5, color=color,
                       label=f"{level} (n={int(mask.sum()):,})", linewidths=0)
        ax.legend(markerscale=3, loc="best", framealpha=0.6)
        scatter = None
    else:
        scatter = ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.6, c=values,
                             cmap="viridis", linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return scatter


def section_embeddings(source, frame, flight_labels, out_dir: Path, files: dict,
                       *, scaler: dict, row_fusion: pd.Series,
                       row_category: pd.Series) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    stats: dict = {}
    title = TITLES[source]
    work = frame
    if source == "alfa":  # 'unknown' kesif ucusu etiketi guvenilmez; projeksiyona girmez
        work = frame[frame["label"] != "unknown"]

    idx = deterministic_subsample_index(
        work, per_flight_cap=PER_FLIGHT_CAP, max_points=MAX_POINTS[source])
    cols = [c for c in scaler["feature_columns"] if c in frame.columns]
    scaled = apply_scaler_params(frame.loc[idx, ["source_id", *cols]], scaler)
    X = np.clip(np.nan_to_num(scaled[cols].to_numpy(dtype=np.float32), nan=0.0),
                -EMBED_CLIP, EMBED_CLIP)
    labels = row_category.loc[idx].to_numpy()
    fusion = row_fusion.loc[idx].to_numpy(dtype=float)
    source_ids = frame.loc[idx, "source_id"].to_numpy()
    is_normal_flight = np.array(
        [flight_labels[s] in NORMAL_LABELS for s in source_ids])
    binary = np.where(labels == "normal", "normal", "anomali")
    n = len(idx)
    stats["n_points"] = int(n)
    stats["n_flights"] = int(pd.Series(source_ids).nunique())

    pca = PCA(n_components=min(50, X.shape[1]), random_state=SEED)
    X50 = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 4))
    ratios = pca.explained_variance_ratio_
    ax.bar(range(1, min(21, len(ratios) + 1)), ratios[:20], color="tab:blue",
           label="bilesen basina")
    ax.plot(range(1, len(ratios) + 1), np.cumsum(ratios), color="tab:red",
            label="kumulatif")
    ax.set_title(f"{title} — PCA Aciklanan Varyans (n={n:,} satir, "
                 f"{X.shape[1]} feature; gorsellestirme icin ±{EMBED_CLIP:.0f} "
                 "IQR kirpma)")
    ax.set_xlabel("bilesen")
    ax.set_ylabel("varyans orani")
    ax.legend()
    _save(fig, out_dir / "pca_variance.png", files)
    stats["pca_var_2comp"] = float(np.cumsum(ratios)[1])

    tsne = TSNE(n_components=2, perplexity=30.0, init="pca",
                random_state=SEED, max_iter=1000)
    X_tsne = tsne.fit_transform(X50)

    projections = {"pca": X50[:, :2], "tsne": X_tsne}
    for method, xy in projections.items():
        method_title = "PCA (2B)" if method == "pca" else "t-SNE (2B)"

        fig, ax = plt.subplots(figsize=(8, 7))
        _scatter(ax, xy, labels, categorical=True,
                 title=f"{title} — {method_title}: Anomali Kategorisi "
                       f"(n={n:,} satir)")
        if method == "tsne":
            fig.text(0.5, 0.005, TSNE_NOTE, ha="center", fontsize=8)
        _save(fig, out_dir / f"{method}_label.png", files)

        fig, ax = plt.subplots(figsize=(8, 7))
        _scatter(ax, xy, binary, categorical=True,
                 title=f"{title} — {method_title}: Normal vs Anomali "
                       f"(n={n:,} satir)")
        if method == "tsne":
            fig.text(0.5, 0.005, TSNE_NOTE, ha="center", fontsize=8)
        _save(fig, out_dir / f"{method}_binary.png", files)

        fig, ax = plt.subplots(figsize=(8, 7))
        scatter = _scatter(ax, xy, fusion, categorical=False,
                           title=f"{title} — {method_title}: IF-Fuzyon Skoru "
                                 f"(n={n:,} satir; modelin 'uzaklik' algisi)")
        fig.colorbar(scatter, ax=ax, label="kalibre fuzyon skoru (0-1)")
        if method == "tsne":
            fig.text(0.5, 0.005, TSNE_NOTE, ha="center", fontsize=8)
        _save(fig, out_dir / f"{method}_fusion.png", files)

        if source == "uav_sead":
            sessions = pd.Series([session_of(s) for s in source_ids])
            codes = sessions[is_normal_flight].astype("category").cat.codes.to_numpy()
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.scatter(xy[~is_normal_flight, 0], xy[~is_normal_flight, 1], s=3,
                       alpha=0.15, color="0.8", linewidths=0, label="anomalili ucus")
            ax.scatter(xy[is_normal_flight, 0], xy[is_normal_flight, 1], s=3,
                       alpha=0.6, c=codes, cmap="tab20", linewidths=0)
            n_sessions = int(sessions[is_normal_flight].nunique())
            ax.set_title(f"{title} — {method_title}: Normal Ucuslarin Oturumu "
                         f"({n_sessions} oturum, n={int(is_normal_flight.sum()):,} satir)")
            ax.set_xticks([])
            ax.set_yticks([])
            if method == "tsne":
                fig.text(0.5, 0.005, TSNE_NOTE, ha="center", fontsize=8)
            _save(fig, out_dir / f"{method}_session.png", files)
    return stats


# --------------------------------------------------------------------------
# Bolum 4 -- model tanilama gorselleri
# --------------------------------------------------------------------------

def _roc_pr_band(ax_roc, ax_pr, curves: list[tuple[np.ndarray, np.ndarray]],
                 pr_curves: list[tuple[np.ndarray, np.ndarray]], label: str,
                 color: str):
    from sklearn.metrics import auc as sk_auc

    fpr_grid = np.linspace(0, 1, 201)
    tprs, aucs = [], []
    for fpr, tpr in curves:
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        aucs.append(sk_auc(fpr, tpr))
    tprs = np.vstack(tprs)
    ax_roc.plot(fpr_grid, tprs.mean(axis=0), color=color,
                label=f"{label} (AUC={np.mean(aucs):.3f})")
    if len(curves) > 1:
        ax_roc.fill_between(fpr_grid, tprs.min(axis=0), tprs.max(axis=0),
                            color=color, alpha=0.2)

    rec_grid = np.linspace(0, 1, 201)
    precs = []
    for precision, recall in pr_curves:
        order = np.argsort(recall)
        precs.append(np.interp(rec_grid, recall[order], precision[order]))
    precs = np.vstack(precs)
    ax_pr.plot(rec_grid, precs.mean(axis=0), color=color, label=label)
    if len(pr_curves) > 1:
        ax_pr.fill_between(rec_grid, precs.min(axis=0), precs.max(axis=0),
                           color=color, alpha=0.2)
    return float(np.mean(aucs))


def _confusion_figures(records: pd.DataFrame, *, title: str, subtitle: str,
                       out_dir: Path, files: dict, prefix: str) -> dict:
    """records: flight_label, is_anomalous, alarm kolonlu ucus tablosu."""
    tp = int(((records["is_anomalous"]) & (records["alarm"])).sum())
    fn = int(((records["is_anomalous"]) & (~records["alarm"])).sum())
    fp = int(((~records["is_anomalous"]) & (records["alarm"])).sum())
    tn = int(((~records["is_anomalous"]) & (~records["alarm"])).sum())
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"n={value}", ha="center", va="center",
                color="black" if value < matrix.max() * 0.6 else "white")
    ax.set_xticks([0, 1], ["alarm yok", "alarm var"])
    ax.set_yticks([0, 1], ["gercek: normal", "gercek: anomali"])
    ax.set_title(f"{title}\n{subtitle}\n"
                 "(model sinif tahmin etmez; ikili alarm karari)")
    fig.colorbar(im, ax=ax)
    _save(fig, out_dir / f"{prefix}_confusion_binary.png", files)

    by_type = (records.groupby("flight_label")
               .agg(alarm=("alarm", "sum"), toplam=("alarm", "size"))
               .assign(miss=lambda d: d["toplam"] - d["alarm"]))
    fig, ax = plt.subplots(figsize=(6.5, max(3.2, 0.5 * len(by_type))))
    data = by_type[["alarm", "miss"]].to_numpy()
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    for (i, j), value in np.ndenumerate(data):
        ax.text(j, i, f"n={int(value)}", ha="center", va="center",
                color="black" if value < data.max() * 0.6 else "white")
    ax.set_xticks([0, 1], ["tespit (alarm)", "kacirilan (miss)"])
    ax.set_yticks(range(len(by_type)),
                  [f"{lab} (n={int(t)})" for lab, t in by_type["toplam"].items()])
    ax.set_title(f"{title} — Tur Bazli Tespit\n{subtitle}\n"
                 "(unsupervised ikili dedektorde cok-sinifli CM'in karsiligi)")
    fig.colorbar(im, ax=ax)
    _save(fig, out_dir / f"{prefix}_confusion_by_type.png", files)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def section_model_sead(frame, dev_ids, flight_labels, config, out_dir: Path,
                       files: dict, sead_masks, t0, split00_scores) -> dict:
    from sklearn.metrics import precision_recall_curve, roc_curve

    stats: dict = {}
    title = TITLES["uav_sead"]
    row_cat = row_category_labels("uav_sead", frame, flight_labels, sead_masks)

    # 1. skor dagilimlari (satir duzeyi, split_00 existing_fusion)
    fusion = split00_scores["existing_fusion"]
    groups, names = [], []
    for level in ["normal"] + sorted(k for k in row_cat.unique()
                                     if k not in ("normal",)):
        values = fusion[(row_cat == level).to_numpy()].dropna().to_numpy()
        if len(values) < 50:
            continue
        groups.append(values)
        names.append(f"{level}\n(n={len(values):,})")
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(groups)), 4.5))
    ax.violinplot(groups, showmedians=True)
    ax.set_xticks(range(1, len(names) + 1), names, fontsize=7)
    ax.set_ylabel("kalibre IF-fuzyon skoru (0-1)")
    ax.set_title(f"{title} — Skor Dagilimi: Normal vs Kategoriler "
                 "(satir duzeyi, split_00)")
    _save(fig, out_dir / "score_violin.png", files)

    # 2-3. ROC/PR (5 seed) + advisory CUSUM calisma noktasinda CM
    roc_curves, pr_curves = [], []
    cm_records = []
    for split_name in sorted(config["splits"]):
        split = config["splits"][split_name]
        test_ids = set(split["test"])
        scored = (split00_scores if split_name == "split_00"
                  else sead_split_scores(frame, split_name, config,
                                         restrict_ids=test_ids))
        streams = last_causal_per_bucket(
            scored[scored["source_id"].isin(test_ids)], stride_seconds=1.0,
            columns=["source_id", "t_rel_s", "existing_fusion"])
        flight_scores = streams.groupby("source_id")["existing_fusion"].max()
        truth = np.array([flight_labels[s] not in NORMAL_LABELS
                          for s in flight_scores.index])
        fpr, tpr, _ = roc_curve(truth, flight_scores.to_numpy())
        precision, recall, _ = precision_recall_curve(truth, flight_scores.to_numpy())
        roc_curves.append((fpr, tpr))
        pr_curves.append((precision, recall))

        policies = json.loads((ML9_RUN / split_name / "policies.json")
                              .read_text(encoding="utf-8"))
        policy = policy_from_dict(policies["existing_fusion:advisory:cusum"])
        for source_id, group in streams.groupby("source_id"):
            onsets = policy.apply(
                group.sort_values("t_rel_s")["existing_fusion"].to_numpy(float))
            cm_records.append({
                "split": split_name, "source_id": source_id,
                "flight_label": flight_labels[source_id],
                "is_anomalous": flight_labels[source_id] not in NORMAL_LABELS,
                "alarm": bool(onsets.any()),
            })

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    mean_auc = _roc_pr_band(ax_roc, ax_pr, roc_curves, pr_curves,
                            "IF-fuzyon (5 seed)", "tab:blue")
    n_test = int(np.mean([len(set(config["splits"][s]["test"]))
                          for s in config["splits"]]))
    fig.suptitle(f"{title} — Ucus Duzeyi ROC/PR (test ort. n={n_test} ucus; "
                 "bant=5 seed min-max)")
    ax_roc.plot([0, 1], [0, 1], ls="--", color="0.6")
    ax_roc.set_xlabel("yanlis pozitif orani")
    ax_roc.set_ylabel("dogru pozitif orani")
    ax_roc.set_title("ROC")
    ax_roc.legend()
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("Ucus Duzeyi Precision-Recall")
    ax_pr.legend()
    _save(fig, out_dir / "roc_pr_flight.png", files)
    stats["flight_roc_auc_mean_5seed"] = mean_auc

    cm_frame = pd.DataFrame(cm_records)
    stats["confusion"] = _confusion_figures(
        cm_frame, title=f"{title} — Ucus Duzeyi Karar Matrisi",
        subtitle="advisory CUSUM calisma noktasi; 5 seed toplam "
                 f"(n={len(cm_frame)} ucus-degerlendirme)",
        out_dir=out_dir, files=files, prefix="sead")

    # 4. ornek zaman serileri (split_00, kategori basina en fazla 2 test ucusu)
    split00 = config["splits"]["split_00"]
    test00 = set(split00["test"])
    policies00 = json.loads((ML9_RUN / "split_00" / "policies.json")
                            .read_text(encoding="utf-8"))
    policy00 = policy_from_dict(policies00["existing_fusion:advisory:cusum"])
    by_flight = load_uav_sead_ranges_by_category(SEAD_LABELS)
    per_category: dict[str, list[str]] = {}
    for sid in sorted(test00):
        for category in _combined_categories(by_flight.get(sid, {})):
            per_category.setdefault(category, []).append(sid)
    for category, flights in sorted(per_category.items()):
        for sid in flights[:2]:
            scored = split00_scores[split00_scores["source_id"] == sid]
            stream = last_causal_per_bucket(
                scored, stride_seconds=1.0,
                columns=["source_id", "t_rel_s", "existing_fusion"])
            t = stream["t_rel_s"].to_numpy(float)
            s = stream["existing_fusion"].to_numpy(float)
            onsets = policy00.apply(s)
            fig, ax = plt.subplots(figsize=(9, 3.4))
            ax.plot(t, s, lw=0.8, color="tab:blue", label="IF-fuzyon skoru")
            spans = _combined_categories(by_flight.get(sid, {})).get(category, [])
            for start, end in spans:
                ax.axvspan((start - t0[sid]) / 1e6, (end - t0[sid]) / 1e6,
                           color="tab:red", alpha=0.2)
            if onsets.any():
                ax.plot(t[onsets], s[onsets], "v", color="tab:red", ms=7,
                        label=f"alarm onset (n={int(onsets.sum())})")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("ucus zamani (s)")
            ax.set_ylabel("skor")
            safe = sid.replace("/", "__")
            ax.set_title(f"{title} — {category}: {sid} "
                         f"(kirmizi bant=etiketli aralik, n={len(spans)}; "
                         "advisory CUSUM alarmi)")
            ax.legend(loc="upper right")
            safe_cat = category.replace("+", "_").replace(".", "_").replace(" ", "_")
            _save(fig, out_dir / f"timeseries_{safe_cat}_{safe}.png", files)
    return stats


def section_model_alfa(frame, flight_labels, out_dir: Path, files: dict,
                       row_fusion: pd.Series) -> dict:
    from sklearn.metrics import precision_recall_curve, roc_curve

    stats: dict = {}
    title = TITLES["alfa"]
    model, scaler, calibration, manifest = load_lstm_bundle(ALFA_LSTM_BUNDLE)
    feature_cols = manifest["feature_columns"]
    scaled = apply_scaler_params(
        frame[["source_id", "label", "t_rel_s", *feature_cols]], scaler)
    X, M, meta = build_windows(
        scaled, feature_cols, window=int(manifest["window"]),
        stride=int(manifest["stride"]), max_gap_s=float(manifest["max_gap_s"]))
    meta["score"] = reconstruction_scores(model, X, M)
    threshold = float(calibration["window_threshold_q99"])
    train_ids = set(manifest["train_flights"])
    eval_meta = meta[~meta["source_id"].isin(train_ids)
                     & (meta["label"] != "unknown")]

    # 1. pencere skoru dagilimi (log10)
    groups, names = [], []
    for level in ["normal"] + sorted(l for l in eval_meta["label"].unique()
                                     if l != "normal"):
        values = np.log10(eval_meta.loc[eval_meta["label"] == level, "score"]
                          .to_numpy(float) + 1.0)
        if len(values) < 20:
            continue
        groups.append(values)
        names.append(f"{level}\n(n={len(values):,})")
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(groups)), 4.5))
    ax.violinplot(groups, showmedians=True)
    ax.axhline(np.log10(threshold + 1.0), color="tab:red", ls="--",
               label="pencere esigi (val q99)")
    ax.set_xticks(range(1, len(names) + 1), names, fontsize=7)
    ax.set_ylabel("log10(LSTM-AE pencere skoru)")
    ax.set_title(f"{title} — Pencere Skoru Dagilimi (egitim ucuslari haric)")
    ax.legend()
    _save(fig, out_dir / "score_violin.png", files)

    # 2. ucus-duzeyi ROC/PR: LSTM-AE + IF-fuzyon (tek paket model = tek egri)
    flight_lstm = eval_meta.groupby("source_id")["score"].max()
    eval_ids = list(flight_lstm.index)
    truth = np.array([flight_labels[s] not in NORMAL_LABELS for s in eval_ids])
    fusion_rows = pd.DataFrame({"source_id": frame["source_id"],
                                "fusion": row_fusion})
    if_flight = (fusion_rows[fusion_rows["source_id"].isin(eval_ids)]
                 .groupby("source_id")["fusion"].max().reindex(eval_ids))
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    aucs = {}
    for name, scores, color in [
        ("LSTM-AE", flight_lstm.to_numpy(float), "tab:blue"),
        ("IF-fuzyon", if_flight.to_numpy(float), "tab:orange"),
    ]:
        fpr, tpr, _ = roc_curve(truth, scores)
        precision, recall, _ = precision_recall_curve(truth, scores)
        aucs[name] = _roc_pr_band(ax_roc, ax_pr, [(fpr, tpr)],
                                  [(precision, recall)], name, color)
    n_norm = int((~truth).sum())
    fig.suptitle(f"{title} — Ucus Duzeyi ROC/PR (n={len(eval_ids)} ucus, "
                 f"{n_norm} normal; tek paket model, band yok)")
    ax_roc.plot([0, 1], [0, 1], ls="--", color="0.6")
    ax_roc.set_xlabel("yanlis pozitif orani")
    ax_roc.set_ylabel("dogru pozitif orani")
    ax_roc.set_title("ROC")
    ax_roc.legend()
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("Ucus Duzeyi Precision-Recall")
    ax_pr.legend()
    _save(fig, out_dir / "roc_pr_flight.png", files)
    stats["flight_roc_auc"] = {k: round(v, 4) for k, v in aucs.items()}

    # 3. calisma noktasinda karar matrisleri (pencere q99 esigi)
    records = pd.DataFrame({
        "source_id": eval_ids,
        "flight_label": [flight_labels[s] for s in eval_ids],
        "is_anomalous": truth,
        "alarm": (flight_lstm > threshold).to_numpy(),
    })
    stats["confusion"] = _confusion_figures(
        records, title=f"{title} — Ucus Duzeyi Karar Matrisi (LSTM-AE)",
        subtitle=f"pencere esigi=val q99; n={len(records)} ucus",
        out_dir=out_dir, files=files, prefix="alfa")

    # 4. ornek zaman serileri: anomali turu basina ilk 2 ucus
    labels_of = {s: flight_labels[s] for s in eval_ids}
    for label in sorted(set(labels_of.values()) - NORMAL_LABELS):
        flights = sorted(s for s, l in labels_of.items() if l == label)[:2]
        for sid in flights:
            flight_meta = eval_meta[eval_meta["source_id"] == sid]
            rows = frame[frame["source_id"] == sid].sort_values("t_rel_s")
            anom_t = rows.loc[rows["label"] != "normal", "t_rel_s"]
            fig, ax = plt.subplots(figsize=(9, 3.4))
            ax.plot(flight_meta["t_end"], np.log10(flight_meta["score"] + 1.0),
                    lw=0.9, color="tab:blue", label="LSTM-AE pencere skoru")
            ax.axhline(np.log10(threshold + 1.0), color="tab:red", ls="--",
                       label="esik (val q99)")
            if len(anom_t):
                ax.axvspan(float(anom_t.min()), float(anom_t.max()),
                           color="tab:red", alpha=0.15)
            ax.set_xlabel("ucus zamani (s)")
            ax.set_ylabel("log10(skor)")
            ax.set_title(f"{title} — {label}: {sid} "
                         f"(kirmizi bant=ariza bolgesi, n={len(flight_meta)} pencere)")
            ax.legend(loc="upper left")
            _save(fig, out_dir / f"timeseries_{label}_{sid}.png", files)
    stats["n_windows"] = int(len(meta))
    return stats


# --------------------------------------------------------------------------
# calistirici
# --------------------------------------------------------------------------

def run(source: str, sections: set[int]) -> Path:
    out_dir = OUT_ROOT / source
    frame, dev_ids, holdout_ids, flight_labels, config = load_development(source)
    files: dict = {}
    section_stats: dict = {}

    sead_masks, t0 = (None, None)
    if source == "uav_sead":
        sead_masks, t0 = sead_category_masks(frame, dev_ids)

    if 1 in sections:
        section_stats["s1_portfolio"] = section_portfolio(
            source, frame, dev_ids, holdout_ids, flight_labels,
            out_dir / "s1_portfolio", files, sead_masks=sead_masks)

    if 3 in sections:
        section_stats["s3_features"] = section_features(
            source, frame, flight_labels, out_dir / "s3_features", files,
            sead_masks=sead_masks)

    row_fusion = split00_scores = None
    if sections & {2, 4}:
        if source == "uav_sead":
            split00_scores = sead_split_scores(frame, "split_00", config)
            row_fusion = split00_scores["existing_fusion"]
            row_fusion.index = frame.index
        else:
            row_fusion = bundle_row_fusion(source, frame)

    if 2 in sections:
        if source == "uav_sead":
            scaler = json.loads((ML9_RUN / "split_00" / "scaler.json")
                                .read_text(encoding="utf-8"))
        else:
            scaler = json.loads((IF_BUNDLES[source] / "scaler.json")
                                .read_text(encoding="utf-8"))
        row_category = row_category_labels(source, frame, flight_labels, sead_masks)
        section_stats["s2_embeddings"] = section_embeddings(
            source, frame, flight_labels, out_dir / "s2_embeddings", files,
            scaler=scaler, row_fusion=row_fusion, row_category=row_category)

    if 4 in sections:
        if source == "uav_sead":
            section_stats["s4_model"] = section_model_sead(
                frame, dev_ids, flight_labels, config, out_dir / "s4_model",
                files, sead_masks, t0, split00_scores)
        elif source == "alfa":
            section_stats["s4_model"] = section_model_alfa(
                frame, flight_labels, out_dir / "s4_model", files, row_fusion)
        else:
            section_stats["s4_model"] = {
                "skipped": "uav_attack icin plan geregi opsiyonel; uretilmedi"}

    # manifest: dizindeki TUM png/csv'leri checksum'la (bolum bolum kosulabilir)
    manifest_path = out_dir / "viz_manifest.json"
    previous_sections = {}
    if manifest_path.exists():
        previous_sections = json.loads(
            manifest_path.read_text(encoding="utf-8")).get("sections", {})
    previous_sections.update(section_stats)
    all_files = sorted(p for p in out_dir.rglob("*")
                       if p.is_file() and p.suffix in {".png", ".csv"})
    manifest = {
        "artifact_schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ML-11 visualization (read-only analysis)",
        "source": source,
        "seed": SEED,
        "per_flight_cap": PER_FLIGHT_CAP,
        "max_points": MAX_POINTS[source],
        "development_flights": len(dev_ids),
        "blind_holdout_flights": len(holdout_ids),
        "blind_holdout_read": False,
        "development_source_ids_sha256": _ids_sha256(dev_ids),
        "feature_table_sha256": _sha256(FEATURE_PATHS[source]),
        "split_manifest_sha256": _sha256(SPLIT_PATH),
        "sections": previous_sections,
        "files": {str(p.relative_to(out_dir)).replace("\\", "/"): _sha256(p)
                  for p in all_files},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCES, required=True)
    parser.add_argument("--sections", default="1,2,3,4",
                        help="virgullu bolum listesi (1=karne 2=embedding "
                             "3=feature 4=model)")
    args = parser.parse_args()
    sections = {int(part) for part in args.sections.split(",")}
    out = run(args.dataset, sections)
    print(f"ML-11 viz artifact: {out}")


if __name__ == "__main__":
    main()
