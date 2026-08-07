"""Kural-bazli penalty skorlayicisini (adsb/rules.py) NN'lerle BIREBIR AYNI
duzende degerlendirir: ayni 60 Silver parca, ayni SEED=0 train/val bolmesi, ayni
kalici sentetik korpus, ayni pencere birimi (WINDOW=12, STRIDE=6, MAX_GAP_S=60),
ayni metrikler (senaryo-bazi AUC/F1 + pooled AUC). Sonuc dogrudan ADR-024
tablosuyla kiyaslanabilir.

Ogrenme yok -- yalniz train-normal satirlarindan median/MAD kalibrasyonu.

Kullanim:
    python scripts/adsb_evaluate_rule_scorer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.diagnostics import fit_score_baseline, z_score_confidence  # noqa: E402
from adsb.features import build_feature_table  # noqa: E402
from adsb.rules import ResidualRuleScorer  # noqa: E402
from adsb.segmentation import segment_flights  # noqa: E402
from adsb.synthetic import PHYSICS_BREAK_RECIPES  # noqa: E402
from adsb.windowing import build_windows  # noqa: E402

SILVER_DIR = Path("data/objectstore/silver/adsblol_historical")
SYNTHETIC_DIR = Path("data/objectstore/synthetic/adsb")
OUT_PATH = Path("artifacts/adsb/models/rule_scorer_report.json")
N_PARTS = 60
WINDOW = 12
STRIDE = 6
MAX_GAP_S = 60.0
SEED = 0
CONFIDENCE_THRESHOLD = 0.95


def window_scores(feat_df: pd.DataFrame, scorer: ResidualRuleScorer) -> np.ndarray:
    """Satir-penalty'lerini NN'lerle ayni pencere birimine indirger: penalty'yi
    tek-kanalli bir 'feature' gibi pencereleyip pencere-ici ortalama alinir."""
    df = feat_df.copy()
    df["rule_penalty"] = scorer.row_penalties(feat_df)
    X, M, _ = build_windows(df, ["rule_penalty"], window=WINDOW, stride=STRIDE, max_gap_s=MAX_GAP_S)
    if len(X) == 0:
        return np.zeros(0)
    return (X * M).sum(axis=(1, 2)) / np.clip(M.sum(axis=(1, 2)), 1.0, None)


def main() -> None:
    files = sorted(SILVER_DIR.glob("*.parquet"))[:N_PARTS]
    print(f"{len(files)} Silver parcasi okunuyor...", flush=True)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {len(df)} satir", flush=True)

    seg = segment_flights(df, gap_s=1800.0)
    feat = build_feature_table(seg)

    flight_ids = np.array(feat["flight_id"].unique(), dtype=object)
    rng = np.random.default_rng(SEED)
    rng.shuffle(flight_ids)
    split = int(len(flight_ids) * 0.8)
    train_ids = set(flight_ids[:split])

    synthetic_flight_ids = set(
        pd.read_parquet(SYNTHETIC_DIR / "clean.parquet", columns=["flight_id"])["flight_id"].unique()
    )
    overlap = train_ids & synthetic_flight_ids
    if overlap:
        raise RuntimeError(f"SIZINTI: {len(overlap)} ucus hem kalibrasyonda hem korpusta")
    print(f"  sizinti kontrolu GECTI ({len(train_ids)} train / {len(synthetic_flight_ids)} korpus)", flush=True)

    train_feat = feat[feat["flight_id"].isin(train_ids)]
    scorer = ResidualRuleScorer().fit(train_feat)
    print("  kalibrasyon:", json.dumps(scorer.calibration_, indent=2), flush=True)

    train_win_scores = window_scores(train_feat, scorer)
    baseline = fit_score_baseline(train_win_scores)
    print(f"  train pencere skoru: n={len(train_win_scores)}, sifir-orani="
          f"{(train_win_scores == 0).mean():.3f}", flush=True)

    print("  clean korpus skorlaniyor...", flush=True)
    clean_feat = build_feature_table(pd.read_parquet(SYNTHETIC_DIR / "clean.parquet"))
    clean_scores = window_scores(clean_feat, scorer)

    report: dict = {
        "scorer": scorer.to_dict(),
        "n_parts": N_PARTS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "train_window_zero_fraction": float((train_win_scores == 0).mean()),
        "score_baseline_median_mad": baseline,
        "per_recipe": {},
    }
    pooled_y, pooled_s = [np.zeros(len(clean_scores))], [clean_scores]

    for recipe_name in PHYSICS_BREAK_RECIPES:
        print(f"  {recipe_name} skorlaniyor...", flush=True)
        r_feat = build_feature_table(pd.read_parquet(SYNTHETIC_DIR / f"{recipe_name}.parquet"))
        r_scores = window_scores(r_feat, scorer)

        y_true = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(r_scores))])
        y_score = np.concatenate([clean_scores, r_scores])
        y_conf = np.concatenate([
            z_score_confidence(clean_scores, baseline), z_score_confidence(r_scores, baseline)
        ])
        y_pred = (y_conf >= CONFIDENCE_THRESHOLD).astype(int)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        report["per_recipe"][recipe_name] = {
            "n_clean": int(len(clean_scores)), "n_corrupt": int(len(r_scores)),
            "auc": float(roc_auc_score(y_true, y_score)),
            "f1_at_conf0.95": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy_at_conf0.95": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix_conf0.95": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "roc_fpr": fpr[:: max(1, len(fpr) // 200)].tolist(),
            "roc_tpr": tpr[:: max(1, len(tpr) // 200)].tolist(),
            "mean_clean_score": float(clean_scores.mean()),
            "mean_corrupt_score": float(r_scores.mean()),
        }
        pooled_y.append(np.ones(len(r_scores)))
        pooled_s.append(r_scores)
        print(f"    AUC={report['per_recipe'][recipe_name]['auc']:.3f}", flush=True)

    report["pooled_auc_all_recipes_vs_clean"] = float(
        roc_auc_score(np.concatenate(pooled_y), np.concatenate(pooled_s))
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npooled AUC: {report['pooled_auc_all_recipes_vs_clean']:.3f}", flush=True)
    print(f"Rapor: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
