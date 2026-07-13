"""ADSB-2: Dense-AE / LSTM-AE / LSTM-forecaster'i gercek tar verisiyle (60/638 Silver
parca) egitir, ZORUNLU magnitude_domination_check calistirir, ve kalici sentetik
korpusu (data/objectstore/synthetic/adsb/, bkz. ADR-023) kullanarak ROC/AUC/F1
hesaplar + z-score tabanli "guven" skoru uretir.

USAD bu turda HARIC -- sayisal kararsizligi (loss ~23 milyara patliyor, gradient
clipping yetmiyor) hala cozulmedi (ADR-022'de acik madde), coz ulmeden rapora
yaniltici sayilar eklenmeyecek.

N_PARTS, adsb_generate_synthetic_dataset.py ile BIREBIR AYNI (60) -- ikisi de ayni
segment_flights + ayni SEED=0 80/20 flight-bazli bolmeyi kullandigi icin train
kumesiyle sentetik-korpus (val-split'ten uretildi) kumesi YAPISAL OLARAK ayrik
olmali; bu script bunu VARSAYMAZ, calisma zamaninda assert eder.

Egitim SADECE normal (enjeksiyonsuz) ucus pencereleriyle yapilir -- etiket
kullanilmaz. Sentetik pencereler hicbir zaman egitime girmez.

Kullanim:
    python scripts/adsb_train_baseline_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.diagnostics import fit_score_baseline, magnitude_domination_check, z_score_confidence  # noqa: E402
from adsb.features import PRIMARY_FEATURES, build_feature_table  # noqa: E402
from adsb.scaling import ClippedRobustScaler  # noqa: E402
from adsb.models.dense_autoencoder import DenseAutoencoder  # noqa: E402
from adsb.models.dense_autoencoder import reconstruction_scores as dense_scores  # noqa: E402
from adsb.models.dense_autoencoder import train_dense_autoencoder  # noqa: E402
from adsb.models.lstm_autoencoder import LSTMAutoencoder  # noqa: E402
from adsb.models.lstm_autoencoder import reconstruction_scores as lstm_ae_scores  # noqa: E402
from adsb.models.lstm_autoencoder import train_lstm_autoencoder  # noqa: E402
from adsb.models.lstm_forecaster import LSTMForecaster  # noqa: E402
from adsb.models.lstm_forecaster import forecast_residual_scores  # noqa: E402
from adsb.models.lstm_forecaster import train_lstm_forecaster  # noqa: E402
from adsb.segmentation import segment_flights  # noqa: E402
from adsb.synthetic import PHYSICS_BREAK_RECIPES  # noqa: E402
from adsb.windowing import build_windows  # noqa: E402

SILVER_DIR = Path("data/objectstore/silver/adsblol_historical")
SYNTHETIC_DIR = Path("data/objectstore/synthetic/adsb")
OUT_DIR = Path("artifacts/adsb/models")
N_PARTS = 60  # adsb_generate_synthetic_dataset.py ile AYNI -- sizinti-guvenligi buna dayanir
WINDOW = 12
STRIDE = 6
HISTORY_LEN = 8
MAX_GAP_S = 60.0
EPOCHS = 15
BATCH_SIZE = 512  # varsayilan 64'te 2.85M pencerede tahmini calisma suresi saatler
                   # surdu (olculdu: Dense-AE tek basina ~36dk); 512'de ayni epoch
                   # sayisinda ~15x daha az batch, CPU'da pratik bir hiz kazanimi
SEED = 0
CONFIDENCE_THRESHOLD = 0.95


def load_real_data(n_parts: int) -> pd.DataFrame:
    files = sorted(SILVER_DIR.glob("*.parquet"))[:n_parts]
    print(f"{len(files)} Silver parcasi okunuyor...", flush=True)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {len(df)} satir, {df.source_id.nunique()} ucak", flush=True)
    return df


def prepare_windows(df: pd.DataFrame):
    seg = segment_flights(df, gap_s=1800.0)
    feat = build_feature_table(seg)

    # .unique() bir ArrowStringArray dondurebiliyor -- np.random.Generator.shuffle
    # bunun icin guvenilmez/yavas calisabiliyor (adsb_generate_synthetic_dataset.py'de
    # somut olarak dogrulandi). Duz numpy object array'e cevirip guvenli hale getiriyoruz.
    flight_ids = np.array(feat["flight_id"].unique(), dtype=object)
    rng = np.random.default_rng(SEED)
    rng.shuffle(flight_ids)
    split = int(len(flight_ids) * 0.8)
    train_ids, val_ids = set(flight_ids[:split]), set(flight_ids[split:])

    # SIZINTI KONTROLU: train kumesi, sentetik korpusun kaynagi olan val-split ile
    # AYNI SEED/segment mantigiyla uretildigi icin yapisal olarak ayrik olmali --
    # bunu varsaymak yerine calisma zamaninda dogruluyoruz.
    if (SYNTHETIC_DIR / "clean.parquet").exists():
        synthetic_flight_ids = set(pd.read_parquet(SYNTHETIC_DIR / "clean.parquet", columns=["flight_id"])["flight_id"].unique())
        overlap = train_ids & synthetic_flight_ids
        if overlap:
            raise RuntimeError(
                f"SIZINTI: {len(overlap)} ucus hem train'de hem sentetik korpusta -- "
                f"orn. {list(overlap)[:5]}. N_PARTS/SEED uyumsuzlugu olabilir."
            )
        print(f"  sizinti kontrolu GECTI: train ({len(train_ids)}) ile sentetik-korpus "
              f"({len(synthetic_flight_ids)}) ucuslari ayrik", flush=True)

    train_df = feat[feat["flight_id"].isin(train_ids)]
    val_df = feat[feat["flight_id"].isin(val_ids)]

    X_train, M_train, _ = build_windows(train_df, PRIMARY_FEATURES, window=WINDOW, stride=STRIDE, max_gap_s=MAX_GAP_S)
    X_val, M_val, meta_val = build_windows(val_df, PRIMARY_FEATURES, window=WINDOW, stride=STRIDE, max_gap_s=MAX_GAP_S)
    print(f"  train pencere: {len(X_train)}, val pencere: {len(X_val)}", flush=True)

    # KIRPILI olcekleme -- SEAD dersi (ADR-016): kirpilmamis olcekleme genlik-
    # baskinligina yol aciyordu. Fit SADECE train'den, val'e aynen uygulanir.
    scaler = ClippedRobustScaler(clip=5.0).fit(X_train, M_train)
    X_train_scaled = scaler.transform(X_train, M_train)
    X_val_scaled = scaler.transform(X_val, M_val)
    print(f"  scaler fit edildi (clip=5.0), median ornek: {scaler.median_[:3]}", flush=True)

    return X_train_scaled, M_train, X_val_scaled, M_val, meta_val, scaler


def _score_batched(score_fn, X: np.ndarray, M: np.ndarray, batch_size: int = 20_000) -> np.ndarray:
    """LSTM skorlama TUM train setini (2.85M pencere) TEK forward'ta ~20.8GB
    istemeye calisip cokuyordu (RuntimeError: not enough memory) -- skorlama,
    egitimin kendisi gibi, kucuk gruplar halinde yapilmali. Gradyan gerekmedigi
    icin (skorlama fonksiyonlari zaten torch.no_grad() kullaniyor) grup boyutu
    egitimdekinden (512) cok daha buyuk secilebilir."""
    if len(X) <= batch_size:
        return score_fn(X, M)
    parts = [score_fn(X[i:i + batch_size], M[i:i + batch_size]) for i in range(0, len(X), batch_size)]
    return np.concatenate(parts)


def load_synthetic_windows(recipe_name: str, scaler: ClippedRobustScaler):
    raw = pd.read_parquet(SYNTHETIC_DIR / f"{recipe_name}.parquet")
    feat = build_feature_table(raw)
    X, M, _ = build_windows(feat, PRIMARY_FEATURES, window=WINDOW, stride=STRIDE, max_gap_s=MAX_GAP_S)
    return scaler.transform(X, M), M


def evaluate_model(score_fn, X_train, M_train, X_val, M_val, scaler) -> dict:
    train_scores = _score_batched(score_fn, X_train, M_train)
    val_scores = _score_batched(score_fn, X_val, M_val)
    baseline = fit_score_baseline(train_scores)

    clean_X, clean_M = load_synthetic_windows("clean", scaler)
    clean_scores = _score_batched(score_fn, clean_X, clean_M)

    per_recipe = {}
    pooled_y_true = [np.zeros(len(clean_scores))]
    pooled_scores = [clean_scores]

    for recipe_name in PHYSICS_BREAK_RECIPES:
        r_X, r_M = load_synthetic_windows(recipe_name, scaler)
        r_scores = _score_batched(score_fn, r_X, r_M)
        r_conf = z_score_confidence(r_scores, baseline)
        clean_conf = z_score_confidence(clean_scores, baseline)

        y_true = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(r_scores))])
        y_score = np.concatenate([clean_scores, r_scores])
        y_conf = np.concatenate([clean_conf, r_conf])
        y_pred = (y_conf >= CONFIDENCE_THRESHOLD).astype(int)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

        per_recipe[recipe_name] = {
            "n_clean": int(len(clean_scores)), "n_corrupt": int(len(r_scores)),
            "auc": float(roc_auc_score(y_true, y_score)),
            "f1_at_conf0.95": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy_at_conf0.95": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix_conf0.95": cm,  # [[TN,FP],[FN,TP]]
            "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist(),
            "mean_clean_score": float(clean_scores.mean()), "mean_corrupt_score": float(r_scores.mean()),
        }
        pooled_y_true.append(np.ones(len(r_scores)))
        pooled_scores.append(r_scores)

    pooled_auc = roc_auc_score(np.concatenate(pooled_y_true), np.concatenate(pooled_scores))

    return {
        "train_score_mean": float(train_scores.mean()),
        "val_score_mean": float(val_scores.mean()),
        "score_baseline_median_mad": baseline,
        "per_recipe": per_recipe,
        "pooled_auc_all_recipes_vs_clean": float(pooled_auc),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_real_data(N_PARTS)
    X_train, M_train, X_val, M_val, meta_val, scaler = prepare_windows(df)
    n_features = len(PRIMARY_FEATURES)

    report: dict = {
        "n_train_windows": len(X_train), "n_val_windows": len(X_val),
        "features": PRIMARY_FEATURES, "n_parts": N_PARTS, "confidence_threshold": CONFIDENCE_THRESHOLD,
    }

    # ---- Dense-AE ----
    print("\n=== Dense-AE egitimi ===", flush=True)
    dense_model, dense_hist = train_dense_autoencoder(
        X_train, M_train, window=WINDOW, n_features=n_features, epochs=EPOCHS,
        batch_size=BATCH_SIZE, seed=SEED
    )
    torch.manual_seed(SEED + 1000)
    dense_untrained = DenseAutoencoder(WINDOW, n_features)
    dense_val_scores = _score_batched(lambda X, M: dense_scores(dense_model, X, M), X_val, M_val)
    dense_untrained_scores = _score_batched(lambda X, M: dense_scores(dense_untrained, X, M), X_val, M_val)
    dense_diag = magnitude_domination_check(dense_val_scores, dense_untrained_scores, X_val, M_val)
    print("  magnitude_domination_check:", dense_diag, flush=True)
    report["dense_ae"] = {
        "train_loss_history": dense_hist, "magnitude_domination_check": dense_diag,
        **evaluate_model(lambda X, M: dense_scores(dense_model, X, M), X_train, M_train, X_val, M_val, scaler),
    }
    print("  pooled AUC:", report["dense_ae"]["pooled_auc_all_recipes_vs_clean"], flush=True)

    # ---- LSTM-AE ----
    print("\n=== LSTM-AE egitimi ===", flush=True)
    lstm_ae_model, lstm_ae_hist = train_lstm_autoencoder(
        X_train, M_train, n_features=n_features, hidden_size=32, epochs=EPOCHS,
        batch_size=BATCH_SIZE, seed=SEED
    )
    torch.manual_seed(SEED + 1000)
    lstm_ae_untrained = LSTMAutoencoder(n_features, hidden_size=32)
    lstm_ae_val_scores = _score_batched(lambda X, M: lstm_ae_scores(lstm_ae_model, X, M), X_val, M_val)
    lstm_ae_untrained_scores = _score_batched(lambda X, M: lstm_ae_scores(lstm_ae_untrained, X, M), X_val, M_val)
    lstm_ae_diag = magnitude_domination_check(lstm_ae_val_scores, lstm_ae_untrained_scores, X_val, M_val)
    print("  magnitude_domination_check:", lstm_ae_diag, flush=True)
    report["lstm_ae"] = {
        "train_loss_history": lstm_ae_hist, "magnitude_domination_check": lstm_ae_diag,
        **evaluate_model(lambda X, M: lstm_ae_scores(lstm_ae_model, X, M), X_train, M_train, X_val, M_val, scaler),
    }
    print("  pooled AUC:", report["lstm_ae"]["pooled_auc_all_recipes_vs_clean"], flush=True)

    # ---- LSTM-forecaster ----
    print("\n=== LSTM-forecaster egitimi ===", flush=True)
    fc_model, fc_hist = train_lstm_forecaster(
        X_train, M_train, history_len=HISTORY_LEN, n_features=n_features, hidden_size=32,
        epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED
    )
    torch.manual_seed(SEED + 1000)
    fc_untrained = LSTMForecaster(n_features, horizon=WINDOW - HISTORY_LEN, hidden_size=32)

    def _fc_scores(model):
        return lambda X, M: forecast_residual_scores(model, X, M, history_len=HISTORY_LEN)

    fc_val_scores = _score_batched(_fc_scores(fc_model), X_val, M_val)
    fc_untrained_scores = _score_batched(_fc_scores(fc_untrained), X_val, M_val)
    fc_diag = magnitude_domination_check(fc_val_scores, fc_untrained_scores, X_val[:, HISTORY_LEN:], M_val[:, HISTORY_LEN:])
    print("  magnitude_domination_check:", fc_diag, flush=True)
    report["lstm_forecaster"] = {
        "train_loss_history": fc_hist, "magnitude_domination_check": fc_diag,
        **evaluate_model(_fc_scores(fc_model), X_train, M_train, X_val, M_val, scaler),
    }
    print("  pooled AUC:", report["lstm_forecaster"]["pooled_auc_all_recipes_vs_clean"], flush=True)

    report["scaler"] = scaler.to_dict()
    report["usad"] = "HARIC -- sayisal kararsizlik cozulmedi, bkz ADR-022/023"
    out_path = OUT_DIR / "baseline_training_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {out_path}", flush=True)


if __name__ == "__main__":
    main()
