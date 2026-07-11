"""ADSB-1: dort mimariyi (Dense-AE, LSTM-AE, USAD, LSTM-forecaster) gercek
tar-1 (2026-02-28) verisiyle egitir, ZORUNLU magnitude_domination_check calistirir,
ve birkac PHYSICS_BREAK_RECIPES senaryosuyla ilk sentetik-dogrulama sinyalini alir.

Egitim SADECE normal (enjeksiyonsuz) ucus pencereleriyle yapilir -- etiket
kullanilmaz (adsb.lol'da zaten yok). Enjekte edilmis pencereler hicbir zaman
egitime girmez, yalniz ayri bir val-corrupted degerlendirmesinde kullanilir.

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.diagnostics import magnitude_domination_check  # noqa: E402
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
from adsb.models.usad import USAD, train_usad, usad_scores  # noqa: E402
from adsb.segmentation import segment_flights  # noqa: E402
from adsb.synthetic import PHYSICS_BREAK_RECIPES  # noqa: E402
from adsb.windowing import build_windows  # noqa: E402

SILVER_DIR = Path("data/objectstore/silver/adsblol_historical")
OUT_DIR = Path("artifacts/adsb/models")
N_PARTS = 10  # ilk uctan uca dogrulama turu icin kucuk tutuluyor; pipeline dogrulaninca buyutulur
WINDOW = 12
STRIDE = 6
HISTORY_LEN = 8  # LSTM-forecaster: ilk 8 adim gecmis, son 4 adim (WINDOW-HISTORY_LEN) hedef
MAX_GAP_S = 60.0  # pencere-ici max bosluk (segmentasyonun kendi 1800s'inden cok daha siki)
EPOCHS = 15
SEED = 0


def load_real_data(n_parts: int) -> pd.DataFrame:
    files = sorted(SILVER_DIR.glob("*.parquet"))[:n_parts]
    print(f"{len(files)} Silver parcasi okunuyor...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {len(df)} satir, {df.source_id.nunique()} ucak")
    return df


def prepare_windows(df: pd.DataFrame):
    seg = segment_flights(df, gap_s=1800.0)  # HAM (residualsiz) -- enjeksiyon icin saklanir
    feat = build_feature_table(seg)

    flight_ids = feat["flight_id"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(flight_ids)
    split = int(len(flight_ids) * 0.8)
    train_ids, val_ids = set(flight_ids[:split]), set(flight_ids[split:])

    train_df = feat[feat["flight_id"].isin(train_ids)]
    val_df = feat[feat["flight_id"].isin(val_ids)]
    val_raw_df = seg[seg["flight_id"].isin(val_ids)]  # residualsiz -- enjeksiyon buradan baslar

    X_train, M_train, _ = build_windows(train_df, PRIMARY_FEATURES, window=WINDOW, stride=STRIDE, max_gap_s=MAX_GAP_S)
    X_val, M_val, meta_val = build_windows(val_df, PRIMARY_FEATURES, window=WINDOW, stride=STRIDE, max_gap_s=MAX_GAP_S)
    print(f"  train pencere: {len(X_train)}, val pencere: {len(X_val)}")

    # KIRPILI olcekleme -- SEAD dersi (ADR-016): kirpilmamis olcekleme genlik-
    # baskinligina yol aciyordu. Fit SADECE train'den, val'e aynen uygulanir.
    scaler = ClippedRobustScaler(clip=5.0).fit(X_train, M_train)
    X_train_scaled = scaler.transform(X_train, M_train)
    X_val_scaled = scaler.transform(X_val, M_val)
    print(f"  scaler fit edildi (clip=5.0), median ornek: {scaler.median_[:3]}")

    return X_train_scaled, M_train, X_val_scaled, M_val, meta_val, val_raw_df, scaler


def run_synthetic_check(score_fn, val_raw_df: pd.DataFrame, scaler: ClippedRobustScaler) -> dict:
    """Birkac val ucusuna (HAM, residualsiz) PHYSICS_BREAK_RECIPES uygular, enjeksiyon
    SONRASI residual'lari yeniden hesaplayip AYNI (train'de fit edilmis) scaler ile
    olcekleyip temiz/bozuk skor karsilastirir."""
    sample_flights = val_raw_df["flight_id"].drop_duplicates().head(20).tolist()
    results = {}
    for recipe_name, (fn, kwargs) in PHYSICS_BREAK_RECIPES.items():
        clean_scores, corrupt_scores = [], []
        for fid in sample_flights:
            flight_raw = val_raw_df[val_raw_df["flight_id"] == fid].sort_values("timestamp_utc").reset_index(drop=True)
            if len(flight_raw) < WINDOW + 2:
                continue
            flight_raw = flight_raw.assign(label=None)
            try:
                bozuk_raw = fn(flight_raw, onset_frac=0.5, **kwargs)
            except Exception:
                continue

            clean_feat = build_feature_table(flight_raw)
            bozuk_feat = build_feature_table(bozuk_raw)

            Xc, Mc, _ = build_windows(clean_feat, PRIMARY_FEATURES, window=WINDOW, stride=WINDOW, max_gap_s=MAX_GAP_S)
            Xb, Mb, _ = build_windows(bozuk_feat, PRIMARY_FEATURES, window=WINDOW, stride=WINDOW, max_gap_s=MAX_GAP_S)
            if len(Xc) == 0 or len(Xb) == 0:
                continue
            Xc, Xb = scaler.transform(Xc, Mc), scaler.transform(Xb, Mb)
            clean_scores.extend(score_fn(Xc, Mc).tolist())
            corrupt_scores.extend(score_fn(Xb, Mb).tolist())

        if clean_scores and corrupt_scores:
            results[recipe_name] = {
                "n_clean_windows": len(clean_scores),
                "n_corrupt_windows": len(corrupt_scores),
                "mean_clean_score": float(np.mean(clean_scores)),
                "mean_corrupt_score": float(np.mean(corrupt_scores)),
                "corrupt_higher": bool(np.mean(corrupt_scores) > np.mean(clean_scores)),
            }
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_real_data(N_PARTS)
    X_train, M_train, X_val, M_val, meta_val, val_raw_df, scaler = prepare_windows(df)
    n_features = len(PRIMARY_FEATURES)

    report: dict = {"n_train_windows": len(X_train), "n_val_windows": len(X_val), "features": PRIMARY_FEATURES}

    # ---- Dense-AE ----
    print("\n=== Dense-AE egitimi ===")
    dense_model, dense_hist = train_dense_autoencoder(
        X_train, M_train, window=WINDOW, n_features=n_features, epochs=EPOCHS, seed=SEED
    )
    torch.manual_seed(SEED + 1000)
    dense_untrained = DenseAutoencoder(WINDOW, n_features)
    dense_val_scores = dense_scores(dense_model, X_val, M_val)
    dense_untrained_scores = dense_scores(dense_untrained, X_val, M_val)
    dense_diag = magnitude_domination_check(dense_val_scores, dense_untrained_scores, X_val, M_val)
    print("  magnitude_domination_check:", dense_diag)
    report["dense_ae"] = {
        "train_loss_history": dense_hist,
        "val_score_mean": float(dense_val_scores.mean()),
        "magnitude_domination_check": dense_diag,
        "synthetic_check": run_synthetic_check(lambda X, M: dense_scores(dense_model, X, M), val_raw_df, scaler),
    }

    # ---- LSTM-AE ----
    print("\n=== LSTM-AE egitimi ===")
    lstm_ae_model, lstm_ae_hist = train_lstm_autoencoder(
        X_train, M_train, n_features=n_features, hidden_size=32, epochs=EPOCHS, seed=SEED
    )
    torch.manual_seed(SEED + 1000)
    lstm_ae_untrained = LSTMAutoencoder(n_features, hidden_size=32)
    lstm_ae_val_scores = lstm_ae_scores(lstm_ae_model, X_val, M_val)
    lstm_ae_untrained_scores = lstm_ae_scores(lstm_ae_untrained, X_val, M_val)
    lstm_ae_diag = magnitude_domination_check(lstm_ae_val_scores, lstm_ae_untrained_scores, X_val, M_val)
    print("  magnitude_domination_check:", lstm_ae_diag)
    report["lstm_ae"] = {
        "train_loss_history": lstm_ae_hist,
        "val_score_mean": float(lstm_ae_val_scores.mean()),
        "magnitude_domination_check": lstm_ae_diag,
        "synthetic_check": run_synthetic_check(lambda X, M: lstm_ae_scores(lstm_ae_model, X, M), val_raw_df, scaler),
    }

    # ---- USAD ----
    print("\n=== USAD egitimi ===")
    usad_model, usad_hist = train_usad(
        X_train, M_train, window=WINDOW, n_features=n_features, epochs=EPOCHS, seed=SEED
    )
    torch.manual_seed(SEED + 1000)
    usad_untrained = USAD(WINDOW, n_features)
    usad_val_scores = usad_scores(usad_model, X_val, M_val)
    usad_untrained_scores = usad_scores(usad_untrained, X_val, M_val)
    usad_diag = magnitude_domination_check(usad_val_scores, usad_untrained_scores, X_val, M_val)
    print("  magnitude_domination_check:", usad_diag)
    report["usad"] = {
        "train_loss_history": usad_hist,
        "val_score_mean": float(usad_val_scores.mean()),
        "magnitude_domination_check": usad_diag,
        "synthetic_check": run_synthetic_check(lambda X, M: usad_scores(usad_model, X, M), val_raw_df, scaler),
    }

    # ---- LSTM-forecaster ----
    print("\n=== LSTM-forecaster egitimi ===")
    fc_model, fc_hist = train_lstm_forecaster(
        X_train, M_train, history_len=HISTORY_LEN, n_features=n_features, hidden_size=32, epochs=EPOCHS, seed=SEED
    )
    torch.manual_seed(SEED + 1000)
    fc_untrained = LSTMForecaster(n_features, horizon=WINDOW - HISTORY_LEN, hidden_size=32)
    fc_val_scores = forecast_residual_scores(fc_model, X_val, M_val, history_len=HISTORY_LEN)

    def _fc_untrained_scores(X, M):
        return forecast_residual_scores(fc_untrained, X, M, history_len=HISTORY_LEN)

    fc_untrained_scores = _fc_untrained_scores(X_val, M_val)
    fc_diag = magnitude_domination_check(fc_val_scores, fc_untrained_scores, X_val[:, HISTORY_LEN:], M_val[:, HISTORY_LEN:])
    print("  magnitude_domination_check:", fc_diag)
    report["lstm_forecaster"] = {
        "train_loss_history": fc_hist,
        "val_score_mean": float(fc_val_scores.mean()),
        "magnitude_domination_check": fc_diag,
        "synthetic_check": run_synthetic_check(
            lambda X, M: forecast_residual_scores(fc_model, X, M, history_len=HISTORY_LEN),
            val_raw_df, scaler,
        ),
    }

    report["scaler"] = scaler.to_dict()
    out_path = OUT_DIR / "baseline_training_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {out_path}")


if __name__ == "__main__":
    main()
