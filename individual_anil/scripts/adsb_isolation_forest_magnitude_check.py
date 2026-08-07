"""isolation_forest_contextual_v1 -- ilk magnitude self-check (prereg'in ilk gate'i).

Ayni Step-5 manifestinin 'fit' rolunden bir alt-orneklemi kullanir (tam 237 parcanin
tamami degil -- bu bir KESIF tanisi, contextual_physics_v1'in tam-hacim/hash-zincirli
uretim kosusu degil; bu fark prereg'de acikca beyan edilmisti).

Olcer:
  - trained-IF-skoru vs ham-genlik (magnitude_only_score) Spearman rho
  - trained-IF-skoru vs "yapisiz" (kanal-bazi karistirilmis) IF skoru Spearman rho
FLAG/PASS iddiasi yapmaz -- yalniz olcer ve raporlar (prereg'in kendi kurali).

Kullanim:
    python scripts/adsb_isolation_forest_magnitude_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.diagnostics import magnitude_only_score  # noqa: E402
from adsb.features import build_feature_table  # noqa: E402
from adsb.models.isolation_forest_residual import (  # noqa: E402
    RESIDUAL_CHANNELS,
    fit_isolation_forest_residual,
    score_isolation_forest_residual,
)
from adsb.segmentation import segment_flights  # noqa: E402

MANIFEST_PATH = Path("artifacts/adsb/runs/20260713_step5_full_streaming_v1/run_manifest.json")
N_FIT_PARTS = 40  # kesif alt-orneklemi -- tam 237 parca degil
SEED = 0


def load_fit_sample(n_parts: int) -> pd.DataFrame:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    fit_paths = [r["path"] for r in manifest["inputs"] if r["role"] == "fit"][:n_parts]
    print(f"{len(fit_paths)} fit-rolu Silver parcasi okunuyor (237 parcanin alt-orneklemi)...", flush=True)
    df = pd.concat([pd.read_parquet(p) for p in fit_paths], ignore_index=True)
    print(f"  {len(df)} satir", flush=True)
    return df


def main() -> None:
    df = load_fit_sample(N_FIT_PARTS)
    seg = segment_flights(df, gap_s=1800.0)
    feat = build_feature_table(seg)

    flight_ids = np.array(feat["flight_id"].unique(), dtype=object)
    rng = np.random.default_rng(SEED)
    rng.shuffle(flight_ids)
    split = int(len(flight_ids) * 0.8)
    fit_ids, diag_ids = set(flight_ids[:split]), set(flight_ids[split:])

    fit_frame = feat[feat["flight_id"].isin(fit_ids)]
    diag_frame = feat[feat["flight_id"].isin(diag_ids)]
    print(f"  fit: {len(fit_frame)} satir ({len(fit_ids)} ucus), "
          f"diagnostic: {len(diag_frame)} satir ({len(diag_ids)} ucus)", flush=True)

    model, scaler, active_channels = fit_isolation_forest_residual(fit_frame)
    print(f"  aktif kanallar: {active_channels}", flush=True)

    trained_scores = score_isolation_forest_residual(model, scaler, diag_frame)
    valid = trained_scores.notna()
    trained_scores = trained_scores[valid]
    scaled_diag = scaler.transform(diag_frame)[valid]
    print(f"  skorlanabilir (tam-durumlu) tani satiri: {len(trained_scores)}", flush=True)

    X = scaled_diag.to_numpy(dtype=float)
    M = np.ones_like(X)
    mag_scores = magnitude_only_score(X[:, None, :], M[:, None, :])
    rho_vs_magnitude, _ = spearmanr(trained_scores, mag_scores)

    rng2 = np.random.default_rng(SEED + 1)
    shuffled_fit = fit_frame.copy()
    for c in active_channels:
        shuffled_fit[c] = rng2.permutation(shuffled_fit[c].to_numpy())
    shuffled_model, shuffled_scaler, _ = fit_isolation_forest_residual(shuffled_fit)
    shuffled_scores = score_isolation_forest_residual(shuffled_model, shuffled_scaler, diag_frame)[valid]
    rho_vs_shuffled, _ = spearmanr(trained_scores, shuffled_scores)

    report = {
        "n_fit_parts_of_237": N_FIT_PARTS,
        "n_fit_rows": int(len(fit_frame)), "n_fit_flights": len(fit_ids),
        "n_diagnostic_rows_scored": int(len(trained_scores)), "n_diagnostic_flights": len(diag_ids),
        "active_channels": list(active_channels),
        "excluded_channels": list(scaler.excluded_channels_),
        "rho_trained_vs_magnitude": float(rho_vs_magnitude),
        "rho_trained_vs_shuffled_channel_fit": float(rho_vs_shuffled),
        "note": "FLAG/PASS iddiasi yok -- yalniz olcum. contextual_physics_v1 fail sinirinin (rho>=0.8) yalniz referans icin.",
    }
    out = Path("artifacts/adsb/models/isolation_forest_magnitude_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"\nRapor: {out}", flush=True)


if __name__ == "__main__":
    main()
