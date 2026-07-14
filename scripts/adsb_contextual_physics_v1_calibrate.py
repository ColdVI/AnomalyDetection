"""contextual_physics_v1 -- ilk gercek natural-calibration turu (veri rolu #2).

Egitilmis checkpoint'i (artifacts/adsb/runs/20260714_contextual_physics_v1_train_v1/)
yukler, Step-5 split_contract'inin 'calibration' rolundeki (37.208 ucus, fit ile AYNI
gunden -- 2026-02-28 -- ama fit'ten TAMAMEN AYRIK bir alt-kume) satirlarini skorlar ve
adsb/conditional_calibration.py::HierarchicalConformalCalibrator'i fit eder.

Bu script alarm URETMEZ -- yalniz conformal tail'i kurar ve saglik/kapsam raporu verir.
Gercek alarm/burden olcumu (ADR-037'nin Pareto izgarasi + kanal paylariyla) sonraki,
development-rolu kosusudur.

Kullanim:
    python scripts/adsb_contextual_physics_v1_calibrate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.conditional_calibration import (  # noqa: E402
    ConditionalCalibrationConfig,
    HierarchicalConformalCalibrator,
    NATURAL_CALIBRATION_ROLE,
)
from adsb.context import CausalContextConfig  # noqa: E402
from adsb.contextual_scaling import StrictNaturalRobustScaler, StrictScalingConfig  # noqa: E402
from adsb.contextual_windowing import build_contextual_forecast_windows  # noqa: E402
from adsb.features import build_feature_table  # noqa: E402
from adsb.models.contextual_residual_forecaster import (  # noqa: E402
    ContextualForecasterConfig,
    ContextualResidualForecaster,
    contextual_channel_scores,
)
from adsb.segmentation import segment_flights  # noqa: E402

RUN_DIR = Path("artifacts/adsb/runs/20260714_contextual_physics_v1_train_v1")
TRAIN_CONFIG_PATH = Path("configs/adsb_contextual_physics_v1_train.json")
STEP5_MANIFEST = Path("artifacts/adsb/runs/20260713_step5_full_streaming_v1/run_manifest.json")
FIT_DAY = "2026-02-28"
SEGMENT_GAP_S = 1800.0
N_PARTS_SUBSET = 60  # ilk gercek tur icin 237'nin alt-orneklemi -- kesif olcegi, tam-hacim degil
MIN_GROUP_SIZE = 1000  # ADR-037/Q2: hedeflenen en kucuk p ~0.001'in 10x'i civari
OUT_DIR = Path("artifacts/adsb/runs/20260714_contextual_physics_v1_calibration_v1")


def _load_checkpoint() -> tuple[ContextualResidualForecaster, StrictNaturalRobustScaler, tuple[str, ...]]:
    derived = json.loads((RUN_DIR / "derived_training_config.json").read_text(encoding="utf-8"))
    model_config = ContextualForecasterConfig(**derived["model_config"])
    model = ContextualResidualForecaster(model_config)
    state = torch.load(RUN_DIR / "model_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    scaler_dict = derived["scaler"]
    scaler = StrictNaturalRobustScaler(StrictScalingConfig(clip=float(scaler_dict["clip"])))
    scaler.calibration_ = scaler_dict["calibration"]
    scaler.excluded_channels_ = tuple(scaler_dict["excluded_channels"])
    return model, scaler, tuple(derived["target_channels"])


def _context_config() -> CausalContextConfig:
    values = json.loads(TRAIN_CONFIG_PATH.read_text(encoding="utf-8"))["context"]
    return CausalContextConfig(
        phase_history_rows=int(values["phase_history_rows"]),
        level_rate_threshold_mps=float(values["level_rate_threshold_mps"]),
        cadence_edges_s=tuple(map(float, values["cadence_edges_s"])),
        max_gap_s=float(values["max_gap_s"]),
    )


def _calibration_flight_ids() -> set[str]:
    # Manifest flight_ids already carry the "YYYY-MM-DD:" prefix (verified against a
    # sample: "2026-02-28:000006_001") -- do NOT re-prefix, that silently produces a
    # zero-match set since segment_flights' raw ids get prefixed exactly once below.
    manifest = json.loads(STEP5_MANIFEST.read_text(encoding="utf-8"))
    return set(manifest["split_contract"]["splits"]["calibration"]["flight_ids"])


def _fit_input_paths(n_parts: int) -> list[Path]:
    manifest = json.loads(STEP5_MANIFEST.read_text(encoding="utf-8"))
    fit_records = [r for r in manifest["inputs"] if r["role"] == "fit"]
    return [Path(r["path"]) for r in fit_records[:n_parts]]


def _load_calibration_features(paths: list[Path], calibration_flights: set[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        raw = pd.read_parquet(path)
        segmented = segment_flights(raw, gap_s=SEGMENT_GAP_S)
        segmented["flight_id"] = segmented["flight_id"].map(lambda v: f"{FIT_DAY}:{v}")
        kept = segmented.loc[segmented["flight_id"].isin(calibration_flights)]
        if not kept.empty:
            frames.append(build_feature_table(kept))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _score_batched(model, X, X_mask, y, y_mask, batch_size: int = 20_000):
    """contextual_channel_scores'u tek forward'ta 3M+ pencere skorlamaya zorlamamak icin
    parcali cagirir -- ayni sorun bu oturumda daha once scripts/adsb_train_baseline_
    models.py'de OOM'a yol acmisti (~20.8GB), ayni cozum burada da gerekli."""
    if len(X) <= batch_size:
        return contextual_channel_scores(model, X, X_mask, y, y_mask)
    score_parts, loc_parts, scale_parts = [], [], []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        s, l, sc = contextual_channel_scores(model, X[start:end], X_mask[start:end], y[start:end], y_mask[start:end])
        score_parts.append(s)
        loc_parts.append(l)
        scale_parts.append(sc)
    return np.concatenate(score_parts), np.concatenate(loc_parts), np.concatenate(scale_parts)


def main() -> None:
    print("Checkpoint yukleniyor...", flush=True)
    model, scaler, target_channels = _load_checkpoint()
    print(f"  aktif kanallar: {target_channels}", flush=True)

    calibration_flights = _calibration_flight_ids()
    print(f"calibration rolunde {len(calibration_flights)} ucus (split_contract)", flush=True)

    paths = _fit_input_paths(N_PARTS_SUBSET)
    print(f"{len(paths)}/237 fit-gunu parcasi taraniyor (calibration-rolu ucuslar icin alt-orneklem)...", flush=True)
    features = _load_calibration_features(paths, calibration_flights)
    if features.empty:
        raise RuntimeError("Bu alt-orneklemde hic calibration-rolu ucus bulunamadi")
    n_calibration_flights_found = features["flight_id"].nunique()
    print(f"  {len(features)} satir, {n_calibration_flights_found} calibration-rolu ucus bulundu", flush=True)

    scaled = features.copy()
    transformed = scaler.transform(features)
    for channel in scaler.active_channels:
        scaled[channel] = transformed[channel]

    batch = build_contextual_forecast_windows(
        scaled, signal_columns=scaler.active_channels, target_channels=target_channels,
        history_rows=12, context_config=_context_config(),
    )
    print(f"  {len(batch.X)} skorlanabilir pencere", flush=True)
    if len(batch.X) == 0:
        raise RuntimeError("Sifir pencere -- calibration verisi bu alt-orneklemde yetersiz")

    scores, _, _ = _score_batched(model, batch.X, batch.X_mask, batch.y, batch.y_mask)

    long_rows = []
    for i, channel in enumerate(target_channels):
        valid = batch.y_mask[:, i] > 0
        long_rows.append(pd.DataFrame({
            "channel": channel,
            "context_phase": batch.meta.loc[valid, "context_phase"].to_numpy(),
            "context_cadence": batch.meta.loc[valid, "context_cadence"].to_numpy(),
            "score": scores[valid, i],
        }))
    long_frame = pd.concat(long_rows, ignore_index=True)
    print(f"  uzun-format calibration tablosu: {len(long_frame)} satir", flush=True)

    calibrator = HierarchicalConformalCalibrator(
        ConditionalCalibrationConfig(min_group_size=MIN_GROUP_SIZE)
    ).fit(long_frame, data_role=NATURAL_CALIBRATION_ROLE, contains_synthetic=False)

    coverage_report: dict[str, dict] = {}
    for channel in target_channels:
        channel_rows = long_frame[long_frame["channel"] == channel]
        level_counts = channel_rows.groupby(["context_phase", "context_cadence"]).size()
        n_direct_groups_with_support = int((level_counts >= MIN_GROUP_SIZE).sum())
        n_total_groups = int(len(level_counts))
        coverage_report[channel] = {
            "n_scored_rows": int(len(channel_rows)),
            "n_phase_cadence_groups_seen": n_total_groups,
            "n_phase_cadence_groups_with_direct_support": n_direct_groups_with_support,
            "channel_level_fallback_available": len(channel_rows) >= MIN_GROUP_SIZE,
        }

    report = {
        "data_role": NATURAL_CALIBRATION_ROLE,
        "n_fit_day_parts_used": len(paths),
        "n_fit_day_parts_total": 237,
        "n_calibration_flights_in_split_contract": len(calibration_flights),
        "n_calibration_flights_found_in_subset": int(n_calibration_flights_found),
        "n_scored_windows": int(len(batch.X)),
        "min_group_size": MIN_GROUP_SIZE,
        "coverage_by_channel": coverage_report,
        "note": "Bu bir kesif alt-orneklemidir (60/237 parca) -- tam-hacim, hash-zincirli uretim "
                "kalibrasyonu degil. Henuz hicbir alarm/burden olcumu yapilmadi.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "calibration_coverage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"\nRapor: {OUT_DIR / 'calibration_coverage_report.json'}", flush=True)


if __name__ == "__main__":
    main()
