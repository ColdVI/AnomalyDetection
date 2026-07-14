"""contextual_physics_v1 -- ilk natural-development dogal-yuk egrisi (veri rolu #3).

2026-03-01 (development gunu) verisini, EGITILMIS checkpoint ile skorlar, calibration
gunundeki (2026-02-28) conformal kalibratoru fit eder, ve HER kanal/mod icin ONCEDEN
BELIRLENMIS (sonuca bakilmadan secilmis) bir log-spaced alfa izgarasinda dogal alarm
yukunu (episode / scoreable ucus-saat) OLCER.

Bu bir arama/optimizasyon DEGIL -- sabit bir izgarada olcum. ADR-037'nin Pareto
bütçe noktalari icin gereken alfa, bu egriden INTERPOLASYONLA okunur, sonuca gore
YENIDEN aranmaz.

Bilinen basitlestirme (durustce beyan): east/north velocity residual, ADR-037/ADR-029'un
tasarladigi ORTAK 2-eksenli Page-CUSUM yerine, mevcut adsb/contextual_decision.py'nin
TEK-kanalli accumulation modu ile AYRI AYRI olculuyor -- ortak-eksen CUSUM'u buraya
kablolamak ayri bir is, burada YAPILMADI.

Kullanim:
    python scripts/adsb_contextual_physics_v1_development_burden.py
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
from adsb.contextual_decision import ChannelAlertBudget, DetectorProfile, apply_detector_profile  # noqa: E402
from adsb.contextual_scaling import StrictNaturalRobustScaler, StrictScalingConfig  # noqa: E402
from adsb.contextual_windowing import build_contextual_forecast_windows  # noqa: E402
from adsb.evaluation import EpisodeContract, natural_alert_burden  # noqa: E402
from adsb.features import build_feature_table  # noqa: E402
from adsb.models.contextual_residual_forecaster import (  # noqa: E402
    ContextualForecasterConfig,
    ContextualResidualForecaster,
    contextual_channel_scores,
)
from adsb.segmentation import segment_flights  # noqa: E402

RUN_DIR = Path("artifacts/adsb/runs/20260714_contextual_physics_v1_train_v1")
TRAIN_CONFIG_PATH = Path("configs/adsb_contextual_physics_v1_train.json")
BUDGET_CONFIG_PATH = Path("configs/adsb_contextual_physics_v1_alarm_budget.json")
STEP5_MANIFEST = Path("artifacts/adsb/runs/20260713_step5_full_streaming_v1/run_manifest.json")
CALIBRATION_DAY = "2026-02-28"
DEVELOPMENT_DAY = "2026-03-01"
SEGMENT_GAP_S = 1800.0
N_PARTS_CALIBRATION = 60   # ayni ADR-039'daki kesif olcegi
N_PARTS_DEVELOPMENT = 1    # kesif olcegi -- 216/216 degil. apply_detector_profile'in
                            # satir-basina .loc atamali Python dongusu, sweep (8 alfa x 5
                            # profil) icin cok yavas cikti (2.1M satirda 15dk+ surdu, tek
                            # kanal bile bitmedi) -- bu bilinen bir performans sinirlamasi,
                            # burada COZULMEDI, yalniz olcek kuculterek etrafindan gecildi.
MIN_GROUP_SIZE = 1000
HISTORY_ROWS = 12
PERSISTENCE_WINDOW_S = 30.0  # ADR-037

# Sonuca bakilmadan secilmis, log-spaced alfa izgarasi.
ALPHA_GRID = tuple(np.geomspace(1e-5, 0.5, 4).tolist())

OUT_DIR = Path("artifacts/adsb/runs/20260714_contextual_physics_v1_development_burden_v1")


def _context_config() -> CausalContextConfig:
    values = json.loads(TRAIN_CONFIG_PATH.read_text(encoding="utf-8"))["context"]
    return CausalContextConfig(
        phase_history_rows=int(values["phase_history_rows"]),
        level_rate_threshold_mps=float(values["level_rate_threshold_mps"]),
        cadence_edges_s=tuple(map(float, values["cadence_edges_s"])),
        max_gap_s=float(values["max_gap_s"]),
    )


def _load_checkpoint() -> tuple[ContextualResidualForecaster, StrictNaturalRobustScaler, tuple[str, ...]]:
    derived = json.loads((RUN_DIR / "derived_training_config.json").read_text(encoding="utf-8"))
    model = ContextualResidualForecaster(ContextualForecasterConfig(**derived["model_config"]))
    model.load_state_dict(torch.load(RUN_DIR / "model_state.pt", map_location="cpu", weights_only=True))
    model.eval()
    scaler_dict = derived["scaler"]
    scaler = StrictNaturalRobustScaler(StrictScalingConfig(clip=float(scaler_dict["clip"])))
    scaler.calibration_ = scaler_dict["calibration"]
    scaler.excluded_channels_ = tuple(scaler_dict["excluded_channels"])
    return model, scaler, tuple(derived["target_channels"])


def _score_batched(model, X, X_mask, y, y_mask, batch_size: int = 20_000):
    if len(X) <= batch_size:
        return contextual_channel_scores(model, X, X_mask, y, y_mask)
    parts = [
        contextual_channel_scores(model, X[i:i + batch_size], X_mask[i:i + batch_size],
                                   y[i:i + batch_size], y_mask[i:i + batch_size])
        for i in range(0, len(X), batch_size)
    ]
    scores = np.concatenate([p[0] for p in parts])
    loc = np.concatenate([p[1] for p in parts])
    scale = np.concatenate([p[2] for p in parts])
    return scores, loc, scale


def _day_flight_ids(role: str, day: str) -> set[str]:
    manifest = json.loads(STEP5_MANIFEST.read_text(encoding="utf-8"))
    raw = manifest["split_contract"]["splits"][role]["flight_ids"]
    ids = set(raw)
    sample = next(iter(ids))
    if not sample.startswith(f"{day}:"):
        raise RuntimeError(f"{role} flight_ids do not start with expected day prefix {day}")
    return ids


def _role_input_paths(role: str, n_parts: int) -> list[Path]:
    manifest = json.loads(STEP5_MANIFEST.read_text(encoding="utf-8"))
    records = [r for r in manifest["inputs"] if r["role"] == role]
    return [Path(r["path"]) for r in records[:n_parts]]


def _load_day_features(paths: list[Path], day: str, keep_flights: set[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        raw = pd.read_parquet(path)
        segmented = segment_flights(raw, gap_s=SEGMENT_GAP_S)
        segmented["flight_id"] = segmented["flight_id"].map(lambda v, d=day: f"{d}:{v}")
        kept = segmented.loc[segmented["flight_id"].isin(keep_flights)]
        if not kept.empty:
            frames.append(build_feature_table(kept))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _score_day(features: pd.DataFrame, model, scaler, target_channels: tuple[str, ...]):
    """Doner: (scores, batch) -- batch.meta/batch.y_mask ayrica yeniden hesaplanmaz."""
    scaled = features.copy()
    transformed = scaler.transform(features)
    for channel in scaler.active_channels:
        scaled[channel] = transformed[channel]
    batch = build_contextual_forecast_windows(
        scaled, signal_columns=scaler.active_channels, target_channels=target_channels,
        history_rows=HISTORY_ROWS, context_config=_context_config(),
    )
    if len(batch.X) == 0:
        return np.zeros((0, len(target_channels))), batch
    scores, _, _ = _score_batched(model, batch.X, batch.X_mask, batch.y, batch.y_mask)
    return scores, batch


def _with_exposure_bounds(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta["t_start"] = meta.groupby("flight_id")["target_timestamp_utc"].shift(1)
    meta["t_start"] = meta["t_start"].fillna(meta["target_timestamp_utc"])
    meta["t_end"] = meta["target_timestamp_utc"]
    return meta


def _long_calibration_table(scores: np.ndarray, y_mask: np.ndarray, meta: pd.DataFrame, target_channels: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for i, channel in enumerate(target_channels):
        valid = y_mask[:, i] > 0
        rows.append(pd.DataFrame({
            "channel": channel,
            "context_phase": meta.loc[valid, "context_phase"].to_numpy(),
            "context_cadence": meta.loc[valid, "context_cadence"].to_numpy(),
            "score": scores[valid, i],
        }))
    return pd.concat(rows, ignore_index=True)


PROFILES: dict[str, list[DetectorProfile]] = {
    "vertical_rate_residual": [
        DetectorProfile("vertical_rate_spike", "vertical_rate_residual", "instant", max_gap_s=60.0),
        DetectorProfile("vertical_rate_freeze", "vertical_rate_residual", "persistence", max_gap_s=60.0, persistence_s=PERSISTENCE_WINDOW_S),
    ],
    "speed_residual": [
        DetectorProfile("speed_spike", "speed_residual", "instant", max_gap_s=60.0),
        DetectorProfile("speed_bias", "speed_residual", "persistence", max_gap_s=60.0, persistence_s=PERSISTENCE_WINDOW_S),
    ],
    "heading_residual": [
        DetectorProfile("heading_inconsistency", "heading_residual", "persistence", max_gap_s=60.0, persistence_s=PERSISTENCE_WINDOW_S),
    ],
}


def main() -> None:
    print("Checkpoint yukleniyor...", flush=True)
    model, scaler, target_channels = _load_checkpoint()

    print("Calibration gunu (2026-02-28) taraniyor...", flush=True)
    cal_flights = _day_flight_ids("calibration", CALIBRATION_DAY)
    cal_paths = _role_input_paths("fit", N_PARTS_CALIBRATION)
    cal_features = _load_day_features(cal_paths, CALIBRATION_DAY, cal_flights)
    cal_scores, cal_batch = _score_day(cal_features, model, scaler, target_channels)
    cal_long = _long_calibration_table(cal_scores, cal_batch.y_mask, cal_batch.meta, target_channels)
    print(f"  calibration: {len(cal_long)} (channel,score) satiri", flush=True)

    calibrator = HierarchicalConformalCalibrator(
        ConditionalCalibrationConfig(min_group_size=MIN_GROUP_SIZE)
    ).fit(cal_long, data_role=NATURAL_CALIBRATION_ROLE, contains_synthetic=False)

    print("Development gunu (2026-03-01) taraniyor (parca-parca, bellek icin)...", flush=True)
    dev_flights = _day_flight_ids("development", DEVELOPMENT_DAY)
    dev_paths = _role_input_paths("development", N_PARTS_DEVELOPMENT)
    # ADR-039'daki 60-parcalik calibration yuku (3.3M pencere) sorunsuzdu ama development
    # gununun parca-basina yogunlugu farkli cikti -- 15 parcada TEK SEFERDE pencereleme
    # 4.56GiB'lik tek array istedi ve coktu. Dosya-basina (kucuk grup) isleyip skorlari/
    # meta'yi biriktirmek, TUM development'i tek array'de tutmaktan kacinir.
    dev_score_parts: list[np.ndarray] = []
    dev_meta_parts: list[pd.DataFrame] = []
    dev_ymask_parts: list[np.ndarray] = []
    for i, path in enumerate(dev_paths):
        chunk_features = _load_day_features([path], DEVELOPMENT_DAY, dev_flights)
        if chunk_features.empty:
            continue
        chunk_scores, chunk_batch = _score_day(chunk_features, model, scaler, target_channels)
        if len(chunk_scores) == 0:
            continue
        dev_score_parts.append(chunk_scores)
        dev_meta_parts.append(chunk_batch.meta)
        dev_ymask_parts.append(chunk_batch.y_mask)
        print(f"  parca {i + 1}/{len(dev_paths)}: {len(chunk_scores)} pencere", flush=True)
    dev_scores = np.concatenate(dev_score_parts)
    dev_ymask = np.concatenate(dev_ymask_parts)
    dev_meta = _with_exposure_bounds(pd.concat(dev_meta_parts, ignore_index=True))
    print(f"  development toplam: {len(dev_meta)} skorlanabilir pencere", flush=True)

    budget = json.loads(BUDGET_CONFIG_PATH.read_text(encoding="utf-8"))
    pareto_grid = budget["budget_grid_episodes_per_100_scoreable_flight_hours"]
    channel_shares = budget["budget_shares_of_total"]

    report: dict = {
        "calibration_day": CALIBRATION_DAY, "development_day": DEVELOPMENT_DAY,
        "n_calibration_parts_used": len(cal_paths), "n_development_parts_used": len(dev_paths),
        "alpha_grid": list(ALPHA_GRID),
        "pareto_grid_episodes_per_100h": pareto_grid,
        "known_simplification": "east/north velocity residual accumulation measured PER-CHANNEL, "
                                 "not via the joint 2-axis Page-CUSUM specified in ADR-037/ADR-029.",
        "channels": {},
    }

    for channel, profiles in PROFILES.items():
        idx = target_channels.index(channel)
        channel_scored = pd.DataFrame({
            "flight_id": dev_meta["flight_id"], "timestamp_utc": dev_meta["target_timestamp_utc"],
            "t_start": dev_meta["t_start"], "t_end": dev_meta["t_end"],
            "channel": channel, "context_phase": dev_meta["context_phase"],
            "context_cadence": dev_meta["context_cadence"], "score": dev_scores[:, idx],
        })
        valid = dev_ymask[:, idx] > 0
        channel_scored = channel_scored.loc[valid].reset_index(drop=True)
        conformal = calibrator.transform(channel_scored)
        channel_scored["conformal_p_value"] = conformal["conformal_p_value"].to_numpy()

        report["channels"][channel] = {"n_scored_rows": int(len(channel_scored)), "profiles": {}}
        for profile in profiles:
            curve = []
            for alpha in ALPHA_GRID:
                decided = apply_detector_profile(
                    channel_scored, profile=profile,
                    budget=ChannelAlertBudget(total_alpha=alpha, channel_alpha={channel: alpha}),
                )
                burden = natural_alert_burden(
                    channel_scored, decided["alarm"].to_numpy(),
                    contract=EpisodeContract(merge_gap_s=60.0, emission_time_col="t_end"),
                )
                curve.append({"alpha": alpha, **burden})
            report["channels"][channel]["profiles"][profile.anomaly_type] = curve
            # Sadece konsol-gostergesi icin: Pareto noktasi V=1.0 episode/100h'deki kanal
            # payinin saat-basi hedefine (share * V / 100) en yakin izgara noktasi.
            target_per_hour = channel_shares.get(channel, 0.0) * 1.0 / 100.0
            best = min(curve, key=lambda r: abs((r["alert_episodes_per_scoreable_flight_hour"] or 1e9) - target_per_hour))
            print(f"  {channel}/{profile.anomaly_type}: [hedef~{target_per_hour:.5f}/saat @ Pareto=1.0] "
                  f"en yakin alpha={best['alpha']:.2e} -> "
                  f"{best['alert_episodes_per_scoreable_flight_hour']:.4f} episode/saat "
                  f"({best['alerted_flight_fraction']:.4f} ucus-oran)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "development_burden_curves.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRapor: {OUT_DIR / 'development_burden_curves.json'}", flush=True)


if __name__ == "__main__":
    main()
