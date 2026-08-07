"""contextual_physics_v1 -- veri rolu #5: truth-v2 (gercek/enjekte olay yakalama).

Bu, projede contextual_physics_v1 icin ILK GERCEK RECALL olcumudur. Su ana kadarki
her ADR (037-041) yalniz "normal veride ne kadar yanlis alarm veriyor" sorusuna
cevap veriyordu -- burada ilk kez "gercek/enjekte bir anomaliyi yakaliyor mu"
sorusu soruluyor.

Kullanilan esik/alfa/h degerleri HICBIR SEKILDE burada secilmiyor -- ADR-041'de
zaten dondurulmus (development_burden_curves.json / cusum_burden_report.json),
Pareto butce noktalarina en yakin degerler AYNEN uygulaniyor. Sentetik veri
(data/objectstore/synthetic/adsb_v2_20260713_01) YALNIZ degerlendirmede
kullaniliyor -- fit/calibration/scaling'e hicbir sekilde girmiyor (prereg kurali).

Kapsam: 5 recipe'den 4'u (vertical_rate_frozen, ground_speed_biased, track_frozen,
position_ramp_stealthy) contextual_physics_v1'in 5 fizik kanaliyla eslesiyor.
altitude_dropout kapsam disi -- o S2 veri-kalitesi katmaninin isi, NN residual
kanallarinin degil (prereg'in kendi ayrimi).

Kullanim:
    python scripts/adsb_contextual_physics_v1_truth_v2_eval.py
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
from adsb.contextual_decision import ChannelAlertBudget, DetectorProfile  # noqa: E402
from adsb.contextual_decision_fast import apply_detector_profile_fast  # noqa: E402
from adsb.contextual_scaling import StrictNaturalRobustScaler, StrictScalingConfig  # noqa: E402
from adsb.contextual_windowing import build_contextual_forecast_windows  # noqa: E402
from adsb.cusum import CusumConfig, VectorPageCUSUM  # noqa: E402
from adsb.evaluation import (  # noqa: E402
    EpisodeContract,
    active_interval_coverage,
    event_detection_metrics,
    event_observability_denominators,
    natural_alert_burden,
    truth_event_table,
)
from adsb.features import VECTOR_RESIDUAL_FEATURES, build_feature_table  # noqa: E402
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
LSTM_DEV_REPORT = Path("artifacts/adsb/runs/20260714_contextual_physics_v1_development_burden_v2/development_burden_curves.json")
CUSUM_DEV_REPORT = Path("artifacts/adsb/runs/20260714_contextual_physics_v1_cusum_burden_v1/cusum_burden_report.json")
CORPUS_DIR = Path("data/objectstore/synthetic/adsb_v2_20260713_01")

CALIBRATION_DAY = "2026-02-28"
FIT_DAY = "2026-02-28"
N_PARTS_CALIBRATION = 60
N_PARTS_FIT = 20
MIN_GROUP_SIZE = 1000
HISTORY_ROWS = 12
PERSISTENCE_WINDOW_S = 30.0
FLIGHT_CHUNK_SIZE = 1000  # bellek guvenligi icin pencereleme bu buyuklukte parcalanir

CUSUM_TARGET_VECTOR_SHIFT_MPS = 2.0
CUSUM_MAX_GAP_S = 60.0
CUSUM_MISSING_RESET_S = 60.0
CUSUM_Z_CLIP = 3.0
_PLACEHOLDER_THRESHOLD_H = 1.0

EPISODE_CONTRACT = EpisodeContract(merge_gap_s=60.0, emission_time_col="t_end")

SOURCE_COLUMNS = [
    "flight_id", "timestamp_utc", "lat", "lon", "alt", "alt_geom_m", "on_ground",
    "ground_speed_ms", "track_deg", "vertical_rate_ms", "roll_deg",
    "event_id", "event_type", "attack_onset", "observable_onset", "event_end",
    "injection_active", "observable_changed", "evaluable_truth",
]

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

RECIPE_TO_LSTM_CHANNEL = {
    "vertical_rate_frozen": "vertical_rate_residual",
    "ground_speed_biased": "speed_residual",
    "track_frozen": "heading_residual",
}
CUSUM_RECIPE = "position_ramp_stealthy"
OUT_OF_SCOPE_RECIPES = ["altitude_dropout"]

OUT_DIR = Path("artifacts/adsb/runs/20260715_contextual_physics_v1_truth_v2_eval_v1")


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


def _load_natural_day_features(paths: list[Path], day: str, keep_flights: set[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        raw = pd.read_parquet(path)
        segmented = segment_flights(raw, gap_s=1800.0)
        segmented["flight_id"] = segmented["flight_id"].map(lambda v, d=day: f"{d}:{v}")
        kept = segmented.loc[segmented["flight_id"].isin(keep_flights)]
        if not kept.empty:
            frames.append(build_feature_table(kept))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_corpus_features(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=SOURCE_COLUMNS)
    if raw["flight_id"].isna().any():
        raise ValueError(f"{path}: null flight_id")
    return build_feature_table(raw)


def _load_checkpoint():
    derived = json.loads((RUN_DIR / "derived_training_config.json").read_text(encoding="utf-8"))
    model = ContextualResidualForecaster(ContextualForecasterConfig(**derived["model_config"]))
    model.load_state_dict(torch.load(RUN_DIR / "model_state.pt", map_location="cpu", weights_only=True))
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


def _score_batched(model, X, X_mask, y, y_mask, batch_size: int = 20_000):
    if len(X) <= batch_size:
        return contextual_channel_scores(model, X, X_mask, y, y_mask)
    parts = [
        contextual_channel_scores(model, X[i:i + batch_size], X_mask[i:i + batch_size],
                                   y[i:i + batch_size], y_mask[i:i + batch_size])
        for i in range(0, len(X), batch_size)
    ]
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def _with_exposure_bounds(meta: pd.DataFrame, time_col: str) -> pd.DataFrame:
    meta = meta.copy()
    meta["t_start"] = meta.groupby("flight_id")[time_col].shift(1)
    meta["t_start"] = meta["t_start"].fillna(meta[time_col])
    meta["t_end"] = meta[time_col]
    return meta


def _score_lstm_chunked(features: pd.DataFrame, model, scaler, target_channels):
    """flight-grubu bazli parcalayarak pencereleme bellek patlamasini onler."""
    flight_ids = features["flight_id"].unique()
    score_parts, meta_parts, ymask_parts = [], [], []
    for start in range(0, len(flight_ids), FLIGHT_CHUNK_SIZE):
        chunk_ids = set(flight_ids[start:start + FLIGHT_CHUNK_SIZE])
        chunk = features.loc[features["flight_id"].isin(chunk_ids)]
        scaled = chunk.copy()
        transformed = scaler.transform(chunk)
        for channel in scaler.active_channels:
            scaled[channel] = transformed[channel]
        batch = build_contextual_forecast_windows(
            scaled, signal_columns=scaler.active_channels, target_channels=target_channels,
            history_rows=HISTORY_ROWS, context_config=_context_config(),
        )
        if len(batch.X) == 0:
            continue
        scores, _, _ = _score_batched(model, batch.X, batch.X_mask, batch.y, batch.y_mask)
        score_parts.append(scores)
        meta_parts.append(batch.meta)
        ymask_parts.append(batch.y_mask)
    scores = np.concatenate(score_parts) if score_parts else np.zeros((0, len(target_channels)))
    y_mask = np.concatenate(ymask_parts) if ymask_parts else np.zeros((0, len(target_channels)))
    meta = (_with_exposure_bounds(pd.concat(meta_parts, ignore_index=True), "target_timestamp_utc")
            if meta_parts else pd.DataFrame(columns=["flight_id", "target_timestamp_utc", "context_phase", "context_cadence", "t_start", "t_end"]))
    return scores, y_mask, meta


def _channel_conformal_frame(scores, y_mask, meta, target_channels, channel, calibrator):
    idx = target_channels.index(channel)
    frame = pd.DataFrame({
        "flight_id": meta["flight_id"], "timestamp_utc": meta["target_timestamp_utc"],
        "t_start": meta["t_start"], "t_end": meta["t_end"],
        "channel": channel, "context_phase": meta["context_phase"],
        "context_cadence": meta["context_cadence"], "score": scores[:, idx],
    })
    valid = y_mask[:, idx] > 0
    frame = frame.loc[valid].reset_index(drop=True)
    conformal = calibrator.transform(frame)
    frame["conformal_p_value"] = conformal["conformal_p_value"].to_numpy()
    return frame


def _frozen_alpha_by_pareto(dev_report: dict, channel: str, profile: str, pareto_grid, channel_shares) -> dict[str, float]:
    curve = dev_report["channels"][channel]["profiles"][profile]
    share = channel_shares.get(channel, 0.0)
    return {
        str(v): min(curve, key=lambda r: abs((r["alert_episodes_per_scoreable_flight_hour"] or 1e9) - share * v / 100.0))["alpha"]
        for v in pareto_grid
    }


def evaluate_lstm_recipe(recipe: str, channel: str, model, scaler, target_channels, calibrator,
                          clean_frames: dict, dev_report: dict, pareto_grid, channel_shares) -> dict:
    print(f"--- {recipe} ({channel}) ---", flush=True)
    corrupt_features = _load_corpus_features(CORPUS_DIR / f"{recipe}.parquet")
    events = truth_event_table(corrupt_features)
    denominators = event_observability_denominators(events)
    eligible = events.loc[events["observable_eligible"].fillna(False)]
    print(f"  events: {len(events)}, observable_eligible: {len(eligible)}", flush=True)

    scores, y_mask, meta = _score_lstm_chunked(corrupt_features, model, scaler, target_channels)
    print(f"  skorlanabilir pencere: {len(meta)}", flush=True)
    corrupt_frame = _channel_conformal_frame(scores, y_mask, meta, target_channels, channel, calibrator)
    clean_frame = clean_frames[channel]

    profiles_out = {}
    for profile in PROFILES[channel]:
        frozen_alpha = _frozen_alpha_by_pareto(dev_report, channel, profile.anomaly_type, pareto_grid, channel_shares)
        points = []
        for v, alpha in frozen_alpha.items():
            budget = ChannelAlertBudget(total_alpha=alpha, channel_alpha={channel: alpha})
            decided = apply_detector_profile_fast(corrupt_frame, profile=profile, budget=budget)
            alarm = decided["alarm"].to_numpy()
            detection = event_detection_metrics(eligible, corrupt_frame, alarm)
            coverage = active_interval_coverage(eligible, corrupt_frame, alarm)
            clean_decided = apply_detector_profile_fast(clean_frame, profile=profile, budget=budget)
            clean_burden = natural_alert_burden(clean_frame, clean_decided["alarm"].to_numpy(), contract=EPISODE_CONTRACT)
            points.append({
                "pareto_v": float(v), "frozen_alpha": alpha,
                "event_recall": detection["event_recall"],
                "n_events": detection["n_events"], "n_detected_events": detection["n_detected_events"],
                "first_alarm_delay_s_median": detection["first_alarm_delay_s"]["median"],
                "first_alarm_delay_s_p95": detection["first_alarm_delay_s"]["p95"],
                "active_interval_coverage_micro_fraction": coverage["micro_fraction"],
                "paired_clean_natural_burden_per_hour": clean_burden["alert_episodes_per_scoreable_flight_hour"],
            })
            print(f"  {profile.anomaly_type} Pareto V={v}: recall={detection['event_recall']}, "
                  f"n_events={detection['n_events']}, clean_burden={clean_burden['alert_episodes_per_scoreable_flight_hour']}", flush=True)
        profiles_out[profile.anomaly_type] = points
    return {
        "recipe": recipe, "channel": channel,
        "event_observability_denominators": denominators,
        "profiles": profiles_out,
    }


def evaluate_cusum_recipe(detector, clean_scored: pd.DataFrame, cusum_report: dict, pareto_grid) -> dict:
    print(f"--- {CUSUM_RECIPE} (east/north CUSUM) ---", flush=True)
    corrupt_features = _load_corpus_features(CORPUS_DIR / f"{CUSUM_RECIPE}.parquet")
    events = truth_event_table(corrupt_features)
    denominators = event_observability_denominators(events)
    eligible = events.loc[events["observable_eligible"].fillna(False)]
    print(f"  events: {len(events)}, observable_eligible: {len(eligible)}", flush=True)

    scored = detector.score_rows(corrupt_features)
    bounded = _with_exposure_bounds(corrupt_features[["flight_id", "timestamp_utc"]].copy(), "timestamp_utc")
    bounded["cusum_joint_score"] = scored["cusum_joint_score"].to_numpy()
    bounded["cusum_evaluable"] = scored["cusum_evaluable"].to_numpy()

    frozen_h_by_v = cusum_report["derived_h_by_pareto_point"]
    points = []
    for v in pareto_grid:
        h = frozen_h_by_v[str(v)]
        alarm = bounded["cusum_evaluable"].to_numpy() & (bounded["cusum_joint_score"].to_numpy() > h)
        detection = event_detection_metrics(eligible, bounded, alarm)
        coverage = active_interval_coverage(eligible, bounded, alarm)
        clean_alarm = clean_scored["cusum_evaluable"].to_numpy() & (clean_scored["cusum_joint_score"].to_numpy() > h)
        clean_scoreable = clean_scored.loc[clean_scored["cusum_evaluable"]].reset_index(drop=True)
        clean_alarm_on_scoreable = clean_alarm[clean_scored["cusum_evaluable"].to_numpy()]
        clean_burden = natural_alert_burden(clean_scoreable, clean_alarm_on_scoreable, contract=EPISODE_CONTRACT)
        points.append({
            "pareto_v": float(v), "frozen_threshold_h": h,
            "event_recall": detection["event_recall"],
            "n_events": detection["n_events"], "n_detected_events": detection["n_detected_events"],
            "first_alarm_delay_s_median": detection["first_alarm_delay_s"]["median"],
            "first_alarm_delay_s_p95": detection["first_alarm_delay_s"]["p95"],
            "active_interval_coverage_micro_fraction": coverage["micro_fraction"],
            "paired_clean_natural_burden_per_hour": clean_burden["alert_episodes_per_scoreable_flight_hour"],
        })
        print(f"  east_north_cusum Pareto V={v}: recall={detection['event_recall']}, "
              f"n_events={detection['n_events']}, clean_burden={clean_burden['alert_episodes_per_scoreable_flight_hour']}", flush=True)
    return {
        "recipe": CUSUM_RECIPE, "channels": list(VECTOR_RESIDUAL_FEATURES),
        "event_observability_denominators": denominators,
        "points": points,
    }


def main() -> None:
    budget = json.loads(BUDGET_CONFIG_PATH.read_text(encoding="utf-8"))
    pareto_grid = budget["budget_grid_episodes_per_100_scoreable_flight_hours"]
    channel_shares = budget["budget_shares_of_total"]
    dev_report = json.loads(LSTM_DEV_REPORT.read_text(encoding="utf-8"))
    cusum_report = json.loads(CUSUM_DEV_REPORT.read_text(encoding="utf-8"))

    print("Checkpoint yukleniyor...", flush=True)
    model, scaler, target_channels = _load_checkpoint()

    print("Calibration gunu taraniyor (conformal kalibratoru yeniden fit etmek icin)...", flush=True)
    cal_flights = _day_flight_ids("calibration", CALIBRATION_DAY)
    cal_paths = _role_input_paths("fit", N_PARTS_CALIBRATION)
    cal_features = _load_natural_day_features(cal_paths, CALIBRATION_DAY, cal_flights)
    cal_scores, cal_ymask, cal_meta = _score_lstm_chunked(cal_features, model, scaler, target_channels)
    long_rows = []
    for i, channel in enumerate(target_channels):
        valid = cal_ymask[:, i] > 0
        long_rows.append(pd.DataFrame({
            "channel": channel,
            "context_phase": cal_meta.loc[valid, "context_phase"].to_numpy(),
            "context_cadence": cal_meta.loc[valid, "context_cadence"].to_numpy(),
            "score": cal_scores[valid, i],
        }))
    long_frame = pd.concat(long_rows, ignore_index=True)
    calibrator = HierarchicalConformalCalibrator(
        ConditionalCalibrationConfig(min_group_size=MIN_GROUP_SIZE)
    ).fit(long_frame, data_role=NATURAL_CALIBRATION_ROLE, contains_synthetic=False)
    print(f"  kalibrasyon tablosu: {len(long_frame)} satir", flush=True)

    print("CUSUM fit gunu taraniyor...", flush=True)
    fit_flights = _day_flight_ids("fit", FIT_DAY)
    fit_paths = _role_input_paths("fit", N_PARTS_FIT)
    fit_features = _load_natural_day_features(fit_paths, FIT_DAY, fit_flights)
    placeholder_config = CusumConfig(
        target_vector_shift_mps=CUSUM_TARGET_VECTOR_SHIFT_MPS, threshold_h=_PLACEHOLDER_THRESHOLD_H,
        max_gap_s=CUSUM_MAX_GAP_S, missing_reset_s=CUSUM_MISSING_RESET_S, z_clip=CUSUM_Z_CLIP,
        channels=tuple(VECTOR_RESIDUAL_FEATURES),
    )
    detector = VectorPageCUSUM(placeholder_config).fit(fit_features)
    print(f"  CUSUM kalibrasyonu: {detector.calibration_}", flush=True)

    print("Truth-v2 corpus -- clean.parquet (paired referans) taraniyor...", flush=True)
    clean_features = _load_corpus_features(CORPUS_DIR / "clean.parquet")
    clean_scores, clean_ymask, clean_meta = _score_lstm_chunked(clean_features, model, scaler, target_channels)
    clean_frames = {
        channel: _channel_conformal_frame(clean_scores, clean_ymask, clean_meta, target_channels, channel, calibrator)
        for channel in RECIPE_TO_LSTM_CHANNEL.values()
    }
    clean_cusum_scored_raw = detector.score_rows(clean_features)
    clean_cusum_scored = _with_exposure_bounds(clean_features[["flight_id", "timestamp_utc"]].copy(), "timestamp_utc")
    clean_cusum_scored["cusum_joint_score"] = clean_cusum_scored_raw["cusum_joint_score"].to_numpy()
    clean_cusum_scored["cusum_evaluable"] = clean_cusum_scored_raw["cusum_evaluable"].to_numpy()

    results = {}
    for recipe, channel in RECIPE_TO_LSTM_CHANNEL.items():
        results[recipe] = evaluate_lstm_recipe(
            recipe, channel, model, scaler, target_channels, calibrator,
            clean_frames, dev_report, pareto_grid, channel_shares,
        )
    results[CUSUM_RECIPE] = evaluate_cusum_recipe(detector, clean_cusum_scored, cusum_report, pareto_grid)

    report = {
        "corpus_dir": str(CORPUS_DIR),
        "n_calibration_parts_used": len(cal_paths),
        "n_cusum_fit_parts_used": len(fit_paths),
        "out_of_scope_recipes": OUT_OF_SCOPE_RECIPES,
        "out_of_scope_reason": "altitude_dropout S2 veri-kalitesi katmaninin isi, "
                                "contextual_physics_v1'in 5 NN/CUSUM fizik kanaliyla dogrudan eslesmiyor",
        "note": "Butun alfa/h degerleri ADR-041'de dondurulmus -- burada hicbir yeni esik "
                "secilmedi/aranmadi.",
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "truth_v2_eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {OUT_DIR / 'truth_v2_eval_report.json'}", flush=True)


if __name__ == "__main__":
    main()
