"""contextual_physics_v1 -- veri rolu #4: donmus natural rehearsal.

Prereg (docs/adsb_contextual_candidate_v1_prereg_2026-07-14.md, "Veri rolleri ve
degerlendirme sirasi"): "4. Donmus natural rehearsal: geri besleme yok." ve ilk
gate kosulu truth-v2'ye (rol #5) gecmeden once development VE rehearsal'in
onceden verilen butceyi sagladigini gerektirir.

Bu script HICBIR yeni esik/alfa/h SECMEZ -- ADR-041'de zaten donmus, Pareto
butce noktalarina en yakin alfa (LSTM) ve h (CUSUM) degerlerini development_
burden_curves.json / cusum_burden_report.json'dan okuyup, ucuncu, bagimsiz bir
gunde (2026-03-16, rehearsal rolu -- ne fit/calibration ne development gormus)
AYNEN uygular. Amac: donmus konfigurasyon farkli bir gunde de kararli mi.

Kullanim:
    python scripts/adsb_contextual_physics_v1_rehearsal.py
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
from adsb.evaluation import EpisodeContract, natural_alert_burden  # noqa: E402
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

CALIBRATION_DAY = "2026-02-28"
FIT_DAY = "2026-02-28"
REHEARSAL_DAY = "2026-03-16"
SEGMENT_GAP_S = 1800.0
N_PARTS_CALIBRATION = 60
N_PARTS_FIT = 20
N_PARTS_REHEARSAL = 20
MIN_GROUP_SIZE = 1000
HISTORY_ROWS = 12
PERSISTENCE_WINDOW_S = 30.0

CUSUM_TARGET_VECTOR_SHIFT_MPS = 2.0
CUSUM_MAX_GAP_S = 60.0
CUSUM_MISSING_RESET_S = 60.0
CUSUM_Z_CLIP = 3.0
_PLACEHOLDER_THRESHOLD_H = 1.0

EPISODE_CONTRACT = EpisodeContract(merge_gap_s=60.0, emission_time_col="t_end")

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

OUT_DIR = Path("artifacts/adsb/runs/20260715_contextual_physics_v1_rehearsal_v1")


def _frozen_alpha_by_pareto(dev_report: dict, channel: str, profile: str, pareto_grid: list[float], channel_shares: dict) -> dict[str, float]:
    curve = dev_report["channels"][channel]["profiles"][profile]
    share = channel_shares.get(channel, 0.0)
    out = {}
    for v in pareto_grid:
        target = share * v / 100.0
        best = min(curve, key=lambda r: abs((r["alert_episodes_per_scoreable_flight_hour"] or 1e9) - target))
        out[str(v)] = best["alpha"]
    return out


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


def run_lstm_rehearsal(model, scaler, target_channels, calibrator, rehearsal_paths, rehearsal_flights) -> dict:
    dev_report = json.loads(LSTM_DEV_REPORT.read_text(encoding="utf-8"))
    budget = json.loads(BUDGET_CONFIG_PATH.read_text(encoding="utf-8"))
    pareto_grid = budget["budget_grid_episodes_per_100_scoreable_flight_hours"]
    channel_shares = budget["budget_shares_of_total"]

    score_parts, meta_parts, ymask_parts = [], [], []
    for i, path in enumerate(rehearsal_paths):
        chunk = _load_day_features([path], REHEARSAL_DAY, rehearsal_flights)
        if chunk.empty:
            continue
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
        print(f"  LSTM rehearsal parca {i + 1}/{len(rehearsal_paths)}: {len(scores)} pencere", flush=True)

    scores = np.concatenate(score_parts)
    y_mask = np.concatenate(ymask_parts)
    meta = _with_exposure_bounds(pd.concat(meta_parts, ignore_index=True), "target_timestamp_utc")
    print(f"  LSTM rehearsal toplam: {len(meta)} skorlanabilir pencere", flush=True)

    result: dict = {}
    for channel, profiles in PROFILES.items():
        idx = target_channels.index(channel)
        channel_scored = pd.DataFrame({
            "flight_id": meta["flight_id"], "timestamp_utc": meta["target_timestamp_utc"],
            "t_start": meta["t_start"], "t_end": meta["t_end"],
            "channel": channel, "context_phase": meta["context_phase"],
            "context_cadence": meta["context_cadence"], "score": scores[:, idx],
        })
        valid = y_mask[:, idx] > 0
        channel_scored = channel_scored.loc[valid].reset_index(drop=True)
        conformal = calibrator.transform(channel_scored)
        channel_scored["conformal_p_value"] = conformal["conformal_p_value"].to_numpy()

        for profile in profiles:
            frozen_alpha = _frozen_alpha_by_pareto(dev_report, channel, profile.anomaly_type, pareto_grid, channel_shares)
            rows = []
            for v, alpha in frozen_alpha.items():
                decided = apply_detector_profile_fast(
                    channel_scored, profile=profile,
                    budget=ChannelAlertBudget(total_alpha=alpha, channel_alpha={channel: alpha}),
                )
                burden = natural_alert_burden(channel_scored, decided["alarm"].to_numpy(), contract=EPISODE_CONTRACT)
                dev_target = channel_shares.get(channel, 0.0) * float(v) / 100.0
                rows.append({
                    "pareto_v": float(v), "frozen_alpha": alpha, "target_per_hour": dev_target,
                    "rehearsal_rate_per_hour": burden["alert_episodes_per_scoreable_flight_hour"],
                    "n_alert_episodes": burden["n_alert_episodes"],
                    "scoreable_flight_hours": burden["scoreable_flight_hours"],
                })
                print(f"  {channel}/{profile.anomaly_type} Pareto V={v} alpha={alpha:.2e}: "
                      f"rehearsal={burden['alert_episodes_per_scoreable_flight_hour']:.5f}/saat "
                      f"(hedef {dev_target:.5f}/saat)", flush=True)
            result[profile.anomaly_type] = {"channel": channel, "points": rows}
    return result


def run_cusum_rehearsal(rehearsal_paths, rehearsal_flights) -> dict:
    cusum_report = json.loads(CUSUM_DEV_REPORT.read_text(encoding="utf-8"))
    frozen_h_by_v = cusum_report["derived_h_by_pareto_point"]
    combined_share = cusum_report["combined_budget_share"]

    print("CUSUM: fit gunu taraniyor...", flush=True)
    fit_flights = _day_flight_ids("fit", FIT_DAY)
    fit_paths = _role_input_paths("fit", N_PARTS_FIT)
    fit_features = _load_day_features(fit_paths, FIT_DAY, fit_flights)

    placeholder_config = CusumConfig(
        target_vector_shift_mps=CUSUM_TARGET_VECTOR_SHIFT_MPS, threshold_h=_PLACEHOLDER_THRESHOLD_H,
        max_gap_s=CUSUM_MAX_GAP_S, missing_reset_s=CUSUM_MISSING_RESET_S, z_clip=CUSUM_Z_CLIP,
        channels=tuple(VECTOR_RESIDUAL_FEATURES),
    )
    detector = VectorPageCUSUM(placeholder_config).fit(fit_features)
    print(f"  CUSUM kalibrasyonu: {detector.calibration_}", flush=True)

    scored_parts = []
    for i, path in enumerate(rehearsal_paths):
        chunk = _load_day_features([path], REHEARSAL_DAY, rehearsal_flights)
        if chunk.empty:
            continue
        scored = detector.score_rows(chunk)
        bounded = _with_exposure_bounds(chunk[["flight_id", "timestamp_utc"]].copy(), "timestamp_utc")
        bounded["cusum_joint_score"] = scored["cusum_joint_score"].to_numpy()
        bounded["cusum_evaluable"] = scored["cusum_evaluable"].to_numpy()
        scored_parts.append(bounded)
        print(f"  CUSUM rehearsal parca {i + 1}/{len(rehearsal_paths)}: {len(chunk)} satir", flush=True)
    scored = pd.concat(scored_parts, ignore_index=True)
    print(f"  CUSUM rehearsal toplam: {len(scored)} satir, {int(scored['cusum_evaluable'].sum())} degerlendirilebilir", flush=True)

    rows = []
    for v, h in frozen_h_by_v.items():
        alarm = scored["cusum_evaluable"].to_numpy() & (scored["cusum_joint_score"].to_numpy() > h)
        scoreable = scored.loc[scored["cusum_evaluable"]].reset_index(drop=True)
        alarm_on_scoreable = alarm[scored["cusum_evaluable"].to_numpy()]
        burden = natural_alert_burden(scoreable, alarm_on_scoreable, contract=EPISODE_CONTRACT)
        target = combined_share * float(v) / 100.0
        rows.append({
            "pareto_v": float(v), "frozen_threshold_h": h, "target_per_hour": target,
            "rehearsal_rate_per_hour": burden["alert_episodes_per_scoreable_flight_hour"],
            "n_alert_episodes": burden["n_alert_episodes"],
            "scoreable_flight_hours": burden["scoreable_flight_hours"],
        })
        print(f"  east_north_cusum Pareto V={v} h={h:.2f}: "
              f"rehearsal={burden['alert_episodes_per_scoreable_flight_hour']:.5f}/saat "
              f"(hedef {target:.5f}/saat)", flush=True)
    return {"channels": list(VECTOR_RESIDUAL_FEATURES), "points": rows}


def main() -> None:
    print("Checkpoint yukleniyor...", flush=True)
    model, scaler, target_channels = _load_checkpoint()

    print("Calibration gunu taraniyor (conformal kalibratoru yeniden fit etmek icin)...", flush=True)
    cal_flights = _day_flight_ids("calibration", CALIBRATION_DAY)
    cal_paths = _role_input_paths("fit", N_PARTS_CALIBRATION)
    cal_features = _load_day_features(cal_paths, CALIBRATION_DAY, cal_flights)
    scaled_cal = cal_features.copy()
    transformed = scaler.transform(cal_features)
    for channel in scaler.active_channels:
        scaled_cal[channel] = transformed[channel]
    cal_batch = build_contextual_forecast_windows(
        scaled_cal, signal_columns=scaler.active_channels, target_channels=target_channels,
        history_rows=HISTORY_ROWS, context_config=_context_config(),
    )
    cal_scores, _, _ = _score_batched(model, cal_batch.X, cal_batch.X_mask, cal_batch.y, cal_batch.y_mask)
    long_rows = []
    for i, channel in enumerate(target_channels):
        valid = cal_batch.y_mask[:, i] > 0
        long_rows.append(pd.DataFrame({
            "channel": channel,
            "context_phase": cal_batch.meta.loc[valid, "context_phase"].to_numpy(),
            "context_cadence": cal_batch.meta.loc[valid, "context_cadence"].to_numpy(),
            "score": cal_scores[valid, i],
        }))
    long_frame = pd.concat(long_rows, ignore_index=True)
    calibrator = HierarchicalConformalCalibrator(
        ConditionalCalibrationConfig(min_group_size=MIN_GROUP_SIZE)
    ).fit(long_frame, data_role=NATURAL_CALIBRATION_ROLE, contains_synthetic=False)
    print(f"  kalibrasyon tablosu: {len(long_frame)} satir", flush=True)

    print("Rehearsal gunu (2026-03-16) taraniyor...", flush=True)
    rehearsal_flights = _day_flight_ids("rehearsal", REHEARSAL_DAY)
    rehearsal_paths = _role_input_paths("rehearsal", N_PARTS_REHEARSAL)

    lstm_result = run_lstm_rehearsal(model, scaler, target_channels, calibrator, rehearsal_paths, rehearsal_flights)
    cusum_result = run_cusum_rehearsal(rehearsal_paths, rehearsal_flights)

    report = {
        "rehearsal_day": REHEARSAL_DAY,
        "n_calibration_parts_used": len(cal_paths),
        "n_rehearsal_parts_used": len(rehearsal_paths),
        "n_cusum_fit_parts_used": N_PARTS_FIT,
        "note": "Bu turda hicbir yeni esik secilmedi -- ADR-041'de dondurulmus alfa/h degerleri "
                "AYNEN uygulandi. Amac: farkli, bagimsiz bir gunde kararlilik.",
        "lstm_profiles": lstm_result,
        "cusum": cusum_result,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rehearsal_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor: {OUT_DIR / 'rehearsal_report.json'}", flush=True)


if __name__ == "__main__":
    main()
