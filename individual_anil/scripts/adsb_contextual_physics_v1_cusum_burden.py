"""contextual_physics_v1 -- east/north velocity residual dogal-yuk olcumu.

ADR-040'in development_burden script'inde bu iki kanal YOKTU (bilinen, durustce
beyan edilmis bir bosluk). configs/adsb_contextual_physics_v1_alarm_budget.json
onlari tek-kanalli instant/persistence/accumulation semasina degil, ORTAK
2-eksenli Page-CUSUM'a (adsb/cusum.py::VectorPageCUSUM, ADR-029) atar -- bu
dedektor zaten mevcut kural-skorlayici hattinda insa edilmisti, burada sifirdan
yazilmiyor, yalniz contextual_physics_v1'in veri-rolu sirasina (fit->calibration
->development) kablolaniyor.

threshold_h'nin p-degeri gibi dogal bir [0,1] araligi yok, bu yuzden config onu
"derived per budget_grid point from natural calibration, NOT set in this file"
olarak birakti. Burada yapilan sira:
  1) fit-rolu veride median/MAD kalibrasyonu (VectorPageCUSUM.fit)
  2) calibration-rolu veride TEK BIR score_rows() gecisi (skor h'ye bagli degil,
     yalniz alarm karsilastirmasi bagli -- ayni skoru bircok h ile karsilastirmak
     score_rows()'u tekrar tekrar cagirmayi GEREKTIRMEZ)
  3) calibration-rolu skor dagilimindan turetilmis bir h adaylari izgarasi + her
     Pareto butce noktasina en yakin h'nin DONDURULMESI
  4) o donmus h degerleri development-rolu veride, DEGISTIRILMEDEN, uygulanir --
     development sonucuna bakip h yeniden aranmaz.

Kullanim:
    python scripts/adsb_contextual_physics_v1_cusum_burden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.cusum import CusumConfig, VectorPageCUSUM  # noqa: E402
from adsb.evaluation import EpisodeContract, natural_alert_burden  # noqa: E402
from adsb.features import VECTOR_RESIDUAL_FEATURES, build_feature_table  # noqa: E402
from adsb.segmentation import segment_flights  # noqa: E402

STEP5_MANIFEST = Path("artifacts/adsb/runs/20260713_step5_full_streaming_v1/run_manifest.json")
BUDGET_CONFIG_PATH = Path("configs/adsb_contextual_physics_v1_alarm_budget.json")
FIT_DAY = "2026-02-28"
CALIBRATION_DAY = "2026-02-28"
DEVELOPMENT_DAY = "2026-03-01"
SEGMENT_GAP_S = 1800.0

N_PARTS_FIT = 20           # sadece VectorPageCUSUM.fit() (median/MAD) icin
N_PARTS_CALIBRATION = 60   # ayni ADR-039/040'daki kesif olcegi
N_PARTS_DEVELOPMENT = 20   # ADR-041 -- ayni v2 development_burden kosusuyla ayni olcek

# ADR-029'daki mevcut kural-skorlayici hattinda ONCEDEN dondurulmus CUSUM sabitleri
# (scripts/adsb_run_full_streaming_baseline.py) -- burada YENIDEN SECILMIYOR, aynen
# tasiniyor, yalniz threshold_h serbest (asagida turetiliyor).
CUSUM_TARGET_VECTOR_SHIFT_MPS = 2.0
CUSUM_MAX_GAP_S = 60.0
CUSUM_MISSING_RESET_S = 60.0
CUSUM_Z_CLIP = 3.0

# score_rows()'u tek gecis icin gereken, henuz-anlamsiz bir yer-tutucu esik.
_PLACEHOLDER_THRESHOLD_H = 1.0

EPISODE_CONTRACT = EpisodeContract(merge_gap_s=60.0, emission_time_col="t_end")

OUT_DIR = Path("artifacts/adsb/runs/20260714_contextual_physics_v1_cusum_burden_v1")


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


def _with_exposure_bounds(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["t_start"] = features.groupby("flight_id")["timestamp_utc"].shift(1)
    features["t_start"] = features["t_start"].fillna(features["timestamp_utc"])
    features["t_end"] = features["timestamp_utc"]
    return features


def _score_with_placeholder(detector: VectorPageCUSUM, features: pd.DataFrame) -> pd.DataFrame:
    scored = detector.score_rows(features)
    out = _with_exposure_bounds(features)[["flight_id", "t_start", "t_end"]].copy()
    out["cusum_joint_score"] = scored["cusum_joint_score"].to_numpy()
    out["cusum_evaluable"] = scored["cusum_evaluable"].to_numpy()
    return out


def _burden_at_threshold(scored: pd.DataFrame, threshold_h: float) -> dict:
    alarm = scored["cusum_evaluable"].to_numpy() & (scored["cusum_joint_score"].to_numpy() > threshold_h)
    scoreable = scored.loc[scored["cusum_evaluable"]].reset_index(drop=True)
    alarm_on_scoreable = alarm[scored["cusum_evaluable"].to_numpy()]
    return natural_alert_burden(scoreable, alarm_on_scoreable, contract=EPISODE_CONTRACT)


def main() -> None:
    budget = json.loads(BUDGET_CONFIG_PATH.read_text(encoding="utf-8"))
    pareto_grid = budget["budget_grid_episodes_per_100_scoreable_flight_hours"]
    channel_shares = budget["budget_shares_of_total"]
    combined_target_share = sum(channel_shares.get(c, 0.0) for c in VECTOR_RESIDUAL_FEATURES)
    print(f"east+north birlesik butce payi: {combined_target_share:.3f}", flush=True)

    print("Fit gunu -- CUSUM median/MAD kalibrasyonu icin taraniyor...", flush=True)
    fit_flights = _day_flight_ids("fit", FIT_DAY)
    fit_paths = _role_input_paths("fit", N_PARTS_FIT)
    fit_features = _load_day_features(fit_paths, FIT_DAY, fit_flights)
    print(f"  {len(fit_features)} satir, {fit_features['flight_id'].nunique()} ucus", flush=True)

    placeholder_config = CusumConfig(
        target_vector_shift_mps=CUSUM_TARGET_VECTOR_SHIFT_MPS,
        threshold_h=_PLACEHOLDER_THRESHOLD_H,
        max_gap_s=CUSUM_MAX_GAP_S,
        missing_reset_s=CUSUM_MISSING_RESET_S,
        z_clip=CUSUM_Z_CLIP,
        channels=tuple(VECTOR_RESIDUAL_FEATURES),
    )
    detector = VectorPageCUSUM(placeholder_config).fit(fit_features)
    print(f"  kalibrasyon: {detector.calibration_}", flush=True)
    print(f"  dislanan kanallar: {detector.excluded_channels_}", flush=True)

    print("Calibration gunu taraniyor (h adaylarini turetmek icin)...", flush=True)
    cal_flights = _day_flight_ids("calibration", CALIBRATION_DAY)
    cal_paths = _role_input_paths("fit", N_PARTS_CALIBRATION)
    cal_features = _load_day_features(cal_paths, CALIBRATION_DAY, cal_flights)
    cal_scored = _score_with_placeholder(detector, cal_features)
    print(f"  {len(cal_scored)} satir, {int(cal_scored['cusum_evaluable'].sum())} degerlendirilebilir", flush=True)

    evaluable_scores = cal_scored.loc[cal_scored["cusum_evaluable"], "cusum_joint_score"].to_numpy()
    quantiles = [
        0.50, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999,
        0.99995, 0.99999, 0.999995, 0.999999, 0.9999995, 0.9999999,
    ]
    h_candidates = sorted(set(float(np.quantile(evaluable_scores, q)) for q in quantiles))
    print(f"  h adaylari (calibration skor kartilinden): {[round(h, 4) for h in h_candidates]}", flush=True)

    cal_curve = [{"threshold_h": h, **_burden_at_threshold(cal_scored, h)} for h in h_candidates]

    derived_h_by_pareto_point: dict[str, float] = {}
    for v in pareto_grid:
        target_per_hour = combined_target_share * v / 100.0
        best = min(
            cal_curve,
            key=lambda r: abs((r["alert_episodes_per_scoreable_flight_hour"] or 1e9) - target_per_hour),
        )
        derived_h_by_pareto_point[str(v)] = best["threshold_h"]
        print(f"  Pareto V={v}: hedef~{target_per_hour:.5f}/saat -> h={best['threshold_h']:.4f} "
              f"({best['alert_episodes_per_scoreable_flight_hour']:.4f} episode/saat, calibration'da)", flush=True)

    print("Development gunu taraniyor (dondurulmus h degerleriyle olculecek)...", flush=True)
    dev_flights = _day_flight_ids("development", DEVELOPMENT_DAY)
    dev_paths = _role_input_paths("development", N_PARTS_DEVELOPMENT)
    dev_scored_parts = []
    for i, path in enumerate(dev_paths):
        chunk = _load_day_features([path], DEVELOPMENT_DAY, dev_flights)
        if chunk.empty:
            continue
        dev_scored_parts.append(_score_with_placeholder(detector, chunk))
        print(f"  parca {i + 1}/{len(dev_paths)}: {len(chunk)} satir", flush=True)
    dev_scored = pd.concat(dev_scored_parts, ignore_index=True)
    print(f"  development toplam: {len(dev_scored)} satir, "
          f"{int(dev_scored['cusum_evaluable'].sum())} degerlendirilebilir", flush=True)

    dev_curve = [{"threshold_h": h, **_burden_at_threshold(dev_scored, h)} for h in h_candidates]
    dev_burden_at_pareto_h = {
        v: next(r for r in dev_curve if r["threshold_h"] == h)
        for v, h in derived_h_by_pareto_point.items()
    }
    for v, row in dev_burden_at_pareto_h.items():
        print(f"  Pareto V={v} (h={row['threshold_h']:.4f}) development'ta: "
              f"{row['alert_episodes_per_scoreable_flight_hour']:.4f} episode/saat "
              f"({row['alerted_flight_fraction']:.4f} ucus-oran)", flush=True)

    report = {
        "channels": list(VECTOR_RESIDUAL_FEATURES),
        "detector": "adsb.cusum.VectorPageCUSUM (joint two-sided Page CUSUM, ADR-029)",
        "cusum_config": {
            "target_vector_shift_mps": CUSUM_TARGET_VECTOR_SHIFT_MPS,
            "max_gap_s": CUSUM_MAX_GAP_S,
            "missing_reset_s": CUSUM_MISSING_RESET_S,
            "z_clip": CUSUM_Z_CLIP,
        },
        "fit_calibration": detector.calibration_,
        "excluded_channels": detector.excluded_channels_,
        "combined_budget_share": combined_target_share,
        "pareto_grid_episodes_per_100h": pareto_grid,
        "n_fit_parts_used": len(fit_paths),
        "n_calibration_parts_used": len(cal_paths),
        "n_development_parts_used": len(dev_paths),
        "h_candidates": h_candidates,
        "calibration_curve": cal_curve,
        "derived_h_by_pareto_point": derived_h_by_pareto_point,
        "development_curve": dev_curve,
        "development_burden_at_derived_h": dev_burden_at_pareto_h,
        "note": "h adaylari yalniz calibration-rolu veriden turetildi; development-rolu "
                "veriye bakip yeniden aranmadi (ADR-041).",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cusum_burden_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRapor: {OUT_DIR / 'cusum_burden_report.json'}", flush=True)


if __name__ == "__main__":
    main()
