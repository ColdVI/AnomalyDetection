"""Plot single-flight alarm emissions for two contextual-v2 anomaly mechanisms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.cusum import VectorPageCUSUM  # noqa: E402
from scripts.adsb_contextual_physics_v2_calibrate import _load_checkpoint  # noqa: E402
from scripts.adsb_contextual_physics_v2_truth_v2_eval import (  # noqa: E402
    _fit_frozen_calibrator,
    _load_json,
    _score_cusum,
)
from scripts.adsb_plot_contextual_v2_timelines import (  # noqa: E402
    FROZEN_TRAIN_CONFIG_PATH,
    _add_persistence,
    _model_channel_scores,
    _pick_flight,
    _read_flight,
)
from scripts.adsb_train_contextual_physics_v2 import _sha256_file  # noqa: E402

BUDGET_V = 5.0
RECIPES = {
    "ground_speed_biased": {
        "channel": "speed_residual",
        "primary": "persistence_v2 / speed bias",
        "instant_profile": "spike",
        "persistence_profile": "bias",
    },
    "position_ramp_stealthy": {
        "channel": "east_velocity_residual",
        "primary": "two-axis Page-CUSUM",
        "instant_profile": None,
        "persistence_profile": None,
    },
}


class AlarmDashboardContractError(RuntimeError):
    pass


def _budget_key(mapping: dict[str, Any], value: float) -> str:
    matches = [key for key in mapping if float(key) == float(value)]
    if len(matches) != 1:
        raise AlarmDashboardContractError(f"Expected one frozen budget key for V={value}")
    return matches[0]


def _relative(values: pd.Series, origin: float) -> np.ndarray:
    return (pd.to_numeric(values, errors="coerce").to_numpy(float) - origin) / 60.0


def _truth_bounds(features: pd.DataFrame) -> tuple[float, float, float]:
    def scalar(column: str) -> float:
        values = pd.to_numeric(features[column], errors="coerce").dropna().unique()
        if len(values) != 1:
            raise AlarmDashboardContractError(f"Expected one {column}, got {len(values)}")
        return float(values[0])

    return scalar("attack_onset"), scalar("observable_onset"), scalar("event_end")


def _alarm_points(
    axis,
    frame: pd.DataFrame,
    values: np.ndarray,
    alarm: np.ndarray,
    origin: float,
    *,
    marker: str,
    label: str,
) -> None:
    flags = np.asarray(alarm, dtype=bool)
    if flags.any():
        axis.scatter(
            _relative(frame.loc[flags, "timestamp_utc"], origin),
            np.asarray(values, dtype=float)[flags],
            s=18,
            marker=marker,
            color="#D00000",
            zorder=6,
            label=f"{label} ({int(flags.sum())} windows)",
        )


def _decorate_truth(axis, origin: float, attack: float, observable: float, end: float) -> None:
    axis.axvspan((observable - origin) / 60.0, (end - origin) / 60.0, color="#F4A261", alpha=0.14)
    axis.axvline((attack - origin) / 60.0, color="#6C757D", ls=":", lw=1.1)
    axis.axvline((observable - origin) / 60.0, color="#111111", ls="--", lw=1.2)
    axis.grid(alpha=0.22)


def _first_delay(frame: pd.DataFrame, alarm: np.ndarray, onset: float) -> float | None:
    times = pd.to_numeric(frame.loc[np.asarray(alarm, dtype=bool), "timestamp_utc"], errors="coerce")
    times = times.loc[times >= onset]
    return None if times.empty else float(times.min() - onset)


def _plot_recipe(
    *,
    recipe: str,
    spec: dict[str, Any],
    flight_id: str,
    clean_features: pd.DataFrame,
    injected_features: pd.DataFrame,
    clean_model: pd.DataFrame,
    injected_model: pd.DataFrame,
    clean_cusum: pd.DataFrame,
    injected_cusum: pd.DataFrame,
    calibration: dict[str, Any],
    budget_v: float,
    output: Path,
) -> dict[str, Any]:
    origin = float(min(clean_features["timestamp_utc"].min(), injected_features["timestamp_utc"].min()))
    attack, observable, event_end = _truth_bounds(injected_features)
    colors = {"clean": "#2A9D8F", "injected": "#264653"}
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    if recipe == "ground_speed_biased":
        axes[0].plot(_relative(clean_features["timestamp_utc"], origin), clean_features["speed_residual"], color=colors["clean"], lw=1.0, label="clean speed residual")
        axes[0].plot(_relative(injected_features["timestamp_utc"], origin), injected_features["speed_residual"], color=colors["injected"], lw=1.0, label="injected speed residual")
        axes[0].set_ylabel("speed residual")
    else:
        axes[0].plot(_relative(clean_features["timestamp_utc"], origin), clean_features["east_velocity_residual"], color="#2A9D8F", lw=0.9, label="clean east")
        axes[0].plot(_relative(injected_features["timestamp_utc"], origin), injected_features["east_velocity_residual"], color="#D62828", lw=0.9, label="injected east")
        axes[0].plot(_relative(clean_features["timestamp_utc"], origin), clean_features["north_velocity_residual"], color="#457B9D", lw=0.9, alpha=0.8, label="clean north")
        axes[0].plot(_relative(injected_features["timestamp_utc"], origin), injected_features["north_velocity_residual"], color="#F4A261", lw=0.9, alpha=0.9, label="injected north")
        axes[0].set_ylabel("velocity residual")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)

    p_clean = clean_model["conformal_p_value"].to_numpy(float)
    p_injected = injected_model["conformal_p_value"].to_numpy(float)
    axes[1].plot(_relative(clean_model["timestamp_utc"], origin), p_clean, color=colors["clean"], lw=0.9, label="clean p")
    axes[1].plot(_relative(injected_model["timestamp_utc"], origin), p_injected, color=colors["injected"], lw=0.9, label="injected p")
    instant_alarm = np.zeros(len(injected_model), dtype=bool)
    instant_alpha = None
    if spec["instant_profile"] is not None:
        profile = calibration["persistence_v2"]["channels"][spec["channel"]]["profiles"][spec["instant_profile"]]
        mapping = profile["selected_alpha_by_budget"]
        instant_alpha = float(mapping[_budget_key(mapping, budget_v)])
        instant_alarm = p_injected <= instant_alpha
        axes[1].axhline(instant_alpha, color="#D00000", ls="--", lw=1.0, label=f"frozen alpha={instant_alpha:.2g}")
        _alarm_points(axes[1], injected_model, p_injected, instant_alarm, origin, marker="v", label="model alarm")
    axes[1].set_ylabel("conformal p")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="upper right", fontsize=8)

    state_clean = clean_model["persistence_v2_state"].to_numpy(float)
    state_injected = injected_model["persistence_v2_state"].to_numpy(float)
    axes[2].plot(_relative(clean_model["timestamp_utc"], origin), state_clean, color=colors["clean"], lw=0.9, label="clean state")
    axes[2].plot(_relative(injected_model["timestamp_utc"], origin), state_injected, color=colors["injected"], lw=0.9, label="injected state")
    persistence_alarm = np.zeros(len(injected_model), dtype=bool)
    persistence_h = None
    if spec["persistence_profile"] is not None:
        profile = calibration["persistence_v2"]["channels"][spec["channel"]]["profiles"][spec["persistence_profile"]]
        mapping = profile["selected_threshold_h_by_budget"]
        persistence_h = float(mapping[_budget_key(mapping, budget_v)])
        persistence_alarm = injected_model["persistence_v2_evaluable"].to_numpy(bool) & (state_injected > persistence_h)
        axes[2].axhline(persistence_h, color="#D00000", ls="--", lw=1.0, label=f"frozen h={persistence_h:.2f}")
        _alarm_points(axes[2], injected_model, state_injected, persistence_alarm, origin, marker="o", label="persistence alarm")
    else:
        axes[2].text(0.01, 0.93, "No frozen persistence threshold for this channel", transform=axes[2].transAxes, va="top", fontsize=8)
    axes[2].set_ylabel("persistence state")
    axes[2].legend(loc="upper left", fontsize=8)

    cusum_clean = clean_cusum["cusum_joint_score"].to_numpy(float)
    cusum_injected = injected_cusum["cusum_joint_score"].to_numpy(float)
    mapping = calibration["cusum"]["selected_threshold_h_by_budget"]
    cusum_h = float(mapping[_budget_key(mapping, budget_v)])
    cusum_alarm = injected_cusum["cusum_evaluable"].to_numpy(bool) & (cusum_injected > cusum_h)
    axes[3].plot(_relative(clean_cusum["timestamp_utc"], origin), cusum_clean, color=colors["clean"], lw=0.9, label="clean CUSUM")
    axes[3].plot(_relative(injected_cusum["timestamp_utc"], origin), cusum_injected, color=colors["injected"], lw=0.9, label="injected CUSUM")
    axes[3].axhline(cusum_h, color="#D00000", ls="--", lw=1.0, label=f"frozen h={cusum_h:.2f}")
    _alarm_points(axes[3], injected_cusum, cusum_injected, cusum_alarm, origin, marker="x", label="CUSUM alarm")
    axes[3].set_ylabel("CUSUM score")
    axes[3].set_xlabel("flight time (minutes)")
    axes[3].legend(loc="upper left", fontsize=8)

    for axis in axes:
        _decorate_truth(axis, origin, attack, observable, event_end)
    fig.suptitle(
        f"contextual_physics_v2 | {recipe} | flight={flight_id} | frozen V={budget_v:g}\n"
        f"primary detector: {spec['primary']} | orange band=observable anomaly, "
        "red markers=alarm windows",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return {
        "recipe": recipe,
        "flight_id": flight_id,
        "budget_v": budget_v,
        "primary_detector": spec["primary"],
        "attack_onset": attack,
        "observable_onset": observable,
        "event_end": event_end,
        "instant": {"threshold_alpha": instant_alpha, "alarm_windows": int(instant_alarm.sum()), "first_delay_s": _first_delay(injected_model, instant_alarm, observable)},
        "persistence_v2": {"threshold_h": persistence_h, "alarm_windows": int(persistence_alarm.sum()), "first_delay_s": _first_delay(injected_model, persistence_alarm, observable)},
        "cusum": {"threshold_h": cusum_h, "alarm_windows": int(cusum_alarm.sum()), "first_delay_s": _first_delay(injected_cusum, cusum_alarm, observable)},
        "output": output.name,
    }


def run(run_dir: Path, calibration_dir: Path, corpus_dir: Path, out_dir: Path, budget_v: float) -> list[dict[str, Any]]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    training = _load_json(run_dir / "training_report.json")
    if training["natural_calibration_diagnostic"]["magnitude_domination_flagged_at_0_8"] is not False:
        raise AlarmDashboardContractError("Magnitude gate is not false")
    manifest = _load_json(run_dir / "run_manifest.json")
    config_path = Path.cwd() / manifest.get("config_path", FROZEN_TRAIN_CONFIG_PATH.as_posix())
    if _sha256_file(config_path) != manifest["config_sha256"]:
        raise AlarmDashboardContractError("Frozen training config SHA-256 mismatch")
    train_config = _load_json(config_path)
    calibration = _load_json(calibration_dir / "calibration_report.json")
    model, scaler, target_channels = _load_checkpoint(run_dir)
    calibrator = _fit_frozen_calibrator(calibration_dir, calibration)
    detector = VectorPageCUSUM.from_dict(calibration["cusum"]["detector_template"])
    multiplier = float(calibration["persistence_v2"]["reference_shift_multiplier"])
    out_dir.mkdir(parents=True, exist_ok=False)
    summaries: list[dict[str, Any]] = []
    for recipe, spec in RECIPES.items():
        injected_path = corpus_dir / f"{recipe}.parquet"
        flight_id = _pick_flight(injected_path)
        clean_features = _read_flight(corpus_dir / "clean.parquet", flight_id)
        injected_features = _read_flight(injected_path, flight_id)
        clean_model = _add_persistence(_model_channel_scores(clean_features, channel=spec["channel"], model=model, scaler=scaler, target_channels=target_channels, train_config=train_config, calibrator=calibrator), multiplier)
        injected_model = _add_persistence(_model_channel_scores(injected_features, channel=spec["channel"], model=model, scaler=scaler, target_channels=target_channels, train_config=train_config, calibrator=calibrator), multiplier)
        output = out_dir / f"{recipe}__{flight_id}__V{budget_v:g}_alarm_dashboard.png"
        summaries.append(_plot_recipe(recipe=recipe, spec=spec, flight_id=flight_id, clean_features=clean_features, injected_features=injected_features, clean_model=clean_model, injected_model=injected_model, clean_cusum=_score_cusum(clean_features, detector), injected_cusum=_score_cusum(injected_features, detector), calibration=calibration, budget_v=budget_v, output=output))
        print(output, flush=True)
    (out_dir / "flight_alarm_dashboard_summary.json").write_text(json.dumps({"schema_version": 1, "selection": "deterministic_first_observable_flight_not_detector_selected", "budget_interpretation": "V episodes per 100 scoreable flight-hours before frozen channel/profile shares", "flights": summaries}, indent=2) + "\n", encoding="utf-8")
    return summaries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/adsb/runs/20260725_contextual_physics_v2_colab_cuda_v1"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("artifacts/adsb/runs/20260724_contextual_physics_v2_calibration_v1"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/objectstore/synthetic/adsb_v2_20260713_01"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/adsb/plots/contextual_v2_flight_alarms"))
    parser.add_argument("--budget-v", type=float, default=BUDGET_V)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = run(args.run_dir.resolve(), args.calibration_dir.resolve(), args.corpus_dir.resolve(), args.out_dir.resolve(), args.budget_v)
    print(json.dumps({"flight_count": len(summaries), "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
