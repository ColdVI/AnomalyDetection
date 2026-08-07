"""Plot paired clean/injected contextual-v2 score timelines.

For one deterministic, sufficiently long observable flight per truth-v2 recipe,
the figure stacks model conformal p-value, cumulative persistence_v2 state, and
the unchanged vector Page-CUSUM score.  Clean and injected streams share the
same axes; the observable injection onset is marked without selecting examples
from detector performance.
"""

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
from adsb.features import build_feature_table  # noqa: E402
from adsb.models.contextual_persistence_v2 import (  # noqa: E402
    CumulativeConformalPersistence,
    PersistenceV2Config,
)
from scripts.adsb_contextual_physics_v2_calibrate import (  # noqa: E402
    PERSISTENCE_MAX_GAP_S,
    PERSISTENCE_MISSING_RESET_S,
    PERSISTENCE_SURPRISE_CLIP,
    _load_checkpoint,
)
from scripts.adsb_contextual_physics_v2_truth_v2_eval import (  # noqa: E402
    CORPUS_DIR,
    SOURCE_COLUMNS,
    _conformal_transform,
    _fit_frozen_calibrator,
    _load_json,
    _score_cusum,
    _score_model_long,
)
from scripts.adsb_train_contextual_physics_v2 import _sha256_file  # noqa: E402

MIN_ROWS = 120
FROZEN_TRAIN_CONFIG_PATH = Path("configs/adsb_contextual_physics_v2_train.json")
RECIPE_CHANNELS = {
    "vertical_rate_frozen": "vertical_rate_residual",
    "ground_speed_biased": "speed_residual",
    "track_frozen": "heading_residual",
    "position_ramp_stealthy": "east_velocity_residual",
    "altitude_dropout": "altitude_source_residual",
}


class TimelineContractError(RuntimeError):
    pass


def _pick_flight(path: Path, *, min_rows: int = MIN_ROWS) -> str:
    metadata = pd.read_parquet(
        path,
        columns=["flight_id", "event_id", "observable_onset", "evaluable_truth"],
    )
    eligible = metadata["event_id"].notna() & metadata["observable_onset"].notna()
    eligible &= metadata["evaluable_truth"].fillna(False).astype(bool)
    counts = metadata.loc[eligible].groupby("flight_id", sort=True).size()
    candidates = sorted(map(str, counts[counts >= min_rows].index))
    if not candidates:
        raise TimelineContractError(f"No deterministic observable example in {path}")
    return candidates[0]


def _read_flight(path: Path, flight_id: str) -> pd.DataFrame:
    try:
        raw = pd.read_parquet(
            path,
            columns=SOURCE_COLUMNS,
            filters=[("flight_id", "==", flight_id)],
        )
    except (TypeError, ValueError):
        raw = pd.read_parquet(path, columns=SOURCE_COLUMNS)
        raw = raw.loc[raw["flight_id"].astype(str) == flight_id]
    if raw.empty:
        raise TimelineContractError(f"Paired flight {flight_id} missing from {path}")
    raw = raw.sort_values("timestamp_utc").reset_index(drop=True)
    return build_feature_table(raw)


def _model_channel_scores(
    features: pd.DataFrame,
    *,
    channel: str,
    model,
    scaler,
    target_channels: tuple[str, ...],
    train_config: dict[str, Any],
    calibrator,
) -> pd.DataFrame:
    scored = _score_model_long(
        features,
        model=model,
        scaler=scaler,
        target_channels=target_channels,
        train_config=train_config,
    )
    scored = _conformal_transform(scored, calibrator)
    result = scored.loc[scored["channel"] == channel].reset_index(drop=True)
    if result.empty:
        raise TimelineContractError(f"No model score for channel={channel}")
    return result


def _add_persistence(frame: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    detector = CumulativeConformalPersistence(
        PersistenceV2Config(
            reference_shift_multiplier=multiplier,
            threshold_h=1.0,
            max_gap_s=PERSISTENCE_MAX_GAP_S,
            missing_reset_s=PERSISTENCE_MISSING_RESET_S,
            surprise_clip=PERSISTENCE_SURPRISE_CLIP,
        )
    )
    return pd.concat([frame, detector.score(frame)], axis=1)


def _relative_minutes(frame: pd.DataFrame, origin: float, column: str) -> np.ndarray:
    return (pd.to_numeric(frame[column], errors="coerce").to_numpy(float) - origin) / 60.0


def _plot_recipe(
    *,
    recipe: str,
    flight_id: str,
    channel: str,
    clean_model: pd.DataFrame,
    injected_model: pd.DataFrame,
    clean_cusum: pd.DataFrame,
    injected_cusum: pd.DataFrame,
    onset: float,
    output_path: Path,
) -> None:
    origin = min(
        float(clean_model["timestamp_utc"].min()),
        float(injected_model["timestamp_utc"].min()),
    )
    onset_minutes = (onset - origin) / 60.0
    colors = {"clean": "#2A9D8F", "injected": "#D62828"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(
        _relative_minutes(clean_model, origin, "timestamp_utc"),
        clean_model["conformal_p_value"],
        color=colors["clean"],
        lw=1.0,
        label="clean",
    )
    axes[0].plot(
        _relative_minutes(injected_model, origin, "timestamp_utc"),
        injected_model["conformal_p_value"],
        color=colors["injected"],
        lw=1.0,
        label="injected",
    )
    axes[0].set_ylabel("model conformal p")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        _relative_minutes(clean_model, origin, "timestamp_utc"),
        clean_model["persistence_v2_state"],
        color=colors["clean"],
        lw=1.0,
    )
    axes[1].plot(
        _relative_minutes(injected_model, origin, "timestamp_utc"),
        injected_model["persistence_v2_state"],
        color=colors["injected"],
        lw=1.0,
    )
    axes[1].set_ylabel("persistence_v2")

    axes[2].plot(
        _relative_minutes(clean_cusum, origin, "timestamp_utc"),
        clean_cusum["cusum_joint_score"],
        color=colors["clean"],
        lw=1.0,
    )
    axes[2].plot(
        _relative_minutes(injected_cusum, origin, "timestamp_utc"),
        injected_cusum["cusum_joint_score"],
        color=colors["injected"],
        lw=1.0,
    )
    axes[2].set_ylabel("CUSUM score")
    axes[2].set_xlabel("flight time (minutes)")

    for axis in axes:
        axis.axvline(onset_minutes, color="black", ls="--", lw=1.2)
        axis.grid(alpha=0.25)
    axes[0].set_title(
        f"contextual_physics_v2 — {recipe} — {flight_id} — channel={channel}\n"
        "dashed line: observable injection onset"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run(
    run_dir: Path,
    calibration_dir: Path,
    corpus_dir: Path,
    out_dir: Path,
) -> list[Path]:
    training_report = _load_json(run_dir / "training_report.json")
    if training_report["natural_calibration_diagnostic"][
        "magnitude_domination_flagged_at_0_8"
    ] is not False:
        raise TimelineContractError("Magnitude-domination gate is not false")
    run_manifest = _load_json(run_dir / "run_manifest.json")
    train_config_path = Path.cwd() / run_manifest.get(
        "config_path", FROZEN_TRAIN_CONFIG_PATH.as_posix()
    )
    if _sha256_file(train_config_path) != run_manifest.get("config_sha256"):
        raise TimelineContractError("Frozen training config SHA-256 mismatch")
    train_config = _load_json(train_config_path)
    calibration_path = calibration_dir / "calibration_report.json"
    calibration_report = _load_json(calibration_path)
    if calibration_report["training_report_sha256"] != _sha256_file(
        run_dir / "training_report.json"
    ):
        raise TimelineContractError("Calibration belongs to another training report")

    model, scaler, target_channels = _load_checkpoint(run_dir)
    calibrator = _fit_frozen_calibrator(calibration_dir, calibration_report)
    cusum = VectorPageCUSUM.from_dict(calibration_report["cusum"]["detector_template"])
    multiplier = float(calibration_report["persistence_v2"]["reference_shift_multiplier"])
    out_dir.mkdir(parents=True, exist_ok=False)
    outputs: list[Path] = []

    for recipe, channel in RECIPE_CHANNELS.items():
        injected_path = corpus_dir / f"{recipe}.parquet"
        clean_path = corpus_dir / "clean.parquet"
        flight_id = _pick_flight(injected_path)
        clean_features = _read_flight(clean_path, flight_id)
        injected_features = _read_flight(injected_path, flight_id)
        clean_model = _add_persistence(
            _model_channel_scores(
                clean_features,
                channel=channel,
                model=model,
                scaler=scaler,
                target_channels=target_channels,
                train_config=train_config,
                calibrator=calibrator,
            ),
            multiplier,
        )
        injected_model = _add_persistence(
            _model_channel_scores(
                injected_features,
                channel=channel,
                model=model,
                scaler=scaler,
                target_channels=target_channels,
                train_config=train_config,
                calibrator=calibrator,
            ),
            multiplier,
        )
        onset_values = pd.to_numeric(
            injected_features["observable_onset"], errors="coerce"
        ).dropna()
        if onset_values.empty:
            raise TimelineContractError(f"No observable onset for {recipe}/{flight_id}")
        output = out_dir / f"{recipe}__{flight_id.replace(':', '_')}.png"
        _plot_recipe(
            recipe=recipe,
            flight_id=flight_id,
            channel=channel,
            clean_model=clean_model,
            injected_model=injected_model,
            clean_cusum=_score_cusum(clean_features, cusum),
            injected_cusum=_score_cusum(injected_features, cusum),
            onset=float(onset_values.iloc[0]),
            output_path=output,
        )
        outputs.append(output)
        print(output, flush=True)
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/adsb/runs/20260724_contextual_physics_v2_train_v4"),
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("artifacts/adsb/runs/20260724_contextual_physics_v2_calibration_v1"),
    )
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/adsb/plots/contextual_v2_timelines"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = run(
        args.run_dir.resolve(),
        args.calibration_dir.resolve(),
        args.corpus_dir.resolve(),
        args.out_dir.resolve(),
    )
    print(json.dumps({"png_count": len(outputs)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
