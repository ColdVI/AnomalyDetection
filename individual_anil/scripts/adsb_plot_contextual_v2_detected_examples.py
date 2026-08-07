"""Create explicitly detector-selected V=5 flight alarm anatomy examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.cusum import VectorPageCUSUM  # noqa: E402
from scripts.adsb_contextual_physics_v2_calibrate import _load_checkpoint  # noqa: E402
from scripts.adsb_contextual_physics_v2_truth_v2_eval import (  # noqa: E402
    _fit_frozen_calibrator,
    _load_json,
    _score_cusum,
)
from scripts.adsb_plot_contextual_v2_flight_alarms import (  # noqa: E402
    BUDGET_V,
    RECIPES,
    _budget_key,
    _plot_recipe,
)
from scripts.adsb_plot_contextual_v2_timelines import (  # noqa: E402
    FROZEN_TRAIN_CONFIG_PATH,
    MIN_ROWS,
    _add_persistence,
    _model_channel_scores,
    _read_flight,
)
from scripts.adsb_train_contextual_physics_v2 import _sha256_file  # noqa: E402


class DetectedExampleContractError(RuntimeError):
    pass


def _candidates(path: Path) -> list[str]:
    metadata = pd.read_parquet(
        path,
        columns=["flight_id", "event_id", "observable_onset", "evaluable_truth"],
    )
    eligible = metadata["event_id"].notna() & metadata["observable_onset"].notna()
    eligible &= metadata["evaluable_truth"].fillna(False).astype(bool)
    counts = metadata.loc[eligible].groupby("flight_id", sort=True).size()
    return sorted(map(str, counts[counts >= MIN_ROWS].index))


def _onset(features: pd.DataFrame) -> float:
    values = pd.to_numeric(features["observable_onset"], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise DetectedExampleContractError("Expected one observable onset")
    return float(values[0])


def run(
    run_dir: Path,
    calibration_dir: Path,
    corpus_dir: Path,
    out_dir: Path,
    budget_v: float,
) -> list[dict[str, Any]]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    training = _load_json(run_dir / "training_report.json")
    if training["natural_calibration_diagnostic"]["magnitude_domination_flagged_at_0_8"] is not False:
        raise DetectedExampleContractError("Magnitude gate is not false")
    manifest = _load_json(run_dir / "run_manifest.json")
    config_path = Path.cwd() / manifest.get("config_path", FROZEN_TRAIN_CONFIG_PATH.as_posix())
    if _sha256_file(config_path) != manifest["config_sha256"]:
        raise DetectedExampleContractError("Frozen training config SHA-256 mismatch")
    train_config = _load_json(config_path)
    calibration = _load_json(calibration_dir / "calibration_report.json")
    model, scaler, target_channels = _load_checkpoint(run_dir)
    calibrator = _fit_frozen_calibrator(calibration_dir, calibration)
    detector = VectorPageCUSUM.from_dict(calibration["cusum"]["detector_template"])
    multiplier = float(calibration["persistence_v2"]["reference_shift_multiplier"])
    cusum_mapping = calibration["cusum"]["selected_threshold_h_by_budget"]
    cusum_h = float(cusum_mapping[_budget_key(cusum_mapping, budget_v)])
    out_dir.mkdir(parents=True, exist_ok=False)
    summaries: list[dict[str, Any]] = []

    for recipe, spec in RECIPES.items():
        selected = None
        for rank, flight_id in enumerate(_candidates(corpus_dir / f"{recipe}.parquet"), start=1):
            injected_features = _read_flight(corpus_dir / f"{recipe}.parquet", flight_id)
            onset = _onset(injected_features)
            if recipe == "ground_speed_biased":
                injected_model = _add_persistence(
                    _model_channel_scores(
                        injected_features,
                        channel=spec["channel"],
                        model=model,
                        scaler=scaler,
                        target_channels=target_channels,
                        train_config=train_config,
                        calibrator=calibrator,
                    ),
                    multiplier,
                )
                profile = calibration["persistence_v2"]["channels"][spec["channel"]]["profiles"][spec["persistence_profile"]]
                mapping = profile["selected_threshold_h_by_budget"]
                threshold = float(mapping[_budget_key(mapping, budget_v)])
                alarm = injected_model["persistence_v2_evaluable"].to_numpy(bool) & (
                    injected_model["persistence_v2_state"].to_numpy(float) > threshold
                )
                after = pd.to_numeric(injected_model["timestamp_utc"], errors="coerce").to_numpy(float) >= onset
                if (alarm & after).any():
                    selected = (rank, flight_id, injected_features, injected_model, None)
                    break
            else:
                injected_cusum = _score_cusum(injected_features, detector)
                alarm = injected_cusum["cusum_evaluable"].to_numpy(bool) & (
                    injected_cusum["cusum_joint_score"].to_numpy(float) > cusum_h
                )
                after = pd.to_numeric(injected_cusum["timestamp_utc"], errors="coerce").to_numpy(float) >= onset
                if (alarm & after).any():
                    selected = (rank, flight_id, injected_features, None, injected_cusum)
                    break
        if selected is None:
            raise DetectedExampleContractError(f"No detected example for {recipe} at V={budget_v}")
        rank, flight_id, injected_features, injected_model, injected_cusum = selected
        clean_features = _read_flight(corpus_dir / "clean.parquet", flight_id)
        clean_model = _add_persistence(
            _model_channel_scores(
                clean_features,
                channel=spec["channel"],
                model=model,
                scaler=scaler,
                target_channels=target_channels,
                train_config=train_config,
                calibrator=calibrator,
            ),
            multiplier,
        )
        if injected_model is None:
            injected_model = _add_persistence(
                _model_channel_scores(
                    injected_features,
                    channel=spec["channel"],
                    model=model,
                    scaler=scaler,
                    target_channels=target_channels,
                    train_config=train_config,
                    calibrator=calibrator,
                ),
                multiplier,
            )
        if injected_cusum is None:
            injected_cusum = _score_cusum(injected_features, detector)
        output = out_dir / f"{recipe}__{flight_id}__V{budget_v:g}_detected_example.png"
        summary = _plot_recipe(
            recipe=recipe,
            spec=spec,
            flight_id=flight_id,
            clean_features=clean_features,
            injected_features=injected_features,
            clean_model=clean_model,
            injected_model=injected_model,
            clean_cusum=_score_cusum(clean_features, detector),
            injected_cusum=injected_cusum,
            calibration=calibration,
            budget_v=budget_v,
            output=output,
        )
        summary["selection_policy"] = "first_lexicographic_flight_detected_after_observable_onset_at_frozen_budget"
        summary["selection_rank"] = rank
        summary["interpretation"] = "alarm_anatomy_example_not_unbiased_performance_estimate"
        summaries.append(summary)
        print(output, flush=True)

    (out_dir / "detected_example_summary.json").write_text(
        json.dumps({"schema_version": 1, "budget_v": budget_v, "selection_warning": "Detector-selected examples demonstrate alarm anatomy; use full-corpus report for performance.", "flights": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summaries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/adsb/runs/20260725_contextual_physics_v2_colab_cuda_v1"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("artifacts/adsb/runs/20260724_contextual_physics_v2_calibration_v1"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/objectstore/synthetic/adsb_v2_20260713_01"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/adsb/plots/contextual_v2_detected_examples"))
    parser.add_argument("--budget-v", type=float, default=BUDGET_V)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args.run_dir.resolve(), args.calibration_dir.resolve(), args.corpus_dir.resolve(), args.out_dir.resolve(), args.budget_v)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
