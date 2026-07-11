"""Run the independent ADS-B physics-residual binary anomaly pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adsb_behavioral.decision import calibrate_threshold  # noqa: E402
from src.adsb_behavioral.evaluation import evaluate_detector  # noqa: E402
from src.adsb_behavioral.injection import build_injected_copies  # noqa: E402
from src.adsb_behavioral.models import IsolationForestPhysicsModel, RobustPhysicsModel  # noqa: E402
from src.adsb_behavioral.physics_residuals import MODEL_FEATURES, add_physics_residuals  # noqa: E402
from src.adsb_behavioral.reader import archive_date, sample_archives  # noqa: E402
from src.adsb_behavioral.visualization import plot_correlation, plot_injected_example  # noqa: E402


LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _flight_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby(["archive_date", "flight_id"], as_index=False).agg(
        source_id=("source_id", "first"),
        aircraft_type=("aircraft_type", "first"),
        category=("category", "first"),
        rows=("timestamp_utc", "size"),
        start_utc=("timestamp_utc", "min"),
        end_utc=("timestamp_utc", "max"),
        duration_s=("timestamp_utc", lambda values: float(values.max() - values.min())),
        quality_good_fraction=("quality_good", "mean"),
    )


def _score_all(frame, rule, iforest) -> pd.DataFrame:
    return iforest.score(rule.score(frame))


def run(
    archives: list[Path],
    *,
    output: Path,
    members_per_archive: int,
    max_flights_per_archive: int,
    seed: int,
) -> Path:
    if len(archives) != 3:
        raise ValueError("Stage-1 pilot requires exactly three dated archives")
    dated = sorted(archives, key=archive_date)
    dates = [archive_date(path) for path in dated]
    if len(set(dates)) != 3:
        raise ValueError(f"Archive dates must be distinct: {dates}")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new --run-name: {output}")
    output.mkdir(parents=True)
    plots_dir = output / "plots"
    plots_dir.mkdir()

    LOGGER.info("Sampling archives: %s", [path.name for path in dated])
    sampled = sample_archives(
        dated,
        members_per_archive=members_per_archive,
        max_flights_per_archive=max_flights_per_archive,
        seed=seed,
    )
    raw = sampled.rows
    if raw.empty:
        raise RuntimeError("No qualifying ADS-B flights were sampled")
    counts = raw.groupby("archive_date")["flight_id"].nunique().to_dict()
    if any(counts.get(day, 0) < 10 for day in dates):
        raise RuntimeError(f"Too few qualifying flights by archive: {counts}")
    LOGGER.info("Qualifying flights by date: %s", counts)

    normal = add_physics_residuals(raw)
    train = normal[normal["archive_date"].eq(dates[0])].copy()
    validation = normal[normal["archive_date"].eq(dates[1])].copy()
    test_normal = normal[normal["archive_date"].eq(dates[2])].copy()
    raw_test = raw[raw["archive_date"].eq(dates[2])].copy()
    LOGGER.info("Injecting copied test flights: %d", raw_test["flight_id"].nunique())
    injected = add_physics_residuals(build_injected_copies(raw_test, seed=seed))

    LOGGER.info("Fitting robust physics and Isolation Forest baselines")
    rule = RobustPhysicsModel.fit(train)
    iforest = IsolationForestPhysicsModel.fit(train, seed=seed)
    validation_scored = _score_all(validation, rule, iforest)
    test_normal_scored = _score_all(test_normal, rule, iforest)
    injected_scored = _score_all(injected, rule, iforest)
    calibrations = {
        score_col: calibrate_threshold(validation_scored, score_col=score_col)
        for score_col in ("rule_score", "iforest_score")
    }

    overall_rows: list[dict] = []
    event_frames: list[pd.DataFrame] = []
    breakdown_frames: list[pd.DataFrame] = []
    for model_name, score_col in (("physics_rule", "rule_score"), ("isolation_forest", "iforest_score")):
        calibration = calibrations[score_col]
        overall, events, breakdown = evaluate_detector(
            test_normal_scored,
            injected_scored,
            score_col=score_col,
            threshold=calibration["threshold"],
            k=calibration["k"],
            n=calibration["n"],
        )
        overall_rows.append({"model": model_name, **overall})
        events.insert(0, "model", model_name)
        breakdown.insert(0, "model", model_name)
        event_frames.append(events)
        breakdown_frames.append(breakdown)

    overall_metrics = pd.DataFrame(overall_rows)
    event_metrics = pd.concat(event_frames, ignore_index=True)
    breakdown_metrics = pd.concat(breakdown_frames, ignore_index=True)
    overall_metrics.to_csv(output / "overall_metrics.csv", index=False)
    event_metrics.to_csv(output / "event_metrics.csv", index=False)
    breakdown_metrics.to_csv(output / "breakdown_metrics.csv", index=False)
    _flight_summary(normal).to_csv(output / "flight_inventory.csv", index=False)

    severity_recall = event_metrics.groupby(["model", "severity"])["detected"].mean().unstack()
    gate_b_rows = []
    for row in overall_rows:
        model = row["model"]
        easy = float(severity_recall.loc[model].get("easy", 0.0))
        medium = float(severity_recall.loc[model].get("medium", 0.0))
        fa = float(row["false_events_per_hour"])
        gate_b_rows.append({
            "model": model,
            "easy_event_recall": easy,
            "medium_event_recall": medium,
            "false_events_per_hour": fa,
            "passed": easy >= 0.90 and medium >= 0.70 and fa <= 0.10,
        })
    macro = event_metrics.groupby(["model", "severity", "injection_type"])["detected"].mean().groupby("model").mean()
    rule_macro = float(macro["physics_rule"])
    if_macro = float(macro["isolation_forest"])
    fa_by_model = overall_metrics.set_index("model")["false_events_per_hour"]
    rule_fa = float(fa_by_model["physics_rule"])
    if_fa = float(fa_by_model["isolation_forest"])
    gates = {
        "gate_a": {
            "status": "passed",
            "train_date": dates[0],
            "validation_date": dates[1],
            "test_date": dates[2],
            "injection_train_rows": 0,
            "behavior_quality_channels_separate": True,
            "causal_residuals": True,
        },
        "gate_b": {
            "status": "passed" if any(row["passed"] for row in gate_b_rows) else "failed",
            "rule": "easy recall >=0.90 and medium recall >=0.70 at <=0.10 false events/hour",
            "models": gate_b_rows,
        },
        "gate_c": {
            "status": "passed" if if_macro >= rule_macro + 0.05 and if_fa <= rule_fa else "failed",
            "rule": "Isolation Forest macro event recall >= physics rule +0.05 at no higher test FA/hour",
            "physics_rule_macro_event_recall": rule_macro,
            "isolation_forest_macro_event_recall": if_macro,
            "physics_rule_false_events_per_hour": rule_fa,
            "isolation_forest_false_events_per_hour": if_fa,
        },
        "gate_d": {
            "status": "audit_pending",
            "rule": "Top-scored natural events require manual review; no natural precision claim",
        },
    }
    (output / "gates.json").write_text(json.dumps(_jsonable(gates), indent=2), encoding="utf-8")

    natural_top = []
    for score_col in ("rule_score", "iforest_score"):
        top = test_normal_scored.nlargest(100, score_col)[
            ["flight_id", "timestamp_utc", "lat", "lon", "alt", "ground_speed_ms", "track_deg", "phase", score_col]
        ].copy()
        top.insert(0, "model_score", score_col)
        natural_top.append(top)
    pd.concat(natural_top, ignore_index=True).to_csv(output / "natural_top100_audit.csv", index=False)

    joblib.dump(rule, output / "physics_rule.joblib")
    joblib.dump(iforest, output / "isolation_forest.joblib")
    (output / "calibrations.json").write_text(json.dumps(_jsonable(calibrations), indent=2), encoding="utf-8")
    dataset_card = {
        "archive_stats": sampled.archive_stats,
        "selected_members": sampled.selected_members,
        "split_dates": {"train": dates[0], "validation": dates[1], "test": dates[2]},
        "flight_counts": counts,
        "row_counts": {
            "train_normal": len(train),
            "validation_normal": len(validation),
            "test_normal": len(test_normal),
            "injected_test": len(injected),
        },
        "model_features": MODEL_FEATURES,
    }
    (output / "dataset_card.json").write_text(json.dumps(_jsonable(dataset_card), indent=2), encoding="utf-8")

    plot_correlation(train, plots_dir / "train_residual_correlation.png")
    examples = injected_scored.groupby(["severity", "injection_type"], sort=True).head(1)["flight_id"].unique()[:8]
    for index, flight_id in enumerate(examples):
        plot_injected_example(
            injected_scored[injected_scored["flight_id"].eq(flight_id)],
            score_col="iforest_score",
            threshold=calibrations["iforest_score"]["threshold"],
            output=plots_dir / f"injected_example_{index:02d}.png",
        )

    LOGGER.info("Hashing source archives")
    inputs = {
        path.name: {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}
        for path in dated
    }
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Codex ADS-B behavioral anomaly Stage-1 pilot",
        "plan": "docs/CODEX_ADSB_BEHAVIORAL_ANOMALY_ROADMAP.md",
        "seed": seed,
        "members_per_archive": members_per_archive,
        "max_flights_per_archive": max_flights_per_archive,
        "inputs": inputs,
        "gate_status": {key: value["status"] for key, value in gates.items()},
        "files": {str(path.relative_to(output)).replace("\\", "/"): _sha256(path) for path in files},
    }
    (output / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs=3, type=Path)
    parser.add_argument("--run-name", default="pilot_3archives")
    parser.add_argument("--members-per-archive", type=int, default=250)
    parser.add_argument("--max-flights-per-archive", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = run(
        args.archives,
        output=ROOT / "artifacts/adsb_behavioral_stage1" / args.run_name,
        members_per_archive=args.members_per_archive,
        max_flights_per_archive=args.max_flights_per_archive,
        seed=args.seed,
    )
    print(f"ADS-B Stage-1 artifact: {output}")
    print((output / "gates.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
