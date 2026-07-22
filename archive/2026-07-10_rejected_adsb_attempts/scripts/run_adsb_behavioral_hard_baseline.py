"""Run the corrected instant hard-physics ADS-B baseline on the frozen V1 sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adsb_behavioral.evaluation import evaluate_detector  # noqa: E402
from src.adsb_behavioral.hard_rules import HARD_LIMITS, add_hard_rule_score  # noqa: E402
from src.adsb_behavioral.injection import build_injected_copies  # noqa: E402
from src.adsb_behavioral.physics_residuals import add_physics_residuals  # noqa: E402
from src.adsb_behavioral.reader import archive_date, sample_archives  # noqa: E402
from src.adsb_behavioral.visualization import plot_injected_example  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(archives: list[Path], output: Path) -> Path:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plots = output / "plots"
    plots.mkdir()
    dated = sorted(archives, key=archive_date)
    dates = [archive_date(path) for path in dated]
    sampled = sample_archives(
        dated,
        members_per_archive=150,
        max_flights_per_archive=250,
        seed=20260710,
    )
    normal = add_hard_rule_score(add_physics_residuals(sampled.rows))
    test_normal = normal[normal["archive_date"].eq(dates[2])].copy()
    raw_test = sampled.rows[sampled.rows["archive_date"].eq(dates[2])].copy()
    injected = add_hard_rule_score(
        add_physics_residuals(build_injected_copies(raw_test, seed=20260710))
    )
    overall, events, breakdown = evaluate_detector(
        test_normal,
        injected,
        score_col="hard_rule_score",
        threshold=1.0,
        k=1,
        n=1,
    )
    overall_frame = pd.DataFrame([overall])
    overall_frame.to_csv(output / "overall_metrics.csv", index=False)
    events.to_csv(output / "event_metrics.csv", index=False)
    breakdown.to_csv(output / "breakdown_metrics.csv", index=False)

    easy = events[events["severity"].eq("easy")]
    obvious_types = {
        "position_jump", "altitude_bias", "speed_bias", "track_bias",
        "vertical_rate_bias", "freeze",
    }
    obvious = easy[easy["injection_type"].isin(obvious_types)]
    obvious_breakdown = obvious.groupby("injection_type")["detected"].mean().to_dict()
    minimum_obvious = min(obvious_breakdown.values()) if obvious_breakdown else 0.0
    gates = {
        "hard_sanity": {
            "status": "passed" if minimum_obvious >= 0.95 else "failed",
            "rule": "each obvious easy anomaly type recall >=0.95",
            "recall_by_type": obvious_breakdown,
        },
        "all_easy": {
            "status": "reported",
            "event_recall": float(easy["detected"].mean()),
        },
        "all_medium": {
            "status": "reported",
            "event_recall": float(events[events["severity"].eq("medium")]["detected"].mean()),
        },
        "natural_alarm_rate": {
            "status": "reported",
            "false_events_per_hour_under_injection_ground_truth": overall["false_events_per_hour"],
            "note": "Natural hard violations may be real data-integrity anomalies; manual audit required",
        },
    }
    (output / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    (output / "hard_limits.json").write_text(json.dumps(HARD_LIMITS, indent=2), encoding="utf-8")

    examples = injected.groupby(["severity", "injection_type"], sort=True).head(1)["flight_id"].unique()[:8]
    for index, flight_id in enumerate(examples):
        plot_injected_example(
            injected[injected["flight_id"].eq(flight_id)],
            score_col="hard_rule_score",
            threshold=1.0,
            output=plots / f"hard_rule_example_{index:02d}.png",
        )

    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "corrected hard-physics implementation sanity baseline",
        "not_independent_final_test": True,
        "reason": "2026-03-16 was already observed in V1; this run validates implementation sanity only",
        "archive_dates": dates,
        "source_archive_sha256_from_v1": json.loads(
            (ROOT / "artifacts/adsb_behavioral_stage1/pilot_3archives_v1/manifest.json").read_text()
        )["inputs"],
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): _sha256(path) for path in files
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs=3, type=Path)
    parser.add_argument("--run-name", default="hard_physics_sanity_v2")
    args = parser.parse_args()
    output = run(
        args.archives,
        ROOT / "artifacts/adsb_behavioral_stage1" / args.run_name,
    )
    print(output)
    print((output / "gates.json").read_text())


if __name__ == "__main__":
    main()
