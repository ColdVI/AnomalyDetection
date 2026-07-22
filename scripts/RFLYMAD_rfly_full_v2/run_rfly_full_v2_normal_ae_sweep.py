import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gecmis_calismalar.rfly_full.normal_ae import OUTPUT_ROOT, run
from gecmis_calismalar.rfly_full.pipeline import _atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotations", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    sweep = OUTPUT_ROOT / datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
    sweep.mkdir(parents=True, exist_ok=False)
    completed = []
    metric_frames = []
    for rotation in args.rotations:
        output = run(
            epochs=args.epochs,
            torch_threads=args.torch_threads,
            validation_rotation=rotation,
        )
        completed.append({"rotation": rotation, "output": str(output)})
        metrics = pd.read_csv(output / "operational_metrics.csv")
        metrics.insert(0, "validation_rotation", rotation)
        metrics.insert(1, "run", output.name)
        metric_frames.append(metrics)
        _atomic_json(sweep / "progress.json", {
            "status": "running", "completed": completed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_metrics.to_csv(sweep / "all_metrics.csv", index=False)
    aggregate = (
        all_metrics.groupby("policy")[[
            "event_recall", "normal_validation_fa_per_hour",
            "all_nonfault_fa_per_hour", "environment_fa_per_hour",
        ]]
        .agg(["mean", "std", "min", "max"])
    )
    aggregate.to_csv(sweep / "aggregate_metrics.csv")
    _atomic_json(sweep / "progress.json", {
        "status": "complete", "completed": completed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"sweep": str(sweep), "runs": completed}, indent=2))


if __name__ == "__main__":
    main()
