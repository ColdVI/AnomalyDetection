"""Run the decision-frozen UAV GNSS-integrity feasibility pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_gnss.pipeline import PilotRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "preflight",
            "fit",
            "calibrate",
            "develop",
            "rehearse",
            "holdout",
            "report",
        ],
    )
    parser.add_argument("--config", default="configs/uav_gnss_integrity_v1.json")
    args = parser.parse_args()
    runner = PilotRunner(args.config)
    if args.stage == "preflight":
        result = runner.preflight()
    elif args.stage == "report":
        result = {"latex_report": runner.report().as_posix()}
    else:
        result = runner.run_through(args.stage)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

