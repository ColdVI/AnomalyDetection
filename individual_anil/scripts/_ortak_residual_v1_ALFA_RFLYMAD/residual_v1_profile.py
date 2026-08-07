from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gecmis_calismalar.residual_v1.ingest.profile import profile_dataset
from gecmis_calismalar.residual_v1.ingest.common import write_json
from gecmis_calismalar.residual_v1.run import create_run_dir, update_manifest
from gecmis_calismalar.residual_v1.tracking import log_run


def main() -> None:
    parser = argparse.ArgumentParser(description="RESIDUAL-V1 Phase-A hygiene profile")
    parser.add_argument("--dataset", choices=("alfa", "rfly"), required=True)
    parser.add_argument("--silver-root")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    silver_root = args.silver_root or f"artifacts/residual_v1/silver/{args.dataset}"
    run_dir, _ = create_run_dir(f"phaseA_profile_{args.dataset}", seed=args.seed)
    summary = profile_dataset(
        silver_root,
        run_dir / "profiles",
        dataset=args.dataset,
    )
    write_json(run_dir / "summary.json", summary, fail_if_exists=True)
    tracking = log_run(
        run_dir,
        run_name="phaseA_profile",
        metrics={
            "flight_count": summary["flight_count"],
            "quarantine_count": summary["quarantine_count"],
        },
        params={"dataset": args.dataset},
    )
    update_manifest(run_dir, silver_root=silver_root, mlflow_status=tracking["status"])


if __name__ == "__main__":
    main()
