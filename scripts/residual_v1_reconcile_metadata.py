from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from residual_v1.ingest.common import write_json
from residual_v1.ingest.rfly import reconcile_rfly_fault_classes
from residual_v1.run import create_run_dir, update_manifest
from residual_v1.tracking import log_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RESIDUAL-V1 metadata reconciliation")
    parser.add_argument("--dataset", choices=("rfly",), required=True)
    parser.add_argument("--silver-root", default="artifacts/residual_v1/silver/rfly")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    report_path = Path(args.silver_root) / "dataset_report.json"
    run_dir, _ = create_run_dir(
        "phaseA_reconcile_rfly_sensor_class",
        seed=args.seed,
        input_paths=[report_path],
    )
    result = reconcile_rfly_fault_classes(args.silver_root)
    write_json(run_dir / "reconciliation.json", result, fail_if_exists=True)
    tracking = log_run(
        run_dir,
        run_name="phaseA_reconcile_rfly_sensor_class",
        metrics={"changed_flights": result["changed_count"]},
    )
    update_manifest(
        run_dir,
        status="complete",
        correction="Real-Sensors subtypes aggregated to pre-registered sensor headline class",
        mlflow_status=tracking["status"],
    )


if __name__ == "__main__":
    main()
