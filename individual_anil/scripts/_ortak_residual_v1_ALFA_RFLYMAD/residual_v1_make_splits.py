from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gecmis_calismalar.residual_v1.ingest.common import write_json
from gecmis_calismalar.residual_v1.ingest.splits import load_flight_metadata, write_split_manifests
from gecmis_calismalar.residual_v1.run import create_run_dir, update_manifest
from gecmis_calismalar.residual_v1.tracking import log_run


def main() -> None:
    parser = argparse.ArgumentParser(description="RESIDUAL-V1 session-level splits")
    parser.add_argument("--dataset", choices=("alfa", "rfly"), required=True)
    parser.add_argument("--silver-root")
    args = parser.parse_args()
    silver_root = args.silver_root or f"artifacts/residual_v1/silver/{args.dataset}"
    rows = load_flight_metadata(silver_root)
    run_dir, _ = create_run_dir(f"phaseA_splits_{args.dataset}", seed=11)
    hashes = write_split_manifests(args.dataset, rows)
    summary = {"dataset": args.dataset, "flight_count": len(rows), "manifest_hashes": hashes}
    write_json(run_dir / "summary.json", summary, fail_if_exists=True)
    tracking = log_run(
        run_dir,
        run_name=f"phaseA_splits_{args.dataset}",
        metrics={"flight_count": len(rows), "seed_count": len(hashes)},
    )
    update_manifest(run_dir, split_manifest_hashes=hashes, mlflow_status=tracking["status"])


if __name__ == "__main__":
    main()
