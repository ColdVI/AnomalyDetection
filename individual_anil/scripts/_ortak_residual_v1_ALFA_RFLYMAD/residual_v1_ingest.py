from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gecmis_calismalar.residual_v1.ingest.alfa import ingest_alfa
from gecmis_calismalar.residual_v1.ingest.rfly import ingest_rfly
from gecmis_calismalar.residual_v1.ingest.common import write_json
from gecmis_calismalar.residual_v1.run import create_run_dir, update_manifest
from gecmis_calismalar.residual_v1.tracking import log_run


def main() -> None:
    parser = argparse.ArgumentParser(description="RESIDUAL-V1 natural-rate ingestion")
    parser.add_argument("--dataset", choices=("alfa", "rfly"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    config_paths = ["configs/residual_v1_rfly_exclusions.json"] if args.dataset == "rfly" else []
    source_path = Path(args.source)
    input_paths = [source_path] if source_path.is_file() else []
    if args.dataset == "rfly" and (source_path / "manifest.json").is_file():
        input_paths = [source_path / "manifest.json"]
    run_dir, _ = create_run_dir(
        f"phaseA_ingest_{args.dataset}",
        seed=args.seed,
        config_paths=config_paths,
        input_paths=input_paths,
    )
    if args.dataset == "alfa":
        summary = ingest_alfa(args.source, args.output)
    else:
        summary = ingest_rfly(args.source, args.output, limit=args.limit)
    write_json(run_dir / "summary.json", summary, fail_if_exists=True)
    tracking = log_run(
        run_dir,
        run_name=f"phaseA_ingest_{args.dataset}",
        metrics={"flight_count": summary["flight_count"], "event_count": summary["event_count"]},
    )
    update_manifest(run_dir, output_root=args.output, mlflow_status=tracking["status"])


if __name__ == "__main__":
    main()
