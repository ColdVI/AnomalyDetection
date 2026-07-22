from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gecmis_calismalar.residual_v1.run import create_run_dir, update_manifest
from gecmis_calismalar.residual_v1.tracking import log_run
from gecmis_calismalar.residual_v1.viz.handout import build_handout, create_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-aware RESIDUAL-V1 Claude handout")
    parser.add_argument("--alfa-root", default="artifacts/residual_v1/silver/alfa")
    parser.add_argument("--rfly-root", default="artifacts/residual_v1/silver/rfly")
    parser.add_argument("--alfa-split", default="artifacts/residual_v1/splits/alfa_seed11.json")
    parser.add_argument("--rfly-split", default="artifacts/residual_v1/splits/rfly_seed11.json")
    parser.add_argument("--sanity-run", default="artifacts/residual_v1/runs/20260717_063051_sanity")
    parser.add_argument("--run-root", default="artifacts/residual_v1/runs")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    docs = [
        Path("docs/RESIDUAL_V1_DENEY_TASARIMI.md"),
        Path("docs/RESIDUAL_V1_IMPLEMENTASYON_TALIMATI.md"),
    ]
    split_paths = [Path(args.alfa_split), Path(args.rfly_split)]
    run_dir, _ = create_run_dir(
        "full_flight_handout",
        seed=args.seed,
        input_paths=[*split_paths, *(Path(args.sanity_run) / name for name in ("sanity_metrics.json", "SANITY_REPORT.md")), *docs],
        run_root=args.run_root,
    )
    handout = run_dir / "handout"
    try:
        summary = build_handout(
            output=handout,
            alfa_root=Path(args.alfa_root),
            rfly_root=Path(args.rfly_root),
            alfa_split_path=split_paths[0],
            rfly_split_path=split_paths[1],
            evidence_root=Path(args.run_root),
            sanity_run=Path(args.sanity_run),
            docs=docs,
            progress=lambda message: print(message, flush=True),
        )
        zip_path = run_dir / "RESIDUAL_V1_CLAUDE_HANDOUT.zip"
        create_zip(handout, zip_path)
        tracking = log_run(
            run_dir,
            run_name="full_flight_handout",
            metrics={
                "visible_flight_count": summary["visible_flight_count"],
                "sealed_holdout_flight_count": summary["sealed_holdout_flight_count"],
                "flight_plot_count": summary["flight_plot_count"],
            },
            params={"holdout_opened_count": 0, "seed": args.seed},
        )
        update_manifest(
            run_dir,
            status="complete_task_3_5_handout",
            handout_root=str(handout),
            zip_path=str(zip_path),
            zip_size_bytes=zip_path.stat().st_size,
            holdout_opened_count=0,
            mlflow_status=tracking["status"],
        )
    except Exception as error:
        update_manifest(run_dir, status="failed", failure_reason=f"{type(error).__name__}: {error}")
        raise
    print(run_dir)


if __name__ == "__main__":
    main()
