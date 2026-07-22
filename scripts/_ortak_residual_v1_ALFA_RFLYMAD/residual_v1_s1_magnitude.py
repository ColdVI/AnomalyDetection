from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gecmis_calismalar.residual_v1.eval.sanity_gates import s1_magnitude_gate
from gecmis_calismalar.residual_v1.ingest.common import write_json
from gecmis_calismalar.residual_v1.run import create_run_dir, update_manifest
from gecmis_calismalar.residual_v1.tracking import log_run


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RESIDUAL-V1 S-1 magnitude gate")
    parser.add_argument("--scaling-run", required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    scaling_run = Path(args.scaling_run)
    scaling_summary = _read_json(scaling_run / "summary.json")
    scalers = _read_json(scaling_run / "scalers.json")
    if not bool(scaling_summary.get("development_only")):
        raise ValueError("S-1 requires a development-only scaling run")

    run_dir, _ = create_run_dir(
        f"phaseE_s1_magnitude_seed{args.seed}",
        seed=args.seed,
        input_paths=[scaling_run / "summary.json", scaling_run / "scalers.json"],
    )
    (run_dir / "channel_reports").mkdir()
    reports = {}
    eligible = {"alfa": [], "rfly": []}
    flagged = {"alfa": [], "rfly": []}
    for dataset, channels in scaling_summary["active_channels"].items():
        for channel in channels:
            frame = pd.read_parquet(scaling_run / "scaled" / dataset / f"{channel}.parquet")
            result = s1_magnitude_gate(frame, dataset=dataset, channel=channel)
            report = result.to_dict()
            report["input_magnitude_definition"] = scalers[channel]["input_magnitude"]
            reports[channel] = report
            write_json(
                run_dir / "channel_reports" / f"{channel}.json",
                report,
                fail_if_exists=True,
            )
            if result.status == "passed":
                eligible[dataset].append(channel)
            elif result.status == "flagged":
                flagged[dataset].append(channel)

    flags = {
        "gate": "S-1",
        "development_only": True,
        "rho_threshold": 0.5,
        "channels": reports,
        "decision_eligible_channels": {key: sorted(value) for key, value in eligible.items()},
        "flagged_channels": {key: sorted(value) for key, value in flagged.items()},
    }
    write_json(run_dir / "flags.json", flags, fail_if_exists=True)
    write_json(
        run_dir / "summary.json",
        {
            "seed": args.seed,
            "development_only": True,
            "scaling_run": str(scaling_run),
            "decision_eligible_channels": flags["decision_eligible_channels"],
            "flagged_channels": flags["flagged_channels"],
        },
        fail_if_exists=True,
    )
    lines = [
        "# RESIDUAL-V1 S-1 — Development Magnitude Gate",
        "",
        "| Veri | Kanal | Girdi büyüklüğü | Spearman rho | Eşik | Durum |",
        "|---|---|---|---:|---:|---:|",
    ]
    for channel, report in reports.items():
        lines.append(
            f"| {report['dataset']} | {channel} | {report['input_magnitude_definition']} | "
            f"{report['metrics'].get('spearman_rho', '—')} | 0.5 | {report['status']} |"
        )
    (run_dir / "S1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tracking = log_run(
        run_dir,
        run_name="phaseE_s1_magnitude",
        metrics={
            "passed_channels": sum(len(value) for value in eligible.values()),
            "flagged_channels": sum(len(value) for value in flagged.values()),
        },
        params={"seed": args.seed, "rho_threshold": 0.5},
    )
    update_manifest(
        run_dir,
        development_only=True,
        scaling_run=str(scaling_run),
        mlflow_status=tracking["status"],
    )
    print(run_dir)


if __name__ == "__main__":
    main()
