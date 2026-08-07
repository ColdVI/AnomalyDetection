from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gecmis_calismalar.residual_v1.eval.sanity_gates import GateError, s3_event_separation_gate
from gecmis_calismalar.residual_v1.features.spec import ALFA_SPECS
from gecmis_calismalar.residual_v1.ingest.common import write_json
from gecmis_calismalar.residual_v1.run import create_run_dir, update_manifest
from gecmis_calismalar.residual_v1.tracking import log_run


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _event_catalog(split_path: Path, silver_root: Path) -> dict[str, list[dict]]:
    split = _read_json(split_path)
    catalog: dict[str, list[dict]] = {}
    for flight_id in split["partitions"]["development"]["flight_ids"]:
        for event in _read_json(silver_root / Path(flight_id) / "events.json"):
            fault_class = str(event["fault_class"])
            catalog.setdefault(fault_class, []).append(
                {"flight_id": flight_id, "onset_s": float(event["onset_s"])}
            )
    return catalog


def _class_status(channel_reports: dict[str, dict]) -> str:
    evaluated = [report for report in channel_reports.values() if report["status"] in {"passed", "failed"}]
    if any(report["status"] == "passed" for report in evaluated):
        return "passed"
    if evaluated:
        return "failed"
    return "not_evaluable"


def _plot_worst_events(
    *,
    destination: Path,
    dataset: str,
    fault_class: str,
    channel_frames: dict[str, pd.DataFrame],
    events: list[dict],
    channel_reports: dict[str, dict],
) -> list[str]:
    shifts: dict[tuple[str, float], list[float]] = {}
    for report in channel_reports.values():
        for event in report.get("metrics", {}).get("event_metrics", []):
            if event["median_shift"] is not None:
                shifts.setdefault((event["flight_id"], float(event["onset_s"])), []).append(
                    float(event["median_shift"])
                )
    ranked = sorted(events, key=lambda event: max(shifts.get((event["flight_id"], float(event["onset_s"])), [-np.inf])))[:3]
    paths = []
    destination.mkdir(parents=True, exist_ok=True)
    for rank, event in enumerate(ranked, start=1):
        flight_id = event["flight_id"]
        onset = float(event["onset_s"])
        fig, axes = plt.subplots(len(channel_frames), 1, figsize=(11, 3.2 * len(channel_frames)), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, (channel, frame) in zip(axes, channel_frames.items(), strict=True):
            flight = frame.loc[frame["flight_id"].astype(str) == flight_id].copy()
            relative = pd.to_numeric(flight["t"], errors="coerce") - onset
            visible = (relative >= -65.0) & (relative <= 20.0)
            axis.plot(relative.loc[visible], pd.to_numeric(flight.loc[visible, "z"], errors="coerce").abs(), linewidth=1.0)
            axis.axvspan(-60.0, -10.0, color="#4C78A8", alpha=0.12, label="pre")
            axis.axvspan(0.0, 15.0, color="#E45756", alpha=0.12, label="post")
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            axis.set_ylabel(f"{channel}\n|z|")
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("onset'e gÃ¶re zaman (s)")
        fig.suptitle(f"S-3 worst {rank}: {dataset}/{fault_class} â€” {flight_id}")
        fig.tight_layout()
        safe_id = flight_id.replace("/", "__").replace(chr(92), "__")
        path = destination / f"{dataset}_{fault_class}_worst_{rank}_{safe_id}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RESIDUAL-V1 S-3 event separation gate")
    parser.add_argument("--scaling-run", required=True)
    parser.add_argument("--s1-run", required=True)
    parser.add_argument("--alfa-silver-root", required=True)
    parser.add_argument("--rfly-silver-root", required=True)
    parser.add_argument("--alfa-split", default="artifacts/residual_v1/splits/alfa_seed11.json")
    parser.add_argument("--rfly-split", default="artifacts/residual_v1/splits/rfly_seed11.json")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    scaling_run = Path(args.scaling_run)
    s1_run = Path(args.s1_run)
    scaling_summary = _read_json(scaling_run / "summary.json")
    s1_summary = _read_json(s1_run / "summary.json")
    if not bool(scaling_summary.get("development_only")) or not bool(s1_summary.get("development_only")):
        raise ValueError("S-3 requires development-only scaling and S-1 runs")
    if Path(s1_summary["scaling_run"]) != scaling_run:
        raise ValueError("S-1 run does not belong to the supplied scaling run")

    alfa_events = _event_catalog(Path(args.alfa_split), Path(args.alfa_silver_root))
    rfly_events = _event_catalog(Path(args.rfly_split), Path(args.rfly_silver_root))
    run_dir, _ = create_run_dir(
        f"phaseE_s3_separation_seed{args.seed}",
        seed=args.seed,
        input_paths=[
            scaling_run / "summary.json",
            s1_run / "summary.json",
            Path(args.alfa_split),
            Path(args.rfly_split),
        ],
    )
    (run_dir / "channel_reports").mkdir()
    diagnostics = run_dir / "diagnostics"
    class_results: dict[str, dict[str, dict]] = {"alfa": {}, "rfly": {}}
    frame_cache: dict[tuple[str, str], pd.DataFrame] = {}

    definitions = {
        "alfa": {"engine": alfa_events.get("engine", [])},
        "rfly": {
            "motor": rfly_events.get("motor", []),
            "sensor": rfly_events.get("sensor", []),
        },
    }
    for dataset, classes in definitions.items():
        eligible = s1_summary["decision_eligible_channels"][dataset]
        for fault_class, events in classes.items():
            channel_reports: dict[str, dict] = {}
            if dataset == "alfa":
                for spec in ALFA_SPECS[:5]:
                    channel_reports[spec.name] = {
                        "gate": "S-3",
                        "dataset": "alfa",
                        "channel": spec.name,
                        "status": "not_evaluable",
                        "reason": "model_unavailable",
                        "metrics": {"fault_class": fault_class},
                    }
            for channel in eligible:
                frame = pd.read_parquet(scaling_run / "scaled" / dataset / f"{channel}.parquet")
                frame_cache[(dataset, channel)] = frame
                channel_reports[channel] = s3_event_separation_gate(
                    frame,
                    dataset=dataset,
                    channel=channel,
                    fault_class=fault_class,
                    events=events,
                ).to_dict()
                write_json(
                    run_dir / "channel_reports" / f"{dataset}_{fault_class}_{channel}.json",
                    channel_reports[channel],
                    fail_if_exists=True,
                )
            status = _class_status(channel_reports)
            class_result = {
                "dataset": dataset,
                "fault_class": fault_class,
                "status": status,
                "decision_rule": "at_least_one_evaluable_channel_ks_pvalue_below_0.01",
                "channels": channel_reports,
                "diagnostic_plots": [],
            }
            if status != "passed":
                active_frames = {
                    channel: frame_cache[(dataset, channel)]
                    for channel in eligible
                    if (dataset, channel) in frame_cache
                }
                if active_frames:
                    class_result["diagnostic_plots"] = _plot_worst_events(
                        destination=diagnostics,
                        dataset=dataset,
                        fault_class=fault_class,
                        channel_frames=active_frames,
                        events=events,
                        channel_reports=channel_reports,
                    )
            class_results[dataset][fault_class] = class_result

    write_json(run_dir / "class_results.json", class_results, fail_if_exists=True)
    summary = {
        "seed": args.seed,
        "development_only": True,
        "scaling_run": str(scaling_run),
        "s1_run": str(s1_run),
        "class_status": {
            dataset: {fault_class: result["status"] for fault_class, result in classes.items()}
            for dataset, classes in class_results.items()
        },
        "combined_status_intentionally_omitted": True,
    }
    write_json(run_dir / "summary.json", summary, fail_if_exists=True)
    failures = [
        (dataset, fault_class, result["status"])
        for dataset, classes in class_results.items()
        for fault_class, result in classes.items()
        if result["status"] != "passed"
    ]
    if failures:
        lines = [
            "# RESIDUAL-V1 S-3 FAILURE REPORT",
            "",
            "Kalibrasyon kilitlidir. SÄ±nÄ±flar birleÅŸtirilmedi; RFLY sonucu ALFA'ya taÅŸÄ±nmadÄ±.",
            "",
            "| Veri | Headline sÄ±nÄ±f | Durum | Kanal | KS | p |",
            "|---|---|---|---|---:|---:|",
        ]
        for dataset, classes in class_results.items():
            for fault_class, class_result in classes.items():
                for channel, report in class_result["channels"].items():
                    metrics = report.get("metrics", {})
                    lines.append(
                        f"| {dataset} | {fault_class} | {report['status']} | {channel} | "
                        f"{metrics.get('ks_statistic', 'â€”')} | {metrics.get('ks_pvalue', 'â€”')} |"
                    )
        lines.extend(["", "## TanÄ± grafikleri", ""])
        for dataset, classes in class_results.items():
            for fault_class, class_result in classes.items():
                for path in class_result["diagnostic_plots"]:
                    lines.append(f"- `{path}`")
        (run_dir / "S3_FAILURE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tracking = log_run(
        run_dir,
        run_name="phaseE_s3_separation",
        metrics={"failed_or_not_evaluable_classes": len(failures)},
        params={"seed": args.seed, "ks_pvalue_threshold": 0.01},
    )
    update_manifest(
        run_dir,
        development_only=True,
        class_status=summary["class_status"],
        calibration_locked=bool(failures),
        mlflow_status=tracking["status"],
    )
    print(run_dir)
    if failures:
        raise GateError(f"S-3 failed; calibration locked; class statuses={summary['class_status']}")


if __name__ == "__main__":
    main()

