from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from residual_v1.features.align import align_to_clock, default_tolerances, observed_tolerances
from residual_v1.features.waypoints import label_waypoint_boundaries
from residual_v1.ingest.common import write_json


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _k5_gallery(output: Path, split_path: Path, silver_root: Path) -> list[dict]:
    ids = _read_json(split_path)["partitions"]["development"]["flight_ids"]
    events = []
    aligned_by_flight = {}
    for flight_id in ids:
        root = silver_root / Path(flight_id)
        topics = {
            name: pd.read_parquet(root / f"{name}.parquet")
            for name in ("mavros-nav_info-roll", "mavros-nav_info-errors")
        }
        aligned = align_to_clock(
            topics,
            "mavros-nav_info-roll",
            observed_tolerances(topics, default_tolerances("alfa")),
        )
        labels = label_waypoint_boundaries(aligned)
        aligned_by_flight[flight_id] = aligned
        for timestamp in aligned.loc[labels["waypoint_event"], "t"].astype(float):
            events.append({"flight_id": flight_id, "event_time_s": timestamp})
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for axis, event in zip(axes.ravel(), events, strict=True):
        frame = aligned_by_flight[event["flight_id"]]
        relative = pd.to_numeric(frame["t"]) - event["event_time_s"]
        visible = relative.abs() <= 6.0
        axis.plot(relative.loc[visible], frame.loc[visible, "waypoint_distance"], color="#4C78A8")
        axis.axvspan(-2.0, 2.0, color="#F2CF5B", alpha=0.25)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        twin = axis.twinx()
        twin.plot(relative.loc[visible], frame.loc[visible, "xtrack_error"], color="#E45756", alpha=0.65)
        axis.set_title(Path(event["flight_id"]).name, fontsize=9)
        axis.set_xlabel("relative time (s)")
        axis.set_ylabel("waypoint distance (m)", color="#4C78A8")
        twin.set_ylabel("xtrack (m)", color="#E45756")
    fig.suptitle("K5 development-only waypoint V-turns (yellow: frozen ±2 s mask)")
    fig.tight_layout()
    fig.savefig(output / "01_K5_WAYPOINT_V_TURNS.png", dpi=160)
    plt.close(fig)
    return events


def _bar_gate(output: Path, reports: list[tuple[str, float]], threshold: float, title: str, name: str) -> None:
    labels = [label for label, _ in reports]
    values = [value for _, value in reports]
    fig, axis = plt.subplots(figsize=(10, 5))
    colors = ["#E45756" if value < threshold else "#59A14F" for value in values]
    axis.bar(labels, values, color=colors)
    axis.axhline(threshold, color="black", linestyle="--", label=f"threshold={threshold}")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=15)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / name, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a read-only Phase E handout for Claude")
    parser.add_argument("--s4-run", required=True)
    parser.add_argument("--scaling-run", required=True)
    parser.add_argument("--s1-run", required=True)
    parser.add_argument("--s3-run", required=True)
    parser.add_argument("--calibration-failure-run", required=True)
    parser.add_argument("--alfa-silver-root", required=True)
    parser.add_argument("--alfa-split", default="artifacts/residual_v1/splits/alfa_seed11.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    s4_run = Path(args.s4_run)
    scaling_run = Path(args.scaling_run)
    s1_run = Path(args.s1_run)
    s3_run = Path(args.s3_run)
    calibration_run = Path(args.calibration_failure_run)

    k5_events = _k5_gallery(output, Path(args.alfa_split), Path(args.alfa_silver_root))
    s4 = _read_json(s4_run / "flags.json")
    s4_values = [
        (channel.split("_")[0], report["variance_ratio"])
        for channel, report in s4["channels"].items()
        if "variance_ratio" in report
    ]
    _bar_gate(
        output,
        s4_values,
        1.15,
        "S-4 command ablation: Var(crippled) / Var(full)",
        "02_S4_ABLATION.png",
    )
    s1 = _read_json(s1_run / "flags.json")
    s1_values = [
        (channel.split("_")[0], report["metrics"]["spearman_rho"])
        for channel, report in s1["channels"].items()
    ]
    _bar_gate(
        output,
        [(label, 0.5 - value) for label, value in s1_values],
        0.0,
        "S-1 margin below rho=0.5 (positive margin passes)",
        "03_S1_MAGNITUDE.png",
    )

    s3 = _read_json(s3_run / "class_results.json")
    ks_rows = []
    alfa_shifts = []
    rfly_shifts = []
    for dataset, classes in s3.items():
        for fault_class, class_report in classes.items():
            for channel, report in class_report["channels"].items():
                metrics = report.get("metrics", {})
                if "ks_statistic" not in metrics:
                    continue
                label = f"{dataset}/{fault_class}/{channel.split('_')[0]}"
                ks_rows.append((label, metrics["ks_statistic"]))
                for event in metrics["event_metrics"]:
                    if event["median_shift"] is not None:
                        target = alfa_shifts if dataset == "alfa" else rfly_shifts
                        target.append(
                            {
                                "class_channel": f"{fault_class}/{channel.split('_')[0]}",
                                "flight_id": event["flight_id"],
                                "median_shift": event["median_shift"],
                            }
                        )
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar([label for label, _ in ks_rows], [value for _, value in ks_rows], color="#4C78A8")
    axis.set_ylabel("KS statistic")
    axis.set_title("S-3 pooled pre/post |z| separation (all headline classes PASS)")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "04_S3_KS.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    alfa_frame = pd.DataFrame(alfa_shifts).sort_values("median_shift")
    axis.barh([Path(value).name for value in alfa_frame["flight_id"]], alfa_frame["median_shift"], color=np.where(alfa_frame["median_shift"] >= 0, "#59A14F", "#E45756"))
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("post median |z| − pre median |z|")
    axis.set_title("ALFA engine/R6 event heterogeneity (pooled KS passes; individual events vary)")
    fig.tight_layout()
    fig.savefig(output / "05_ALFA_ENGINE_EVENT_SHIFTS.png", dpi=160)
    plt.close(fig)

    rfly_frame = pd.DataFrame(rfly_shifts)
    groups = sorted(rfly_frame["class_channel"].unique())
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(
        [rfly_frame.loc[rfly_frame["class_channel"] == group, "median_shift"] for group in groups],
        showfliers=False,
    )
    axis.set_xticks(np.arange(1, len(groups) + 1), groups)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("post median |z| − pre median |z|")
    axis.set_title("RFLY event-level shift distributions")
    fig.tight_layout()
    fig.savefig(output / "06_RFLY_EVENT_SHIFTS.png", dpi=160)
    plt.close(fig)

    coverage = _read_json(calibration_run / "summary.json")["coverage"]
    coverage_rows = [
        (f"{dataset}/{channel.split('_')[0]}", record["exposure_hours"], record["minimum_one_alarm_resolution_hours"])
        for dataset, channels in coverage.items()
        for channel, record in channels.items()
    ]
    x = np.arange(len(coverage_rows))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - 0.18, [row[1] for row in coverage_rows], 0.36, label="available normal hours", color="#E45756")
    axis.bar(x + 0.18, [row[2] for row in coverage_rows], 0.36, label="minimum one-alarm hours", color="#4C78A8")
    axis.set_xticks(x, [row[0] for row in coverage_rows])
    axis.set_ylabel("flight-hours")
    axis.set_title("Calibration STOP: insufficient independent normal exposure")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "07_CALIBRATION_COVERAGE_STOP.png", dpi=160)
    plt.close(fig)

    compact = {
        "development_only": True,
        "holdout_opened": False,
        "k5": {"event_count": len(k5_events), "events": k5_events},
        "s4": {
            "eligible": s4["decision_eligible_channels"],
            "flagged": s4["flagged_channels"],
            "ratios": dict(s4_values),
        },
        "s1": {
            "eligible": s1["decision_eligible_channels"],
            "rho": dict(s1_values),
        },
        "s3_class_status": {
            dataset: {fault_class: report["status"] for fault_class, report in classes.items()}
            for dataset, classes in s3.items()
        },
        "calibration": {
            "status": "STOP_insufficient_development_normal_exposure",
            "thresholds_written": False,
            "coverage": coverage,
        },
        "source_runs": {
            "s4": str(s4_run),
            "scaling": str(scaling_run),
            "s1": str(s1_run),
            "s3": str(s3_run),
            "calibration_failure": str(calibration_run),
        },
    }
    write_json(output / "SUMMARY_FOR_CLAUDE.json", compact, fail_if_exists=True)
    readme = """# RESIDUAL-V1 Phase E handout for Claude

Scope is development-only. Test and sealed holdout were not opened.

Read `SUMMARY_FOR_CLAUDE.json` first, then inspect the seven numbered figures. The decisive
outcome is not a frozen threshold: S-4, scaling, S-1 and S-3 completed, but calibration
stopped because independent development-normal exposure cannot resolve the frozen false-
alarm targets. The earlier exploratory threshold run is explicitly marked DO_NOT_USE.

Important nuance: pooled S-3 KS passes all headline classes, while the ALFA event-level plot
shows heterogeneous individual engine flights. Do not rewrite the pooled class result as
"every event detected".
"""
    (output / "README_FOR_CLAUDE.md").write_text(readme, encoding="utf-8")
    images = sorted(output.glob("*.png"))
    html = ["<html><body><h1>RESIDUAL-V1 Phase E</h1><p>Development-only; holdout sealed.</p>"]
    for image in images:
        html.append(f"<h2>{image.stem}</h2><img src='{image.name}' style='max-width:100%;height:auto'>")
    html.append("</body></html>")
    (output / "GALLERY.html").write_text("\n".join(html), encoding="utf-8")
    manifest_files = []
    for path in sorted(item for item in output.iterdir() if item.is_file()):
        manifest_files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    write_json(
        output / "PACKAGE_MANIFEST.json",
        {
            "development_only": True,
            "holdout_opened": False,
            "file_count_excluding_manifest": len(manifest_files),
            "files": manifest_files,
        },
        fail_if_exists=True,
    )
    print(output)


if __name__ == "__main__":
    main()
