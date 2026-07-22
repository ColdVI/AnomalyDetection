from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gecmis_calismalar.residual_v1.decision.calibrate import (
    CalibrationConfig,
    calibrate_channel_threshold,
    normal_block_exposure,
)
from gecmis_calismalar.residual_v1.eval.sanity_gates import GateError, require_s3_pass
from gecmis_calismalar.residual_v1.ingest.common import write_json
from gecmis_calismalar.residual_v1.run import create_run_dir, sha256_file, update_manifest
from gecmis_calismalar.residual_v1.tracking import log_run


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normal_flight_ids(split_path: Path, silver_root: Path) -> set[str]:
    split = _read_json(split_path)
    normal = set()
    for flight_id in split["partitions"]["development"]["flight_ids"]:
        events = _read_json(silver_root / Path(flight_id) / "events.json")
        if not events:
            normal.add(flight_id)
    if not normal:
        raise ValueError(f"no development-normal flights under {silver_root}")
    return normal


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate frozen RESIDUAL-V1 CUSUM thresholds")
    parser.add_argument("--scaling-run", required=True)
    parser.add_argument("--s1-run", required=True)
    parser.add_argument("--s3-run", required=True)
    parser.add_argument("--alfa-silver-root", required=True)
    parser.add_argument("--rfly-silver-root", required=True)
    parser.add_argument("--alfa-split", default="artifacts/residual_v1/splits/alfa_seed11.json")
    parser.add_argument("--rfly-split", default="artifacts/residual_v1/splits/rfly_seed11.json")
    parser.add_argument("--config", default="configs/residual_v1_cusum.json")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    scaling_run = Path(args.scaling_run)
    s1_run = Path(args.s1_run)
    s3_run = Path(args.s3_run)
    config_path = Path(args.config)
    config_payload = _read_json(config_path)
    scaling_summary = _read_json(scaling_run / "summary.json")
    s1_summary = _read_json(s1_run / "summary.json")
    s3_summary = _read_json(s3_run / "summary.json")
    if not all(
        bool(payload.get("development_only"))
        for payload in (scaling_summary, s1_summary, s3_summary)
    ):
        raise ValueError("calibration requires development-only scaling/S-1/S-3 runs")
    if Path(s1_summary["scaling_run"]) != scaling_run or Path(s3_summary["scaling_run"]) != scaling_run:
        raise ValueError("gate runs do not belong to the supplied scaling run")
    require_s3_pass(s3_summary["class_status"]["alfa"], ["engine"])
    require_s3_pass(s3_summary["class_status"]["rfly"], ["motor", "sensor"])

    normal_ids = {
        "alfa": _normal_flight_ids(Path(args.alfa_split), Path(args.alfa_silver_root)),
        "rfly": _normal_flight_ids(Path(args.rfly_split), Path(args.rfly_silver_root)),
    }
    calibration_config = CalibrationConfig(
        k=float(config_payload["k"]),
        z_clip=float(config_payload["z_clip"]),
        max_gap_s=float(config_payload["max_gap_s"]),
        refractory_s=float(config_payload["refractory_s"]),
        block_s=float(config_payload["bootstrap_block_s"]),
        repetitions=int(config_payload["bootstrap_repetitions"]),
        seed=int(config_payload["seed"]),
    )
    run_dir, _ = create_run_dir(
        f"phaseE_cusum_calibration_seed{args.seed}",
        seed=args.seed,
        config_paths=[config_path],
        input_paths=[
            scaling_run / "summary.json",
            s1_run / "summary.json",
            s3_run / "summary.json",
            Path(args.alfa_split),
            Path(args.rfly_split),
        ],
    )
    (run_dir / "channel_reports").mkdir()
    (run_dir / "plots").mkdir()
    reports = {"alfa": {}, "rfly": {}}
    total_target = float(config_payload["total_false_alarms_per_flight_hour"])
    normal_frames = {}
    coverage = {"alfa": {}, "rfly": {}}
    coverage_failures = []
    for dataset, channels in s1_summary["decision_eligible_channels"].items():
        if not channels:
            raise ValueError(f"{dataset}: no S-1-eligible channels for calibration")
        target = total_target / len(channels)
        for channel in channels:
            frame = pd.read_parquet(scaling_run / "scaled" / dataset / f"{channel}.parquet")
            normal = frame.loc[frame["flight_id"].astype(str).isin(normal_ids[dataset])].copy()
            if normal.empty:
                raise ValueError(f"{dataset}/{channel}: no development-normal score rows")
            normal_frames[(dataset, channel)] = normal
            exposure = normal_block_exposure(normal, block_s=calibration_config.block_s)
            exposure["target_alarms_per_flight_hour"] = target
            exposure["minimum_one_alarm_resolution_hours"] = 1.0 / target
            exposure["sufficient"] = exposure["exposure_hours"] >= exposure["minimum_one_alarm_resolution_hours"]
            coverage[dataset][channel] = exposure
            if not exposure["sufficient"]:
                coverage_failures.append((dataset, channel, exposure))

    if coverage_failures:
        failure = {
            "status": "stopped_insufficient_calibration_exposure",
            "development_only": True,
            "thresholds_written": False,
            "coverage": coverage,
            "s3_gate_run": str(s3_run),
        }
        write_json(run_dir / "summary.json", failure, fail_if_exists=True)
        lines = [
            "# RESIDUAL-V1 Calibration Coverage Failure",
            "",
            "`thresholds_frozen.json` yazılmadı. Bootstrap, gözlenmemiş normal maruziyeti yaratamaz.",
            "",
            "| Veri | Kanal | Normal saat | Hedef alarm/saat | Tek-alarm için minimum saat |",
            "|---|---|---:|---:|---:|",
        ]
        for dataset, channel, exposure in coverage_failures:
            lines.append(
                f"| {dataset} | {channel} | {exposure['exposure_hours']:.6f} | "
                f"{exposure['target_alarms_per_flight_hour']:.6f} | "
                f"{exposure['minimum_one_alarm_resolution_hours']:.6f} |"
            )
        (run_dir / "CALIBRATION_COVERAGE_FAILURE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        update_manifest(
            run_dir,
            development_only=True,
            calibration_locked=True,
            thresholds_written=False,
            coverage=coverage,
        )
        print(run_dir)
        raise GateError("insufficient development-normal exposure; thresholds not written")

    for dataset, channels in s1_summary["decision_eligible_channels"].items():
        target = total_target / len(channels)
        for channel in channels:
            normal = normal_frames[(dataset, channel)]
            report = calibrate_channel_threshold(
                normal,
                channel=channel,
                target_alarms_per_hour=target,
                config=calibration_config,
            )
            report["dataset"] = dataset
            report["development_normal_flights"] = int(normal["flight_id"].nunique())
            report["non_normal_rows_read_for_threshold"] = 0
            reports[dataset][channel] = report
            write_json(
                run_dir / "channel_reports" / f"{dataset}_{channel}.json",
                report,
                fail_if_exists=True,
            )
            fig, axis = plt.subplots(figsize=(8, 4.5))
            axis.hist(report["bootstrap_rates"], bins=30, color="#4C78A8", alpha=0.85)
            axis.axvline(target, color="#E45756", linestyle="--", label=f"target={target:.3f}")
            axis.set_xlabel("false alarms / flight-hour")
            axis.set_ylabel("bootstrap repetitions")
            axis.set_title(f"{dataset}/{channel} — h={report['threshold_h']:.4f}")
            axis.legend()
            fig.tight_layout()
            fig.savefig(run_dir / "plots" / f"{dataset}_{channel}_bootstrap_fa.png", dpi=150)
            plt.close(fig)

    frozen = {
        "schema_version": 1,
        "development_only": True,
        "immutable_fail_if_exists": True,
        "s3_gate_run": str(s3_run),
        "s3_class_status": s3_summary["class_status"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "total_false_alarms_per_flight_hour": total_target,
        "datasets": {
            dataset: {
                "active_channels": sorted(channels),
                "target_per_channel_alarms_per_flight_hour": total_target / len(channels),
                "thresholds_h": {
                    channel: reports[dataset][channel]["threshold_h"] for channel in channels
                },
            }
            for dataset, channels in s1_summary["decision_eligible_channels"].items()
        },
    }
    write_json(run_dir / "thresholds_frozen.json", frozen, fail_if_exists=True)
    summary = {
        "seed": args.seed,
        "development_only": True,
        "thresholds_file": str(run_dir / "thresholds_frozen.json"),
        "datasets": frozen["datasets"],
        "s3_gate_run": str(s3_run),
    }
    write_json(run_dir / "summary.json", summary, fail_if_exists=True)
    tracking = log_run(
        run_dir,
        run_name="phaseE_cusum_calibration",
        metrics={
            "calibrated_channels": sum(len(channels) for channels in reports.values()),
            "target_total_fa_per_hour": total_target,
        },
        params={"seed": args.seed, "bootstrap_repetitions": calibration_config.repetitions},
    )
    update_manifest(
        run_dir,
        development_only=True,
        thresholds_file=str(run_dir / "thresholds_frozen.json"),
        s3_gate_run=str(s3_run),
        mlflow_status=tracking["status"],
    )
    print(run_dir)


if __name__ == "__main__":
    main()
