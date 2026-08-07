"""Durable, read-only monitor for the long contextual_physics_v2 training run.

The monitor deliberately writes outside the training run directory so it cannot
race with the run's final artifact checksum generation.  It never signals or
mutates the training process.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


def _artifact_snapshot(run_dir: Path) -> dict[str, dict[str, Any]]:
    if not run_dir.exists():
        return {}
    return {
        path.name: {
            "bytes": path.stat().st_size,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
        }
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }


def _training_report(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "training_report.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    diagnostic = payload.get("natural_calibration_diagnostic", {})
    return {
        "status": payload.get("status"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "magnitude_domination_flagged_at_0_8": diagnostic.get(
            "magnitude_domination_flagged_at_0_8"
        ),
        "rho_trained_vs_untrained": diagnostic.get("rho_trained_vs_untrained"),
        "rho_trained_vs_target_magnitude": diagnostic.get(
            "rho_trained_vs_target_magnitude"
        ),
    }


def _sample(process: psutil.Process, run_dir: Path, expected_create_time: float) -> dict[str, Any]:
    timestamp = datetime.now().astimezone().isoformat()
    try:
        if abs(process.create_time() - expected_create_time) > 1e-6:
            raise psutil.NoSuchProcess(process.pid)
        memory = process.memory_info()
        cpu = process.cpu_times()
        process_state: dict[str, Any] = {
            "running": process.is_running(),
            "status": process.status(),
            "cpu_seconds": cpu.user + cpu.system,
            "rss_bytes": memory.rss,
            "vms_bytes": memory.vms,
        }
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        process_state = {"running": False, "status": "not_found"}
    return {
        "timestamp": timestamp,
        "pid": process.pid,
        "process": process_state,
        "artifacts": _artifact_snapshot(run_dir),
        "training_report": _training_report(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--deadline", required=True, help="ISO-8601 local deadline")
    parser.add_argument("--interval-s", type=float, default=300.0)
    args = parser.parse_args()
    if args.interval_s <= 0:
        raise ValueError("interval-s must be positive")

    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        raise ValueError("deadline must include a UTC offset")
    process = psutil.Process(args.pid)
    expected_create_time = process.create_time()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    samples = 0
    last: dict[str, Any] | None = None
    stop_reason = "deadline"
    with args.log.open("a", encoding="utf-8") as handle:
        while datetime.now().astimezone() < deadline:
            last = _sample(process, args.run_dir, expected_create_time)
            handle.write(json.dumps(last, ensure_ascii=False) + "\n")
            handle.flush()
            samples += 1
            report = last["training_report"]
            if report is not None:
                stop_reason = "training_report_created"
                break
            if not last["process"]["running"]:
                stop_reason = "training_process_stopped_without_report"
                break
            time.sleep(args.interval_s)

    summary = {
        "schema_version": 1,
        "pid": args.pid,
        "expected_process_create_time": expected_create_time,
        "run_dir": str(args.run_dir),
        "deadline": deadline.isoformat(),
        "interval_s": args.interval_s,
        "samples": samples,
        "stop_reason": stop_reason,
        "last_sample": last,
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
