"""Run a development-only TCN smoke while measuring process-tree peak RSS."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rfly_full.contract import V2_ROOT
from rfly_full.pipeline import _atomic_json


def _tree_rss(process: psutil.Process) -> int:
    total = 0
    candidates = [process]
    try:
        candidates.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for candidate in candidates:
        try:
            total += candidate.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _stop_tree(process: psutil.Process) -> None:
    try:
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        process.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--development-smoke-fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-windows", type=int, default=5_000)
    parser.add_argument("--max-val-windows", type=int, default=2_000)
    parser.add_argument("--max-peak-mb", type=float, default=4_096)
    args = parser.parse_args()
    if args.validation_fold == args.development_smoke_fold:
        raise ValueError("validation and development smoke folds must differ")

    output = V2_ROOT / "supervised_tcn" / datetime.now().strftime("memory_smoke_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    state_path = output / "memory_state.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_rfly_full_v2_supervised.py"),
        "--validation-fold", str(args.validation_fold),
        "--development-smoke-fold", str(args.development_smoke_fold),
        "--epochs", str(args.epochs),
        "--max-train-windows", str(args.max_train_windows),
        "--max-val-windows", str(args.max_val_windows),
        "--protocol", "full",
    ]
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    peak = 0
    aborted = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            peak = max(peak, _tree_rss(process))
            elapsed = time.monotonic() - started
            _atomic_json(state_path, {
                "status": "running", "pid": child.pid,
                "started_at": started_at, "elapsed_seconds": elapsed,
                "peak_rss_mb": peak / (1024 ** 2),
                "max_peak_mb": args.max_peak_mb,
                "locked_test_features_read": False,
                "command": command,
            })
            if peak / (1024 ** 2) > args.max_peak_mb:
                aborted = True
                _stop_tree(process)
                break
            time.sleep(0.25)
        return_code = child.wait(timeout=30)
        peak = max(peak, _tree_rss(process))

    elapsed = time.monotonic() - started
    stdout_text = stdout_path.read_text(encoding="utf-8").strip()
    tcn_output = stdout_text.splitlines()[-1] if stdout_text else None
    final = {
        "status": "memory_limit_aborted" if aborted else ("complete" if return_code == 0 else "failed"),
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "peak_rss_mb": peak / (1024 ** 2),
        "max_peak_mb": args.max_peak_mb,
        "memory_gate_passed": bool(not aborted and return_code == 0 and peak / (1024 ** 2) <= args.max_peak_mb),
        "locked_test_features_read": False,
        "validation_fold": args.validation_fold,
        "development_smoke_fold": args.development_smoke_fold,
        "max_train_windows": args.max_train_windows,
        "max_val_windows": args.max_val_windows,
        "epochs": args.epochs,
        "tcn_output": tcn_output,
        "command": command,
    }
    _atomic_json(state_path, final)
    print(json.dumps({"benchmark": str(output), **final}, indent=2))
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
