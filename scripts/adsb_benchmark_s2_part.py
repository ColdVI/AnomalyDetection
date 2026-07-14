"""Compare direct and spawned-worker S2 timing on one already-open Silver part."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.adsb_report_s2_natural_burden import (
    S2_COLUMNS,
    _load_base_manifest,
    _process_silver_part,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=Path(
            "artifacts/adsb/runs/20260713_step5_full_streaming_v1/run_manifest.json"
        ),
    )
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--mode", choices=("direct", "spawn"), required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    contract = _load_base_manifest(args.baseline_manifest, root)
    if args.count <= 0 or args.workers <= 0:
        raise ValueError("count and workers must be positive")
    if not 0 <= args.part_index < len(contract["silver_inputs"]):
        raise ValueError("part-index is out of range")
    stop = args.part_index + args.count
    if stop > len(contract["silver_inputs"]):
        raise ValueError("requested part range is out of bounds")
    tasks = [
        (index, contract["silver_inputs"][index], tuple(S2_COLUMNS))
        for index in range(args.part_index, stop)
    ]
    started = time.perf_counter()
    if args.mode == "direct":
        results = [_process_silver_part(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_process_silver_part, tasks, chunksize=1))
    elapsed = time.perf_counter() - started
    print(
        f"mode={args.mode} first_part={args.part_index} count={args.count} "
        f"workers={args.workers} "
        f"rows={sum(result['summary']['n_rows'] for result in results)} "
        f"elapsed_seconds={elapsed:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
