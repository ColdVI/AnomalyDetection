"""Evaluate the frozen Step-5 h=1 vector CUSUM on truth-v2 synthetic data.

No detector fitting, calibration, threshold sweep, or score fusion is exposed
by this command. It refuses an existing run directory and validates the full
Step-5 completion/hash/config chain before creating an output manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsb.cusum_truth_v2_eval import (  # noqa: E402
    DEFAULT_SAMPLE_CAPACITY_PER_CLASS,
    DEFAULT_SAMPLE_SEED,
    run_evaluation,
)


DEFAULT_STEP5_RUN = Path("artifacts/adsb/runs/20260713_step5_full_streaming_v1")
DEFAULT_CORPUS = Path("data/objectstore/synthetic/adsb_v2_20260713_01")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parent.parent)
    parser.add_argument("--step5-run", type=Path, default=DEFAULT_STEP5_RUN)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--diagnostic-sample-capacity-per-class",
        type=int,
        default=DEFAULT_SAMPLE_CAPACITY_PER_CLASS,
    )
    parser.add_argument(
        "--diagnostic-sample-seed", type=int, default=DEFAULT_SAMPLE_SEED
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_evaluation(
        repo_root=args.repo_root,
        step5_dir=args.step5_run,
        corpus_dir=args.corpus_dir,
        run_dir=args.run_dir,
        sample_capacity_per_class=args.diagnostic_sample_capacity_per_class,
        sample_seed=args.diagnostic_sample_seed,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
