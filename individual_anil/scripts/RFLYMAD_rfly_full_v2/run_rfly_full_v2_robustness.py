import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gecmis_calismalar.rfly_full.robustness import (
    CANDIDATES,
    CONVERGED_CANDIDATE,
    finalize_experiment,
    extend_convergence_ceiling,
    initialize_experiment,
    prepare_base_models,
    refresh_candidate_reports,
    run_candidate,
    run_converged_real_candidate,
    run_rw1,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name", default="approved_20260722_nested_v1",
        help="Stable artifact directory name used by every resumable stage.",
    )
    parser.add_argument(
        "--stage", required=True,
        choices=(
            "initialize", "prepare", *CANDIDATES, CONVERGED_CANDIDATE,
            "R4_extend", "RW1", "refresh", "finalize",
        ),
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--convergence-ceiling", type=int, default=500)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    root = initialize_experiment(args.run_name)
    result: object = {"experiment": str(root), "stage": args.stage}
    if args.stage == "prepare":
        prepare_base_models(
            root, epochs=args.epochs, torch_threads=args.torch_threads
        )
    elif args.stage in CANDIDATES:
        result = run_candidate(root, args.stage, torch_threads=args.torch_threads)
    elif args.stage == CONVERGED_CANDIDATE:
        result = run_converged_real_candidate(
            root, torch_threads=args.torch_threads
        )
    elif args.stage == "R4_extend":
        result = extend_convergence_ceiling(
            root, max_epochs=args.convergence_ceiling,
            torch_threads=args.torch_threads,
        )
    elif args.stage == "RW1":
        result = run_rw1(root, torch_threads=args.torch_threads)
    elif args.stage == "refresh":
        refresh_candidate_reports(root)
    elif args.stage == "finalize":
        result = finalize_experiment(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(root)


if __name__ == "__main__":
    main()
