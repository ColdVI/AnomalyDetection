"""CLI for the frozen RflyMAD direct deep-learning experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gecmis_calismalar.rfly_dl.config import MAX_EPOCHS, MODEL_NAMES
from gecmis_calismalar.rfly_dl.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--splits", nargs="+",
        default=["split_00", "split_01", "split_02", "split_03", "split_04"],
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    output = run_experiment(
        args.run_name,
        selected_splits=tuple(args.splits),
        selected_models=tuple(args.models),
        max_epochs=args.max_epochs,
        device=args.device,
        torch_threads=args.torch_threads,
    )
    print(output)


if __name__ == "__main__":
    main()
