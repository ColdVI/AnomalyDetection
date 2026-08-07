"""Render development-only RflyMAD v2 EDA and diagnostic visuals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gecmis_calismalar.rfly_full.visualize import DEFAULT_OUTPUT, render


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    print(json.dumps(render(args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
