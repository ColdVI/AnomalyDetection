"""Regenerate RflyMAD DL visual diagnostics from persisted result tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rfly_dl.reporting import refresh_manifest_hashes, render_additional_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "artifacts/rfly_dl/direct_v1_5split_20260720",
    )
    args = parser.parse_args()
    outputs = render_additional_plots(args.artifact_dir)
    refresh_manifest_hashes(args.artifact_dir)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
