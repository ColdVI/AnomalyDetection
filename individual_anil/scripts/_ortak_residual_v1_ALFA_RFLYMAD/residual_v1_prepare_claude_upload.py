from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gecmis_calismalar.residual_v1.run import update_manifest
from gecmis_calismalar.residual_v1.viz.handout import create_claude_upload_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Claude-compatible handout upload files")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run)
    handout = run_dir / "handout"
    destination = run_dir / "claude_upload"
    manifest = create_claude_upload_set(handout, destination)
    update_manifest(
        run_dir,
        claude_upload_root=str(destination),
        claude_upload_file_count=manifest["upload_file_count_including_manifest"],
        claude_pdf_flight_page_count=manifest["pdf_flight_page_count"],
    )
    print(destination)


if __name__ == "__main__":
    main()
