"""Build the detailed Overleaf-ready GNSS pilot report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gecmis_calismalar.uav_gnss.final_report import build_report


if __name__ == "__main__":
    print(build_report("artifacts/uav_gnss_integrity_v1"))

