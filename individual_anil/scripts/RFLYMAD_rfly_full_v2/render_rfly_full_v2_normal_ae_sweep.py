import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gecmis_calismalar.rfly_full.normal_ae_reporting import render


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", type=Path)
    args = parser.parse_args()
    render(args.sweep)
