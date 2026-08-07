import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gecmis_calismalar.rfly_full.v2_parser import postprocess_crosscheck_metrics


if __name__ == "__main__":
    print(json.dumps(
        postprocess_crosscheck_metrics(split="development"),
        indent=2,
        ensure_ascii=False,
    ))
