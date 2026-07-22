"""Wait for the active v2 parser, then refresh the normal-only AE once."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gecmis_calismalar.rfly_full.normal_ae import OUTPUT_ROOT, run
from gecmis_calismalar.rfly_full.pipeline import _atomic_json
from gecmis_calismalar.rfly_full.v2_parser import PARSE_STATE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-wait-hours", type=float, default=3.0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    status_path = OUTPUT_ROOT / "postparse_training_state.json"
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.max_wait_hours)
    while datetime.now(timezone.utc) < deadline:
        state = json.loads(PARSE_STATE.read_text(encoding="utf-8"))
        _atomic_json(status_path, {
            "status": "waiting_for_parser",
            "parser_stop_reason": state.get("stop_reason"),
            "parsed_flights": len(state.get("completed", [])),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        if state.get("stop_reason") != "running":
            output = run(epochs=args.epochs, torch_threads=4, validation_rotation=0)
            _atomic_json(status_path, {
                "status": "complete",
                "parser_stop_reason": state.get("stop_reason"),
                "parsed_flights": len(state.get("completed", [])),
                "model_output": str(output),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            print(output)
            return
        time.sleep(max(5, args.poll_seconds))
    _atomic_json(status_path, {
        "status": "wait_timeout",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
