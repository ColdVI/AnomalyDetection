"""Entry point for the resumable RflyMAD overnight job."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rfly_full.pipeline import main


if __name__ == "__main__":
    # Keep Windows awake only while this process is alive; no power-plan change.
    execution_state = ctypes.windll.kernel32.SetThreadExecutionState
    execution_state(0x80000000 | 0x00000001)
    try:
        main()
    finally:
        execution_state(0x80000000)
        sys.stdout.flush()
        sys.stderr.flush()
