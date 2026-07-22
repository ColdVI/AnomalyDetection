"""Root entry point for the resumable RflyMAD overnight job."""

from __future__ import annotations

import ctypes
import sys

from rfly_full.pipeline import main


if __name__ == "__main__":
    execution_state = ctypes.windll.kernel32.SetThreadExecutionState
    execution_state(0x80000000 | 0x00000001)
    try:
        main()
    finally:
        execution_state(0x80000000)
        sys.stdout.flush()
        sys.stderr.flush()
