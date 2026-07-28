"""Balanced-split v2 entry point for the four-dataset probabilistic baseline."""

from pathlib import Path

import four_dataset_probabilistic_gpu_v1_runner as core


core.CONFIG_PATH = Path("configs/four_dataset_probabilistic_gpu_v2.json")
core.PREREG_PATH = Path("docs/FOUR_DATASET_PROBABILISTIC_GPU_V2_PREREG_20260727.md")
core.EXPECTED_NAMESPACE = "four_dataset_probabilistic_gpu_v2"


if __name__ == "__main__":
    raise SystemExit(core.main())
