"""Contract-aware v3.1 runner for ALFA CV smoke and RflyMAD development."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import four_dataset_probabilistic_gpu_v1_runner as core  # noqa: E402


NAMESPACE = "four_dataset_probabilistic_v31"
PREREG = Path("docs/FOUR_DATASET_NORMAL_ONLY_V31_PROTOCOL_DECISIONS_20260728.md")
REGISTRY = Path("artifacts/four_dataset_probabilistic_v3/group_registry_v1.parquet")


class V31RunnerError(core.ProbabilisticV1Error):
    """Raised when v3.1 role or group-balanced sampling contracts fail."""


def allocate_equal(total: int, keys: list[str], purpose_seed: int) -> dict[str, int]:
    """Allocate an integer total equally with deterministic remainder placement."""

    if total < 0 or not keys:
        raise V31RunnerError("Invalid balanced quota request")
    ordered = sorted(
        map(str, keys),
        key=lambda value: core._canonical_sha([purpose_seed, value]),
    )
    base, remainder = divmod(total, len(ordered))
    return {key: base + int(index < remainder) for index, key in enumerate(ordered)}


def _source_groups(root: Path, dataset: str, sources: tuple[str, ...]) -> dict[str, str]:
    import pandas as pd

    registry = pd.read_parquet(root / REGISTRY, columns=["dataset", "source_id", "group_id"])
    part = registry.loc[registry["dataset"].eq(dataset)].copy()
    mapping = dict(zip(part["source_id"].astype(str), part["group_id"].astype(str)))
    missing = set(sources) - set(mapping)
    if missing:
        raise V31RunnerError(f"Group registry misses train sources: {sorted(missing)[:5]}")
    return {source_id: mapping[source_id] for source_id in sources}


def _balanced_train_epoch(
    access: core.DatasetAccess,
    scaler: core.RobustScaler,
    model: core.GaussianForecaster,
    optimizer: Any,
    common: dict[str, Any],
    device: Any,
    epoch_index: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Give groups equal epoch quota and sources equal quota within each group."""

    seed = int(common["seed"])
    rng = np.random.default_rng(seed + epoch_index)
    sources = tuple(map(str, access.roles["train"]))
    source_group = _source_groups(ROOT, access.dataset, sources)
    groups: dict[str, list[str]] = defaultdict(list)
    for source_id in sources:
        groups[source_group[source_id]].append(source_id)
    if len(groups) < 2:
        raise V31RunnerError("Group-balanced training requires at least two train groups")

    candidate_counts: dict[str, int] = {}
    for source_id in sources:
        count = len(core._make_windows(access.load_source(source_id), scaler, common).x)
        if count <= 0:
            raise V31RunnerError(f"Train source has zero causal windows: {source_id}")
        candidate_counts[source_id] = count
    total_candidates = int(sum(candidate_counts.values()))
    group_quota = allocate_equal(total_candidates, list(groups), seed + epoch_index * 1009)
    source_quota: dict[str, int] = {}
    for group_id, group_sources in groups.items():
        source_quota.update(
            allocate_equal(group_quota[group_id], group_sources, seed + epoch_index * 9173)
        )
    if sum(source_quota.values()) != total_candidates:
        raise V31RunnerError("Balanced epoch quota does not preserve total exposure")

    order = list(sources)
    rng.shuffle(order)
    block_limit = int(common["block_windows"])
    x_buffer: list[np.ndarray] = []
    y_buffer: list[np.ndarray] = []
    mask_buffer: list[np.ndarray] = []
    buffered = 0
    loss_sum = 0.0
    observed_cells = 0
    batches = 0
    sampled_windows = 0
    replacement_windows = 0
    started = time.monotonic()

    def flush() -> None:
        nonlocal buffered, loss_sum, observed_cells, batches, sampled_windows
        if not x_buffer:
            return
        x = np.concatenate(x_buffer, axis=0)
        y = np.concatenate(y_buffer, axis=0)
        mask = np.concatenate(mask_buffer, axis=0)
        block_loss, block_cells, block_batches, block_windows = core._optimize_arrays(
            model, optimizer, x, y, mask, common, device, rng
        )
        loss_sum += block_loss
        observed_cells += block_cells
        batches += block_batches
        sampled_windows += block_windows
        x_buffer.clear()
        y_buffer.clear()
        mask_buffer.clear()
        buffered = 0

    for source_index, source_id in enumerate(order, start=1):
        block = core._make_windows(access.load_source(source_id), scaler, common)
        quota = source_quota[source_id]
        replace = quota > len(block.x)
        selected = rng.choice(len(block.x), size=quota, replace=replace)
        if replace:
            replacement_windows += quota - len(np.unique(selected))
        cursor = 0
        while cursor < quota:
            room = block_limit - buffered
            take = min(room, quota - cursor)
            index = selected[cursor : cursor + take]
            x_buffer.append(block.x[index])
            y_buffer.append(block.y[index])
            mask_buffer.append(block.y_mask[index])
            buffered += take
            cursor += take
            if buffered >= block_limit:
                flush()
        if source_index == 1 or source_index % 10 == 0 or source_index == len(order):
            core._write_json_atomic(
                run_dir / "training_progress.json",
                {
                    "schema_version": 1,
                    "dataset": access.dataset,
                    "stage": "training_group_balanced_v31",
                    "epoch": epoch_index + 1,
                    "source_index": source_index,
                    "source_count": len(order),
                    "train_group_count": len(groups),
                    "candidate_windows": total_candidates,
                    "sampled_or_buffered_windows": sampled_windows + buffered,
                    "elapsed_seconds": time.monotonic() - started,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(
                f"epoch={epoch_index + 1} source={source_index}/{len(order)} "
                f"balanced_windows={sampled_windows + buffered}",
                flush=True,
            )
    flush()
    if observed_cells == 0:
        raise V31RunnerError("Balanced epoch produced zero observed target cells")
    quotas = list(group_quota.values())
    return {
        "epoch": epoch_index + 1,
        "mean_train_gaussian_nll": loss_sum / observed_cells,
        "windows": sampled_windows,
        "candidate_windows_before_balancing": total_candidates,
        "observed_target_cells": observed_cells,
        "batches": batches,
        "elapsed_seconds": time.monotonic() - started,
        "sampler": "group_then_source_balanced_epoch_quota",
        "train_group_count": len(groups),
        "group_quota_min": min(quotas),
        "group_quota_max": max(quotas),
        "replacement_window_draws": replacement_windows,
    }


def _configure(dataset: str, fold: int) -> Path:
    if dataset == "alfa":
        if fold not in range(4):
            raise V31RunnerError("ALFA fold must be 0..3")
        path = Path(f"configs/four_dataset_probabilistic_v31_alfa_fold_{fold}.json")
    elif dataset == "rflymad":
        path = Path("configs/four_dataset_probabilistic_v31_rflymad.json")
    else:
        raise V31RunnerError("v3.1 runner authorizes only ALFA and RflyMAD")
    core.CONFIG_PATH = path
    core.PREREG_PATH = PREREG
    core.EXPECTED_NAMESPACE = NAMESPACE
    core._train_epoch = _balanced_train_epoch
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "inventory", "train"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dataset", choices=("alfa", "rflymad"), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    try:
        config_path = _configure(args.dataset, args.fold)
        config, spec = core._load_contract(root, args.dataset)
        access = core._load_access(root, config, spec, args.dataset)
        scaler = core._fit_scaler_access(access, config["common"], str(spec["normal_label"]))
        evidence = core._contract_evidence(root, config, access)
        group_map = _source_groups(root, args.dataset, access.roles["train"])
        if args.command in {"verify", "inventory"}:
            output = core._inventory(access, scaler, evidence)
            output["v31_config_path"] = config_path.as_posix()
            output["train_group_count"] = len(set(group_map.values()))
            output["final_fault_test_accessed"] = False
            output["sampler"] = config["common"]["training_sampler"]
            print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)
            return 0
        if args.run_dir is None:
            raise V31RunnerError("--run-dir is required for train")
        core._run_training(root, args.dataset, args.run_dir.resolve(), args.device)
        return 0
    except core.ProbabilisticV1Error as exc:
        print(f"CONTRACT ERROR: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
