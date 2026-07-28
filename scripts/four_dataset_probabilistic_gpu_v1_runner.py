"""GPU-resumable probabilistic temporal baseline for four archived UAV datasets.

This is a new, explicitly exploratory namespace.  It reads the frozen Gold
feature parquet (ALFA/UAV-Attack/UAV-SEAD) or frozen Full-v2 parsed flights
(RflyMAD), trains only on normal/benign development flights, and predicts the
next standardized feature vector as independent Gaussian channels.  Test
labels never affect scaling, optimization, or the validation-only alarm
threshold.  The RflyMAD locked-test partition is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


CONFIG_PATH = Path("configs/four_dataset_probabilistic_gpu_v1.json")
PREREG_PATH = Path("docs/FOUR_DATASET_PROBABILISTIC_GPU_V1_PREREG_20260727.md")
EXPECTED_NAMESPACE = "four_dataset_probabilistic_gpu_v1"
REQUIRED_META_COLUMNS = ("t_rel_s", "source_id", "label")


class ProbabilisticV1Error(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Tolerate short Windows scanner/sync locks while keeping atomic replace."""

    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary, path)


def _torch_save_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    _replace_with_retry(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_contract(root: Path, dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = root / CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("candidate_namespace") != EXPECTED_NAMESPACE:
        raise ProbabilisticV1Error("Unexpected candidate namespace")
    if dataset not in config["datasets"]:
        raise ProbabilisticV1Error(f"Unknown dataset {dataset!r}")
    if not all(config["prohibitions"].values()):
        raise ProbabilisticV1Error("Every prohibition must remain true")
    return config, config["datasets"][dataset]


def _load_split(root: Path, config: dict[str, Any], dataset: str) -> dict[str, Any]:
    path = root / config["split_manifest"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    try:
        split = manifest["sources"][dataset]["splits"][config["split_name"]]
    except KeyError as exc:
        raise ProbabilisticV1Error("Frozen split is missing") from exc
    roles = {role: tuple(map(str, split[role])) for role in ("train", "val", "test")}
    if set(roles["train"]) & set(roles["val"]):
        raise ProbabilisticV1Error("Train/validation source overlap")
    if (set(roles["train"]) | set(roles["val"])) & set(roles["test"]):
        raise ProbabilisticV1Error("Development/test source overlap")
    return {"manifest": manifest, "split": split, "roles": roles, "path": path}


def _load_frame(root: Path, spec: dict[str, Any]) -> tuple[pd.DataFrame, Path]:
    path = root / spec["data_path"]
    requested = list(REQUIRED_META_COLUMNS) + list(spec["features"])
    frame = pd.read_parquet(path, columns=requested)
    missing = set(requested) - set(frame.columns)
    if missing:
        raise ProbabilisticV1Error(f"Missing parquet columns: {sorted(missing)}")
    frame["source_id"] = frame["source_id"].astype("string")
    frame["label"] = frame["label"].astype("string")
    frame["t_rel_s"] = pd.to_numeric(frame["t_rel_s"], errors="coerce")
    frame = frame.sort_values(["source_id", "t_rel_s"], kind="mergesort").reset_index(drop=True)
    return frame, path


@dataclass
class DatasetAccess:
    dataset: str
    features: tuple[str, ...]
    roles: dict[str, tuple[str, ...]]
    labels: dict[str, str]
    data_fingerprint: dict[str, Any]
    contract_paths: tuple[Path, ...]
    frame: pd.DataFrame | None = None
    row_indices: dict[str, np.ndarray] | None = None
    source_paths: dict[str, Path] | None = None

    def load_source(self, source_id: str) -> pd.DataFrame:
        if source_id not in self.labels:
            raise ProbabilisticV1Error(f"Unknown source {source_id!r}")
        if self.frame is not None and self.row_indices is not None:
            result = self.frame.iloc[self.row_indices[source_id]].copy()
        elif self.source_paths is not None:
            path = self.source_paths[source_id]
            requested = ["t_rel_s", "canonical_case_id", "fault_family", *self.features]
            result = pd.read_parquet(path, columns=requested)
            ids = set(result["canonical_case_id"].dropna().astype(str).unique())
            if ids != {source_id}:
                raise ProbabilisticV1Error(
                    f"RflyMAD source identity mismatch in {path}: {sorted(ids)}"
                )
            result = result.rename(
                columns={"canonical_case_id": "source_id", "fault_family": "label"}
            )
        else:
            raise ProbabilisticV1Error("Dataset access has no backing data")
        result["source_id"] = result["source_id"].astype("string")
        result["label"] = result["label"].astype("string")
        result["t_rel_s"] = pd.to_numeric(result["t_rel_s"], errors="coerce")
        return result.sort_values("t_rel_s", kind="mergesort").reset_index(drop=True)


def _load_access(
    root: Path,
    config: dict[str, Any],
    spec: dict[str, Any],
    dataset: str,
) -> DatasetAccess:
    features = tuple(map(str, spec["features"]))
    if spec.get("data_mode") != "rfly_full_v2_parsed":
        split_info = _load_split(root, config, dataset)
        frame, data_path = _load_frame(root, spec)
        roles = split_info["roles"]
        required_ids = set().union(*map(set, roles.values()))
        available_ids = set(frame["source_id"].dropna().astype(str).unique())
        missing_ids = required_ids - available_ids
        if missing_ids:
            raise ProbabilisticV1Error(
                f"Frozen split references missing sources: {sorted(missing_ids)[:5]}"
            )
        frozen_labels = split_info["manifest"]["sources"][dataset].get("flight_labels", {})
        missing_labels = required_ids - set(map(str, frozen_labels))
        if missing_labels:
            raise ProbabilisticV1Error(
                f"Frozen flight labels missing sources: {sorted(missing_labels)[:5]}"
            )
        normal_label = str(spec["normal_label"])
        labels = {
            str(source_id): (
                normal_label if str(label) == "normal" else str(label)
            )
            for source_id, label in frozen_labels.items()
        }
        row_indices = {
            str(source_id): np.asarray(indices, dtype=np.int64)
            for source_id, indices in frame.groupby("source_id", sort=False).indices.items()
        }
        data_fingerprint = {
            "mode": "gold_parquet",
            "data_path": data_path.relative_to(root).as_posix(),
            "data_bytes": data_path.stat().st_size,
            "data_sha256": _sha256(data_path),
            "split_path": split_info["path"].relative_to(root).as_posix(),
            "split_sha256": _sha256(split_info["path"]),
        }
        return DatasetAccess(
            dataset=dataset,
            features=features,
            roles=roles,
            labels={source_id: labels[source_id] for source_id in required_ids},
            data_fingerprint=data_fingerprint,
            contract_paths=(data_path, split_info["path"]),
            frame=frame,
            row_indices=row_indices,
        )

    manifest_path = root / spec["manifest_path"]
    split_registry_path = root / spec["split_registry_path"]
    base_path = root / spec["data_path"]
    manifest = pd.read_parquet(manifest_path)
    required_manifest = {
        "canonical_case_id",
        "domain",
        "fault_family",
        "split",
        "cv_fold",
    }
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ProbabilisticV1Error(
            f"RflyMAD manifest columns missing: {sorted(missing_manifest)}"
        )
    if manifest["canonical_case_id"].astype(str).duplicated().any():
        raise ProbabilisticV1Error("RflyMAD manifest source ids are not unique")
    development = manifest.loc[manifest["split"].astype(str).eq("development")].copy()
    development["canonical_case_id"] = development["canonical_case_id"].astype(str)
    development["fault_family"] = development["fault_family"].astype(str)
    development["cv_fold"] = pd.to_numeric(development["cv_fold"], errors="raise").astype(int)
    normal = str(spec["normal_label"])
    explicit_split_path: Path | None = None
    if spec.get("explicit_split_manifest_path"):
        explicit_split_path = root / str(spec["explicit_split_manifest_path"])
        explicit = json.loads(explicit_split_path.read_text(encoding="utf-8"))
        try:
            declared = explicit["sources"][dataset]["roles"]
            roles = {
                role: tuple(map(str, declared[role]))
                for role in ("train", "val", "test")
            }
        except (KeyError, TypeError) as exc:
            raise ProbabilisticV1Error("Explicit RflyMAD split is malformed") from exc
        available_ids = set(development["canonical_case_id"])
        missing_ids = set().union(*map(set, roles.values())) - available_ids
        if missing_ids:
            raise ProbabilisticV1Error(
                f"Explicit RflyMAD split references missing sources: {sorted(missing_ids)[:5]}"
            )
        declared_normal = set(roles["train"]) | set(roles["val"])
        actual_normal = set(
            development.loc[
                development["fault_family"].eq(normal), "canonical_case_id"
            ]
        )
        if not declared_normal <= actual_normal:
            raise ProbabilisticV1Error(
                "Explicit RflyMAD train/validation roles must be normal-only"
            )
    else:
        train_folds = set(map(int, spec["train_folds"]))
        validation_fold = int(spec["validation_fold"])
        test_fold = int(spec["test_fold"])
        roles = {
            "train": tuple(
                development.loc[
                    development["fault_family"].eq(normal)
                    & development["cv_fold"].isin(train_folds),
                    "canonical_case_id",
                ].tolist()
            ),
            "val": tuple(
                development.loc[
                    development["fault_family"].eq(normal)
                    & development["cv_fold"].eq(validation_fold),
                    "canonical_case_id",
                ].tolist()
            ),
            "test": tuple(
                development.loc[
                    development["cv_fold"].eq(test_fold), "canonical_case_id"
                ].tolist()
            ),
        }
    if not all(roles.values()):
        raise ProbabilisticV1Error("One or more RflyMAD roles are empty")
    if set(roles["train"]) & set(roles["val"]):
        raise ProbabilisticV1Error("RflyMAD train/validation overlap")
    if (set(roles["train"]) | set(roles["val"])) & set(roles["test"]):
        raise ProbabilisticV1Error("RflyMAD development/test overlap")
    selected_ids = set().union(*map(set, roles.values()))
    selected = development.loc[development["canonical_case_id"].isin(selected_ids)]
    labels = selected.set_index("canonical_case_id")["fault_family"].to_dict()
    domains = selected.set_index("canonical_case_id")["domain"].astype(str).to_dict()
    source_paths = {
        source_id: base_path / domains[source_id] / f"{source_id}.parquet"
        for source_id in sorted(selected_ids)
    }
    missing_files = [path for path in source_paths.values() if not path.is_file()]
    if missing_files:
        raise ProbabilisticV1Error(
            f"Missing RflyMAD parsed files: {[str(p) for p in missing_files[:5]]}"
        )
    expected_columns = {"t_rel_s", "canonical_case_id", "fault_family", *features}
    file_evidence: list[dict[str, Any]] = []
    for source_id in sorted(source_paths):
        path = source_paths[source_id]
        columns = set(pq.read_schema(path).names)
        missing = expected_columns - columns
        if missing:
            raise ProbabilisticV1Error(
                f"RflyMAD source {source_id} schema missing columns: {sorted(missing)}"
            )
        file_evidence.append(
            {
                "source_id": source_id,
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    data_fingerprint = {
        "mode": "rfly_full_v2_parsed",
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "split_registry_path": split_registry_path.relative_to(root).as_posix(),
        "split_registry_sha256": _sha256(split_registry_path),
        "selected_file_count": len(file_evidence),
        "selected_files_sha256": _canonical_sha(file_evidence),
    }
    if explicit_split_path is not None:
        data_fingerprint["explicit_split_manifest_path"] = (
            explicit_split_path.relative_to(root).as_posix()
        )
        data_fingerprint["explicit_split_manifest_sha256"] = _sha256(
            explicit_split_path
        )
    contract_paths = [manifest_path, split_registry_path, base_path]
    if explicit_split_path is not None:
        contract_paths.append(explicit_split_path)
    return DatasetAccess(
        dataset=dataset,
        features=features,
        roles=roles,
        labels={str(key): str(value) for key, value in labels.items()},
        data_fingerprint=data_fingerprint,
        contract_paths=tuple(contract_paths),
        source_paths=source_paths,
    )


def _fit_scaler_access(
    access: DatasetAccess,
    common: dict[str, Any],
    normal_label: str,
) -> RobustScaler:
    raw_parts: list[np.ndarray] = []
    labels: set[str] = set()
    for source_id in access.roles["train"]:
        frame = access.load_source(source_id)
        labels.update(frame["label"].dropna().astype(str).unique())
        raw_parts.append(
            frame.loc[:, access.features]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )
    if labels != {normal_label}:
        raise ProbabilisticV1Error(f"Optimizer role is not normal-only: {sorted(labels)}")
    if not raw_parts:
        raise ProbabilisticV1Error("No train rows")
    raw = np.concatenate(raw_parts, axis=0)
    keep: list[str] = []
    excluded: list[str] = []
    medians: list[float] = []
    scales: list[float] = []
    minimum_mad = float(common["minimum_mad"])
    factor = float(common["robust_scale_factor"])
    for column_index, feature in enumerate(access.features):
        values = raw[:, column_index]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            excluded.append(feature)
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        if not np.isfinite(mad) or mad <= minimum_mad:
            excluded.append(feature)
            continue
        keep.append(feature)
        medians.append(median)
        scales.append(factor * mad)
    if len(keep) < 2:
        raise ProbabilisticV1Error("Fewer than two non-degenerate train channels")
    return RobustScaler(
        features=tuple(keep),
        median=np.asarray(medians, dtype=np.float64),
        scale=np.asarray(scales, dtype=np.float64),
        excluded=tuple(excluded),
        clip=float(common["robust_clip"]),
    )


@dataclass(frozen=True)
class RobustScaler:
    features: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray
    excluded: tuple[str, ...]
    clip: float

    def transform(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(raw, dtype=np.float64)
        mask = np.isfinite(values)
        scaled = (values - self.median) / self.scale
        scaled = np.clip(scaled, -self.clip, self.clip)
        scaled[~mask] = 0.0
        return scaled.astype(np.float32), mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "median": {name: float(v) for name, v in zip(self.features, self.median)},
            "scale": {name: float(v) for name, v in zip(self.features, self.scale)},
            "excluded": list(self.excluded),
            "clip": self.clip,
        }


def _fit_scaler(
    frame: pd.DataFrame,
    train_sources: Iterable[str],
    requested_features: Iterable[str],
    common: dict[str, Any],
    normal_label: str,
) -> RobustScaler:
    train_set = set(train_sources)
    selected = frame.loc[frame["source_id"].isin(train_set)]
    if selected.empty:
        raise ProbabilisticV1Error("No train rows")
    labels = set(selected["label"].dropna().astype(str).unique())
    if labels != {normal_label}:
        raise ProbabilisticV1Error(f"Optimizer role is not normal-only: {sorted(labels)}")
    keep: list[str] = []
    excluded: list[str] = []
    medians: list[float] = []
    scales: list[float] = []
    minimum_mad = float(common["minimum_mad"])
    factor = float(common["robust_scale_factor"])
    for feature in requested_features:
        values = pd.to_numeric(selected[feature], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            excluded.append(feature)
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        if not np.isfinite(mad) or mad <= minimum_mad:
            excluded.append(feature)
            continue
        keep.append(feature)
        medians.append(median)
        scales.append(factor * mad)
    if len(keep) < 2:
        raise ProbabilisticV1Error("Fewer than two non-degenerate train channels")
    return RobustScaler(
        features=tuple(keep),
        median=np.asarray(medians, dtype=np.float64),
        scale=np.asarray(scales, dtype=np.float64),
        excluded=tuple(excluded),
        clip=float(common["robust_clip"]),
    )


@dataclass(frozen=True)
class WindowBlock:
    x: np.ndarray
    y: np.ndarray
    y_mask: np.ndarray
    target_time: np.ndarray


def _continuous_slices(time_values: np.ndarray, max_gap_s: float) -> Iterator[slice]:
    finite = np.isfinite(time_values)
    breaks = np.ones(len(time_values), dtype=bool)
    if len(time_values) > 1:
        delta = np.diff(time_values)
        breaks[1:] = (~finite[1:]) | (~finite[:-1]) | (delta <= 0.0) | (delta > max_gap_s)
    starts = np.flatnonzero(breaks)
    ends = np.r_[starts[1:], len(time_values)]
    for start, end in zip(starts, ends):
        yield slice(int(start), int(end))


def _make_windows(
    source_frame: pd.DataFrame,
    scaler: RobustScaler,
    common: dict[str, Any],
) -> WindowBlock:
    history = int(common["history_rows"])
    max_gap = float(common["max_gap_s"])
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    times_all = source_frame["t_rel_s"].to_numpy(float)
    raw_all = source_frame.loc[:, scaler.features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    for segment in _continuous_slices(times_all, max_gap):
        times = times_all[segment]
        raw = raw_all[segment]
        if len(times) <= history:
            continue
        values, observed = scaler.transform(raw)
        delta = np.r_[0.0, np.diff(times)]
        delta = np.clip(np.nan_to_num(delta, nan=0.0, posinf=max_gap, neginf=0.0), 0.0, max_gap)
        cadence = (np.log1p(delta) / math.log1p(max_gap)).astype(np.float32)[:, None]
        inputs = np.concatenate([values, observed.astype(np.float32), cadence], axis=1)
        windows = sliding_window_view(inputs[:-1], history, axis=0).transpose(0, 2, 1).copy()
        target = values[history:].copy()
        target_mask = observed[history:].copy()
        valid = target_mask.any(axis=1)
        if valid.any():
            x_parts.append(windows[valid])
            y_parts.append(target[valid])
            mask_parts.append(target_mask[valid])
            time_parts.append(times[history:][valid])
    feature_count = len(scaler.features)
    input_count = feature_count * 2 + 1
    if not x_parts:
        return WindowBlock(
            np.empty((0, history, input_count), np.float32),
            np.empty((0, feature_count), np.float32),
            np.empty((0, feature_count), bool),
            np.empty(0, np.float64),
        )
    return WindowBlock(
        np.concatenate(x_parts),
        np.concatenate(y_parts),
        np.concatenate(mask_parts),
        np.concatenate(time_parts),
    )


class GaussianForecaster(nn.Module):
    def __init__(self, input_size: int, channels: int, common: dict[str, Any]) -> None:
        super().__init__()
        self.channels = channels
        self.min_scale = float(common["min_scale"])
        self.max_scale = float(common["max_scale"])
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=int(common["hidden_size"]),
            num_layers=int(common["num_layers"]),
            batch_first=True,
        )
        self.head = nn.Linear(int(common["hidden_size"]), channels * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.lstm(x)
        raw = self.head(encoded[:, -1])
        mean, raw_scale = raw.split(self.channels, dim=1)
        scale = self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(raw_scale)
        return mean, scale


def _masked_gaussian_nll(
    mean: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cells = 0.5 * ((target - mean) / scale).square() + torch.log(scale)
    weights = mask.to(cells.dtype)
    valid = weights.sum().clamp_min(1.0)
    return (cells * weights).sum() / valid, cells


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _optimize_arrays(
    model: GaussianForecaster,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    common: dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[float, int, int, int]:
    if len(x) == 0:
        return 0.0, 0, 0, 0
    order = rng.permutation(len(x))
    batch_size = int(common["batch_size"])
    gradient_clip = float(common["gradient_clip_norm"])
    loss_sum = 0.0
    observed_cells = 0
    batches = 0
    model.train()
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        xb = torch.from_numpy(x[index]).to(device, non_blocking=True)
        yb = torch.from_numpy(y[index]).to(device, non_blocking=True)
        mb = torch.from_numpy(mask[index]).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        mean, scale = model(xb)
        loss, _ = _masked_gaussian_nll(mean, scale, yb, mb)
        if not torch.isfinite(loss):
            raise ProbabilisticV1Error("Non-finite Gaussian NLL during optimization")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        cells = int(mb.sum().item())
        loss_sum += float(loss.detach().item()) * cells
        observed_cells += cells
        batches += 1
    return loss_sum, observed_cells, batches, len(x)


def _train_epoch(
    access: DatasetAccess,
    scaler: RobustScaler,
    model: GaussianForecaster,
    optimizer: torch.optim.Optimizer,
    common: dict[str, Any],
    device: torch.device,
    epoch_index: int,
    run_dir: Path,
) -> dict[str, Any]:
    seed = int(common["seed"])
    rng = np.random.default_rng(seed + epoch_index)
    sources = list(access.roles["train"])
    rng.shuffle(sources)
    block_limit = int(common["block_windows"])
    x_buffer: list[np.ndarray] = []
    y_buffer: list[np.ndarray] = []
    mask_buffer: list[np.ndarray] = []
    buffered = 0
    loss_sum = 0.0
    observed_cells = 0
    batches = 0
    windows = 0
    started = time.monotonic()

    def flush() -> None:
        nonlocal buffered, loss_sum, observed_cells, batches, windows
        if not x_buffer:
            return
        x = np.concatenate(x_buffer, axis=0)
        y = np.concatenate(y_buffer, axis=0)
        mask = np.concatenate(mask_buffer, axis=0)
        block_loss, block_cells, block_batches, block_windows = _optimize_arrays(
            model, optimizer, x, y, mask, common, device, rng
        )
        loss_sum += block_loss
        observed_cells += block_cells
        batches += block_batches
        windows += block_windows
        x_buffer.clear()
        y_buffer.clear()
        mask_buffer.clear()
        buffered = 0

    for source_index, source_id in enumerate(sources, start=1):
        block = _make_windows(access.load_source(source_id), scaler, common)
        cursor = 0
        while cursor < len(block.x):
            room = block_limit - buffered
            take = min(room, len(block.x) - cursor)
            x_buffer.append(block.x[cursor : cursor + take])
            y_buffer.append(block.y[cursor : cursor + take])
            mask_buffer.append(block.y_mask[cursor : cursor + take])
            buffered += take
            cursor += take
            if buffered >= block_limit:
                flush()
        if source_index == 1 or source_index % 10 == 0 or source_index == len(sources):
            progress = {
                "schema_version": 1,
                "dataset": access.dataset,
                "stage": "training",
                "epoch": epoch_index + 1,
                "source_index": source_index,
                "source_count": len(sources),
                "committed_windows": windows,
                "buffered_windows": buffered,
                "committed_batches": batches,
                "elapsed_seconds": time.monotonic() - started,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "note": "Only completed epochs are resumable; this file is live progress.",
            }
            _write_json_atomic(run_dir / "training_progress.json", progress)
            print(
                f"epoch={epoch_index + 1} source={source_index}/{len(sources)} "
                f"windows={windows + buffered} batches={batches}",
                flush=True,
            )
    flush()
    if observed_cells == 0:
        raise ProbabilisticV1Error("Epoch produced zero observed target cells")
    return {
        "epoch": epoch_index + 1,
        "mean_train_gaussian_nll": loss_sum / observed_cells,
        "windows": windows,
        "observed_target_cells": observed_cells,
        "batches": batches,
        "elapsed_seconds": time.monotonic() - started,
    }


@dataclass(frozen=True)
class ScoreSet:
    scores: np.ndarray
    magnitudes: np.ndarray
    source_codes: np.ndarray
    target_times: np.ndarray
    sources: tuple[str, ...]
    labels: tuple[str, ...]


@torch.inference_mode()
def _score_window_block(
    model: GaussianForecaster,
    block: WindowBlock,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    score_parts: list[np.ndarray] = []
    for start in range(0, len(block.x), batch_size):
        end = start + batch_size
        xb = torch.from_numpy(block.x[start:end]).to(device, non_blocking=True)
        yb = torch.from_numpy(block.y[start:end]).to(device, non_blocking=True)
        mb = torch.from_numpy(block.y_mask[start:end]).to(device, non_blocking=True)
        mean, scale = model(xb)
        _, cells = _masked_gaussian_nll(mean, scale, yb, mb)
        weights = mb.to(cells.dtype)
        score = (cells * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        score_parts.append(score.cpu().numpy().astype(np.float64, copy=False))
    weights_np = block.y_mask.astype(np.float64)
    magnitude = np.sqrt(
        (np.square(block.y, dtype=np.float64) * weights_np).sum(axis=1)
        / np.maximum(weights_np.sum(axis=1), 1.0)
    )
    scores = np.concatenate(score_parts) if score_parts else np.empty(0, np.float64)
    return scores, magnitude


def _score_sources(
    access: DatasetAccess,
    role: str,
    scaler: RobustScaler,
    model: GaussianForecaster,
    common: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    stage: str,
) -> ScoreSet:
    sources = access.roles[role]
    scores: list[np.ndarray] = []
    magnitudes: list[np.ndarray] = []
    source_codes: list[np.ndarray] = []
    target_times: list[np.ndarray] = []
    kept_sources: list[str] = []
    kept_labels: list[str] = []
    model.eval()
    for source_index, source_id in enumerate(sources, start=1):
        block = _make_windows(access.load_source(source_id), scaler, common)
        if len(block.x):
            source_score, source_magnitude = _score_window_block(
                model, block, int(common["batch_size"]), device
            )
            code = len(kept_sources)
            scores.append(source_score)
            magnitudes.append(source_magnitude)
            source_codes.append(np.full(len(source_score), code, dtype=np.int32))
            target_times.append(block.target_time.astype(np.float64, copy=False))
            kept_sources.append(source_id)
            kept_labels.append(access.labels[source_id])
        if source_index == 1 or source_index % 25 == 0 or source_index == len(sources):
            _write_json_atomic(
                run_dir / "training_progress.json",
                {
                    "schema_version": 1,
                    "dataset": access.dataset,
                    "stage": stage,
                    "source_index": source_index,
                    "source_count": len(sources),
                    "scored_windows": int(sum(map(len, scores))),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"{stage} source={source_index}/{len(sources)}", flush=True)
    if not scores:
        raise ProbabilisticV1Error(f"Role {role!r} produced zero windows")
    return ScoreSet(
        scores=np.concatenate(scores),
        magnitudes=np.concatenate(magnitudes),
        source_codes=np.concatenate(source_codes),
        target_times=np.concatenate(target_times),
        sources=tuple(kept_sources),
        labels=tuple(kept_labels),
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return float("nan")
    left_rank = pd.Series(left[finite]).rank(method="average").to_numpy(float)
    right_rank = pd.Series(right[finite]).rank(method="average").to_numpy(float)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _safe_binary_metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    finite = np.isfinite(score)
    truth = truth[finite].astype(np.int8, copy=False)
    score = score[finite]
    if len(np.unique(truth)) < 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(truth, score)),
        "average_precision": float(average_precision_score(truth, score)),
    }


def _build_evaluation(
    scored: ScoreSet,
    normal_label: str,
    threshold: float,
    run_dir: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    window_truth = np.zeros(len(scored.scores), dtype=np.int8)
    for code, (source_id, label) in enumerate(zip(scored.sources, scored.labels)):
        index = scored.source_codes == code
        score = scored.scores[index]
        times = scored.target_times[index]
        is_anomaly = label != normal_label
        window_truth[index] = int(is_anomaly)
        alarms = score >= threshold
        finite_times = times[np.isfinite(times)]
        duration = (
            float(finite_times.max() - finite_times.min())
            if len(finite_times) > 1
            else 0.0
        )
        records.append(
            {
                "source_id": source_id,
                "label": label,
                "is_anomaly": int(is_anomaly),
                "windows": int(len(score)),
                "duration_s": duration,
                "mean_score": float(np.mean(score)),
                "max_score": float(np.max(score)),
                "alarm_windows": int(alarms.sum()),
                "alarm_fraction": float(alarms.mean()),
                "detected": int(alarms.any()),
            }
        )
    flight_frame = pd.DataFrame.from_records(records)
    temporary = run_dir / "flight_metrics.csv.tmp"
    flight_frame.to_csv(temporary, index=False)
    _replace_with_retry(temporary, run_dir / "flight_metrics.csv")
    flight_truth = flight_frame["is_anomaly"].to_numpy(np.int8)
    flight_score = flight_frame["max_score"].to_numpy(float)
    normal_flights = flight_frame.loc[flight_frame["is_anomaly"].eq(0)]
    anomalous_flights = flight_frame.loc[flight_frame["is_anomaly"].eq(1)]
    normal_hours = float(normal_flights["duration_s"].sum() / 3600.0)
    label_summary: dict[str, Any] = {}
    for label, group in flight_frame.groupby("label", sort=True):
        label_summary[str(label)] = {
            "flights": int(len(group)),
            "detected_flights": int(group["detected"].sum()),
            "detection_rate": float(group["detected"].mean()),
            "median_max_score": float(group["max_score"].median()),
        }
    return {
        "threshold": threshold,
        "window_count": int(len(scored.scores)),
        "flight_count": int(len(flight_frame)),
        "window_level": _safe_binary_metrics(window_truth, scored.scores),
        "flight_level": _safe_binary_metrics(flight_truth, flight_score),
        "normal_flights": int(len(normal_flights)),
        "anomalous_flights": int(len(anomalous_flights)),
        "normal_flights_with_any_alarm": int(normal_flights["detected"].sum()),
        "normal_flight_alarm_fraction": (
            float(normal_flights["detected"].mean()) if len(normal_flights) else None
        ),
        "anomalous_flight_detection_rate": (
            float(anomalous_flights["detected"].mean()) if len(anomalous_flights) else None
        ),
        "normal_alarm_windows_per_hour": (
            float(normal_flights["alarm_windows"].sum() / normal_hours)
            if normal_hours > 0
            else None
        ),
        "normal_observed_hours": normal_hours,
        "by_label": label_summary,
        "warning": (
            "Window alarms are not debounced events; alarm-windows/hour must not be "
            "compared directly with prior event-level false-alarm metrics."
        ),
    }


def _magnitude_diagnostic(
    trained: ScoreSet,
    random_init: ScoreSet,
    common: dict[str, Any],
) -> dict[str, Any]:
    if trained.sources != random_init.sources or not np.array_equal(
        trained.source_codes, random_init.source_codes
    ):
        raise ProbabilisticV1Error("Random-init diagnostic window alignment failed")
    count = min(
        len(trained.scores), int(common["magnitude_diagnostic_max_windows"])
    )
    if count < 3:
        raise ProbabilisticV1Error("Too few validation windows for magnitude diagnostic")
    rng = np.random.default_rng(int(common["seed"]) + 7001)
    index = (
        np.arange(len(trained.scores), dtype=np.int64)
        if count == len(trained.scores)
        else np.sort(rng.choice(len(trained.scores), size=count, replace=False))
    )
    trained_vs_random = _spearman(
        trained.scores[index], random_init.scores[index]
    )
    trained_vs_magnitude = _spearman(
        trained.scores[index], trained.magnitudes[index]
    )
    if not np.isfinite(trained_vs_random) or not np.isfinite(trained_vs_magnitude):
        raise ProbabilisticV1Error("Magnitude diagnostic correlation is undefined")
    gate = float(common["magnitude_domination_rho_gate"])
    flagged = trained_vs_random >= gate or trained_vs_magnitude >= gate
    return {
        "sampled_validation_windows": count,
        "spearman_trained_score_vs_random_init_score": trained_vs_random,
        "spearman_trained_score_vs_standardized_target_magnitude": trained_vs_magnitude,
        "gate": gate,
        "flagged": bool(flagged),
    }


def _write_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    _write_json_atomic(run_dir / "training_history.json", history)
    temporary = run_dir / "training_history.csv.tmp"
    pd.DataFrame(history).to_csv(temporary, index=False)
    _replace_with_retry(temporary, run_dir / "training_history.csv")


def _checkpoint_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _contract_evidence(
    root: Path,
    config: dict[str, Any],
    access: DatasetAccess,
) -> dict[str, Any]:
    config_path = root / CONFIG_PATH
    prereg_path = root / PREREG_PATH
    if not prereg_path.is_file():
        raise ProbabilisticV1Error(f"Missing preregistration: {prereg_path}")
    roles = {role: list(ids) for role, ids in access.roles.items()}
    evidence = {
        "candidate_namespace": config["candidate_namespace"],
        "config_sha256": _sha256(config_path),
        "preregistration_sha256": _sha256(prereg_path),
        "data_fingerprint": access.data_fingerprint,
        "roles_sha256": _canonical_sha(roles),
        "role_counts": {role: len(ids) for role, ids in access.roles.items()},
    }
    evidence["contract_sha256"] = _canonical_sha(evidence)
    return evidence


def _inventory(
    access: DatasetAccess,
    scaler: RobustScaler,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    role_labels: dict[str, dict[str, int]] = {}
    for role, sources in access.roles.items():
        counts: dict[str, int] = {}
        for source_id in sources:
            label = access.labels[source_id]
            counts[label] = counts.get(label, 0) + 1
        role_labels[role] = dict(sorted(counts.items()))
    return {
        "dataset": access.dataset,
        "contract_sha256": evidence["contract_sha256"],
        "roles": evidence["role_counts"],
        "role_labels": role_labels,
        "requested_features": list(access.features),
        "active_features": list(scaler.features),
        "excluded_degenerate_features": list(scaler.excluded),
        "data_fingerprint": access.data_fingerprint,
    }


def _run_training(
    root: Path,
    dataset: str,
    run_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    config, spec = _load_contract(root, dataset)
    common = config["common"]
    access = _load_access(root, config, spec, dataset)
    evidence = _contract_evidence(root, config, access)
    scaler = _fit_scaler_access(access, common, str(spec["normal_label"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    scaler_dict = scaler.to_dict()
    scaler_path = run_dir / "scaler.json"
    if scaler_path.exists():
        saved_scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
        if _canonical_sha(saved_scaler) != _canonical_sha(scaler_dict):
            raise ProbabilisticV1Error("Saved scaler does not match frozen train data")
    else:
        _write_json_atomic(scaler_path, scaler_dict)

    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("contract_sha256") != evidence["contract_sha256"]:
            raise ProbabilisticV1Error("Run directory belongs to a different contract")
    else:
        run_manifest = {
            "schema_version": 1,
            "candidate_namespace": config["candidate_namespace"],
            "dataset": dataset,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "training",
            "contract_sha256": evidence["contract_sha256"],
            "contract_evidence": evidence,
            "config": config,
            "dataset_spec": spec,
            "scaler": scaler_dict,
        }
        _write_json_atomic(run_manifest_path, run_manifest)

    report_path = run_dir / "training_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract_sha256") != evidence["contract_sha256"]:
            raise ProbabilisticV1Error("Existing report contract mismatch")
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        return report

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ProbabilisticV1Error(
            "CUDA was requested but is unavailable; select a GPU Colab runtime"
        )
    device = torch.device(device_name)
    _seed_everything(int(common["seed"]))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    model = GaussianForecaster(
        input_size=len(scaler.features) * 2 + 1,
        channels=len(scaler.features),
        common=common,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(common["learning_rate"]))
    history: list[dict[str, Any]] = []
    completed_epochs = 0
    checkpoint_path = run_dir / "training_epoch_checkpoint.pt"
    select_best_validation = bool(common.get("select_best_validation_checkpoint", False))
    best_checkpoint_path = run_dir / "best_validation_checkpoint.pt"
    best_validation_nll = math.inf
    best_epoch = 0
    if checkpoint_path.exists():
        checkpoint = _checkpoint_load(checkpoint_path, device)
        if checkpoint.get("contract_sha256") != evidence["contract_sha256"]:
            raise ProbabilisticV1Error("Checkpoint contract mismatch")
        if checkpoint.get("dataset") != dataset:
            raise ProbabilisticV1Error("Checkpoint dataset mismatch")
        completed_epochs = int(checkpoint["completed_epochs"])
        history = list(checkpoint["history"])
        if completed_epochs != len(history):
            raise ProbabilisticV1Error("Checkpoint epoch/history mismatch")
        if completed_epochs > int(common["epochs"]):
            raise ProbabilisticV1Error("Checkpoint exceeds frozen epoch count")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        if history:
            best_row = min(
                history,
                key=lambda row: (
                    float(row["mean_validation_gaussian_nll"]), int(row["epoch"])
                ),
            )
            best_validation_nll = float(best_row["mean_validation_gaussian_nll"])
            best_epoch = int(best_row["epoch"])
        print(f"resumed completed_epochs={completed_epochs}", flush=True)

    for epoch_index in range(completed_epochs, int(common["epochs"])):
        epoch_row = _train_epoch(
            access,
            scaler,
            model,
            optimizer,
            common,
            device,
            epoch_index,
            run_dir,
        )
        validation = _score_sources(
            access,
            "val",
            scaler,
            model,
            common,
            device,
            run_dir,
            stage=f"validation_epoch_{epoch_index + 1}",
        )
        epoch_row["mean_validation_gaussian_nll"] = float(np.mean(validation.scores))
        history.append(epoch_row)
        completed_epochs = epoch_index + 1
        checkpoint = {
            "schema_version": 1,
            "candidate_namespace": config["candidate_namespace"],
            "dataset": dataset,
            "contract_sha256": evidence["contract_sha256"],
            "completed_epochs": completed_epochs,
            "history": history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _torch_save_atomic(checkpoint_path, checkpoint)
        candidate_nll = float(epoch_row["mean_validation_gaussian_nll"])
        if select_best_validation and (
            candidate_nll < best_validation_nll
            or (candidate_nll == best_validation_nll and completed_epochs < best_epoch)
        ):
            best_validation_nll = candidate_nll
            best_epoch = completed_epochs
            _torch_save_atomic(
                best_checkpoint_path,
                {
                    "schema_version": 1,
                    "candidate_namespace": config["candidate_namespace"],
                    "dataset": dataset,
                    "contract_sha256": evidence["contract_sha256"],
                    "selected_epoch": best_epoch,
                    "mean_validation_gaussian_nll": best_validation_nll,
                    "model_state_dict": model.state_dict(),
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        _write_history(run_dir, history)
        print(
            f"epoch={completed_epochs}/{common['epochs']} "
            f"train_nll={epoch_row['mean_train_gaussian_nll']:.8f} "
            f"val_nll={epoch_row['mean_validation_gaussian_nll']:.8f}",
            flush=True,
        )
        del validation

    if select_best_validation:
        if not best_checkpoint_path.is_file():
            raise ProbabilisticV1Error("Best-validation checkpoint is missing")
        selected = _checkpoint_load(best_checkpoint_path, device)
        if selected.get("contract_sha256") != evidence["contract_sha256"]:
            raise ProbabilisticV1Error("Best-validation checkpoint contract mismatch")
        model.load_state_dict(selected["model_state_dict"])
        best_epoch = int(selected["selected_epoch"])
        best_validation_nll = float(selected["mean_validation_gaussian_nll"])
    else:
        best_epoch = completed_epochs
        best_validation_nll = float(history[-1]["mean_validation_gaussian_nll"])

    model_payload = {
        "schema_version": 1,
        "dataset": dataset,
        "contract_sha256": evidence["contract_sha256"],
        "features": list(scaler.features),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "selected_epoch": best_epoch,
        "selection_metric": "normal_validation_gaussian_nll",
    }
    _torch_save_atomic(run_dir / "model_state.pt", model_payload)
    validation = _score_sources(
        access, "val", scaler, model, common, device, run_dir, stage="final_validation"
    )
    threshold = float(
        np.quantile(validation.scores, float(common["validation_alarm_quantile"]))
    )
    _seed_everything(int(common["seed"]) + 991)
    random_model = GaussianForecaster(
        input_size=len(scaler.features) * 2 + 1,
        channels=len(scaler.features),
        common=common,
    ).to(device)
    random_validation = _score_sources(
        access,
        "val",
        scaler,
        random_model,
        common,
        device,
        run_dir,
        stage="random_init_diagnostic",
    )
    diagnostic = _magnitude_diagnostic(validation, random_validation, common)
    del random_model, random_validation
    test_scored = _score_sources(
        access, "test", scaler, model, common, device, run_dir, stage="test_evaluation"
    )
    evaluation = _build_evaluation(
        test_scored, str(spec["normal_label"]), threshold, run_dir
    )
    _write_json_atomic(
        run_dir / "validation_summary.json",
        {
            "normal_validation_windows": int(len(validation.scores)),
            "score_mean": float(np.mean(validation.scores)),
            "score_std": float(np.std(validation.scores)),
            "score_min": float(np.min(validation.scores)),
            "score_max": float(np.max(validation.scores)),
            "alarm_quantile": float(common["validation_alarm_quantile"]),
            "alarm_threshold": threshold,
        },
    )
    run_manifest["status"] = "complete"
    run_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["completed_epochs"] = completed_epochs
    _write_json_atomic(run_manifest_path, run_manifest)
    artifact_names = [
        "run_manifest.json",
        "scaler.json",
        "training_history.json",
        "training_history.csv",
        "training_epoch_checkpoint.pt",
        "model_state.pt",
        "validation_summary.json",
        "flight_metrics.csv",
    ]
    if select_best_validation:
        artifact_names.append("best_validation_checkpoint.pt")
    artifact_hashes = {
        name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
        for name in artifact_names
    }
    report = {
        "schema_version": 1,
        "candidate_namespace": config["candidate_namespace"],
        "dataset": dataset,
        "contract_sha256": evidence["contract_sha256"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_epochs": completed_epochs,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "evaluation_status": spec["evaluation_status"],
        "active_features": list(scaler.features),
        "excluded_degenerate_features": list(scaler.excluded),
        "final_epoch": history[-1],
        "selected_checkpoint": {
            "policy": (
                "lowest_normal_validation_gaussian_nll_tie_earliest_epoch"
                if select_best_validation
                else "final_epoch"
            ),
            "epoch": best_epoch,
            "mean_validation_gaussian_nll": best_validation_nll,
        },
        "validation_alarm_quantile": float(common["validation_alarm_quantile"]),
        "validation_alarm_threshold": threshold,
        "magnitude_diagnostic": diagnostic,
        "magnitude_domination_flagged_at_0_8": diagnostic["flagged"],
        "evaluation": evaluation,
        "artifact_hashes": artifact_hashes,
        "interpretation_constraint": (
            "Exploratory development result; it does not replace prior official "
            "dataset-specific gates or open any frozen holdout."
        ),
    }
    _write_json_atomic(report_path, report)
    _write_json_atomic(
        run_dir / "training_progress.json",
        {
            "schema_version": 1,
            "dataset": dataset,
            "stage": "complete",
            "completed_epochs": completed_epochs,
            "training_report": "training_report.json",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "inventory", "train"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--dataset", required=True, choices=("alfa", "uav_attack", "uav_sead", "rflymad")
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.repo_root.resolve()
    try:
        if args.command in {"verify", "inventory"}:
            config, spec = _load_contract(root, args.dataset)
            access = _load_access(root, config, spec, args.dataset)
            scaler = _fit_scaler_access(access, config["common"], str(spec["normal_label"]))
            evidence = _contract_evidence(root, config, access)
            print(
                json.dumps(_inventory(access, scaler, evidence), indent=2, ensure_ascii=False),
                flush=True,
            )
            return 0
        if args.run_dir is None:
            raise ProbabilisticV1Error("--run-dir is required for train")
        _run_training(root, args.dataset, args.run_dir.resolve(), args.device)
        return 0
    except ProbabilisticV1Error as exc:
        print(f"CONTRACT ERROR: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
