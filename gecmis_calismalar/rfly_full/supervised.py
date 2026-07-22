"""Supervised temporal RflyMAD-Full v2 baseline.

This track consumes only the causal 10 Hz v2 parquets.  Training uses NoFault
windows and windows wholly inside an active system-fault interval.  Wind is held
out as an environmental robustness set, and mixed transition windows do not
contribute training loss.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from rfly_full.contract import DATASET_MANIFEST, V2_ROOT
from rfly_full.pipeline import _atomic_json
from rfly_full.v2_parser import PARSED_10HZ_ROOT, PARSE_STATE, SAMPLE_HZ, V2_FEATURES

SUPERVISED_ROOT = V2_ROOT / "supervised_tcn"
WINDOW_SECONDS = 20
TRAIN_STRIDE_SECONDS = 5
EVAL_STRIDE_SECONDS = 1
ALARM_K = 4
ALARM_N_SECONDS = 6
REFRACTORY_SECONDS = 30
SEED = 20260721


def development_run_status(
    *, epochs: int, parse_complete: bool, development_smoke_fold: int | None,
) -> str:
    """Classify a run without granting development-only work an operational label."""
    if epochs < 12 or not parse_complete:
        return "smoke_only"
    if development_smoke_fold is not None:
        return "development_only"
    return "complete"


@dataclass(frozen=True)
class WindowSet:
    x: np.ndarray
    binary: np.ndarray
    family: np.ndarray
    meta: pd.DataFrame


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.ReLU(), nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.ReLU(), nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalFaultClassifier(nn.Module):
    """Small TCN with binary anomaly and conditional fault-family heads."""

    def __init__(self, channels_in: int, families: int, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels_in, hidden, 1), nn.ReLU(),
            ResidualTemporalBlock(hidden, 1),
            ResidualTemporalBlock(hidden, 2),
            ResidualTemporalBlock(hidden, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.binary_head = nn.Linear(hidden, 1)
        self.family_head = nn.Linear(hidden, families)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x.transpose(1, 2)).squeeze(-1)
        return self.binary_head(encoded).squeeze(-1), self.family_head(encoded)


def alarm_onsets(
    scores: np.ndarray, threshold: float, *, k: int = ALARM_K,
    n_seconds: int = ALARM_N_SECONDS,
    refractory_seconds: int = REFRACTORY_SECONDS,
) -> np.ndarray:
    """Time-based K-of-N policy; input is one calibrated decision per second."""
    scores = np.asarray(scores, dtype=float)
    above = np.isfinite(scores) & (scores >= threshold)
    counts = np.convolve(above.astype(np.int16), np.ones(n_seconds, np.int16), mode="full")[: len(above)]
    state = counts >= k
    raw = np.flatnonzero(state & ~np.r_[False, state[:-1]])
    keep: list[int] = []
    next_allowed = 0
    for index in raw:
        if index >= next_allowed:
            keep.append(int(index))
            next_allowed = int(index) + refractory_seconds
    result = np.zeros(len(scores), dtype=bool)
    result[keep] = True
    return result


def training_window_label(fault_active: np.ndarray, role: str) -> int | None:
    """Return 0/1 for eligible training windows, None for excluded windows."""
    active = np.asarray(fault_active, dtype=bool)
    if role == "normal_reference" and not active.any():
        return 0
    if role == "fault_detection" and len(active) and active.all():
        return 1
    return None


def _available_manifest() -> pd.DataFrame:
    manifest = pd.read_parquet(DATASET_MANIFEST)
    manifest = manifest.drop_duplicates("canonical_case_id")
    available = {path.stem for path in PARSED_10HZ_ROOT.glob("*/*.parquet")}
    return manifest[manifest["canonical_case_id"].isin(available)].copy()


def _load_flight(canonical: str, domain: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(PARSED_10HZ_ROOT / domain / f"{canonical}.parquet", columns=columns)


def _fit_scaler(manifest: pd.DataFrame, columns: list[str]) -> dict:
    samples = []
    for row in manifest.itertuples(index=False):
        frame = _load_flight(str(row.canonical_case_id), str(row.domain), columns)
        step = max(1, len(frame) // 250)
        samples.append(frame.iloc[::step][columns])
    data = pd.concat(samples, ignore_index=True)
    coverage = data.notna().mean()
    selected = [column for column in columns if coverage[column] >= 0.70]
    if len(selected) < 8:
        raise RuntimeError(f"only {len(selected)} features meet 70% train coverage")
    medians = data[selected].median()
    scales = (data[selected].quantile(0.75) - data[selected].quantile(0.25)).replace(0, 1).fillna(1)
    return {
        "features": selected,
        "medians": medians.to_dict(),
        "scales": scales.to_dict(),
        "clip": 10.0,
        "fit_scope": "development train groups only",
    }


def _scaled_values(frame: pd.DataFrame, scaler: dict) -> np.ndarray:
    columns = scaler["features"]
    medians = pd.Series(scaler["medians"])
    scales = pd.Series(scaler["scales"])
    observed = frame[columns].notna().to_numpy(np.float32)
    values = ((frame[columns].fillna(medians) - medians) / scales).clip(
        -scaler["clip"], scaler["clip"]
    ).to_numpy(np.float32)
    # Missingness is an explicit input channel, not an accidental zero value.
    return np.concatenate([values, observed], axis=1)


def _window_read_columns(scaler: dict) -> list[str]:
    return [
        "t_rel_s", "fault_active", "fault_family", "fault_subtype",
        "evaluation_role", *scaler["features"],
    ]


def _flight_windows(
    row, scaler: dict, columns: list[str], length: int, stride: int,
):
    """Yield one flight's sliding windows lazily.

    Only a single flight's parquet is ever resident in memory at a time; the
    caller decides what to do with each window (reservoir-sample it for
    training, or score it immediately and discard it for evaluation) instead
    of every window for every flight being materialized up front.
    """
    flight = _load_flight(str(row.canonical_case_id), str(row.domain), columns)
    values = _scaled_values(flight, scaler)
    active = flight["fault_active"].to_numpy(bool)
    t_rel = flight["t_rel_s"].to_numpy()
    for end in range(length - 1, len(flight), stride):
        start = end - length + 1
        yield values[start : end + 1], float(t_rel[end]), bool(active[end]), active[start : end + 1]


def _reservoir_update(reservoir: list, seen: int, capacity: int, item, rng: np.random.Generator) -> int:
    """Algorithm R: uniform random sample of `capacity` items from a stream of
    unknown length, without ever holding more than `capacity` items."""
    seen += 1
    if len(reservoir) < capacity:
        reservoir.append(item)
    else:
        slot = int(rng.integers(0, seen))
        if slot < capacity:
            reservoir[slot] = item
    return seen


def _build_training_windows(
    manifest: pd.DataFrame, scaler: dict, family_index: dict[str, int], *,
    stride_seconds: int, max_windows: int, seed: int,
) -> WindowSet:
    """Eligible training windows, reservoir-sampled per class while streaming.

    Statistically equivalent to building every eligible window across the
    whole manifest and then uniformly subsampling `max_windows` of them (the
    old `_build_windows` + `_cap_balanced` two-step), but peak memory is
    bounded by `max_windows` regardless of how many flights or how many
    eligible windows the manifest actually contains.
    """
    length = WINDOW_SECONDS * SAMPLE_HZ
    stride = stride_seconds * SAMPLE_HZ
    columns = _window_read_columns(scaler)
    capacity = {0: max_windows // 2, 1: max_windows - max_windows // 2}
    reservoirs: dict[int, list] = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    rng = np.random.default_rng(seed)
    for row in manifest.itertuples(index=False):
        family_name = str(row.fault_family)
        family_label = family_index.get(family_name, -1)
        for window, t_end_s, _active_at_end, active_slice in _flight_windows(
            row, scaler, columns, length, stride,
        ):
            target = training_window_label(active_slice, str(row.evaluation_role))
            if target is None:
                continue
            item = (
                window, target, family_label if target else -1,
                {
                    "canonical_case_id": str(row.canonical_case_id), "t_end_s": t_end_s,
                    "domain": str(row.domain), "fault_family": family_name,
                    "fault_subtype": str(row.fault_subtype),
                    "evaluation_role": str(row.evaluation_role), "fault_active": bool(target),
                },
            )
            seen[target] = _reservoir_update(reservoirs[target], seen[target], capacity[target], item, rng)
    items = reservoirs[0] + reservoirs[1]
    feature_count = len(scaler["features"]) * 2
    if not items:
        return WindowSet(
            np.empty((0, length, feature_count), np.float32),
            np.empty(0, np.int64), np.empty(0, np.int64), pd.DataFrame(),
        )
    order = rng.permutation(len(items))
    windows, binary, family, metadata = zip(*(items[index] for index in order))
    return WindowSet(
        np.stack(windows).astype(np.float32), np.asarray(binary, np.int64),
        np.asarray(family, np.int64), pd.DataFrame(list(metadata)),
    )


def _score_streaming(
    manifest: pd.DataFrame, model: nn.Module, scaler: dict, *,
    stride_seconds: int, batch_size: int = 512,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Dense per-second inference windows, scored and discarded flight by flight.

    Replaces building every dense evaluation window for a whole split (which
    can run into the hundreds of thousands for the locked-test partition)
    before running the model. Only the lightweight outputs — per-window
    probability logits, family logits, and scalar metadata — survive past
    each flush; the raw window tensors never accumulate beyond `batch_size`.
    """
    length = WINDOW_SECONDS * SAMPLE_HZ
    stride = stride_seconds * SAMPLE_HZ
    columns = _window_read_columns(scaler)
    metadata: list[dict] = []
    binary_chunks: list[np.ndarray] = []
    family_chunks: list[np.ndarray] = []
    window_buffer: list[np.ndarray] = []

    def flush() -> None:
        if not window_buffer:
            return
        batch = torch.from_numpy(np.stack(window_buffer).astype(np.float32))
        binary, family = model(batch)
        binary_chunks.append(binary.numpy())
        family_chunks.append(family.numpy())
        window_buffer.clear()

    model.eval()
    with torch.no_grad():
        for row in manifest.itertuples(index=False):
            family_name = str(row.fault_family)
            for window, t_end_s, active_at_end, _active_slice in _flight_windows(
                row, scaler, columns, length, stride,
            ):
                window_buffer.append(window)
                metadata.append({
                    "canonical_case_id": str(row.canonical_case_id), "t_end_s": t_end_s,
                    "domain": str(row.domain), "fault_family": family_name,
                    "fault_subtype": str(row.fault_subtype),
                    "evaluation_role": str(row.evaluation_role), "fault_active": bool(active_at_end),
                })
                if len(window_buffer) >= batch_size:
                    flush()
        flush()
    if not metadata:
        empty_meta = pd.DataFrame(columns=[
            "canonical_case_id", "t_end_s", "domain", "fault_family",
            "fault_subtype", "evaluation_role", "fault_active",
        ])
        return empty_meta, np.empty(0, np.float32), np.empty((0, model.family_head.out_features), np.float32)
    return pd.DataFrame(metadata), np.concatenate(binary_chunks), np.concatenate(family_chunks)


def _validation_loss(
    model: nn.Module, validation: WindowSet, binary_loss: nn.Module, family_loss: nn.Module,
    *, batch_size: int = 512,
) -> float:
    """Batched validation forward pass so a large capped validation set never
    needs a single (validation_size, window_length, channels) tensor at once."""
    model.eval()
    chunk_losses = []
    with torch.no_grad():
        for start in range(0, len(validation.x), batch_size):
            end = start + batch_size
            x = torch.from_numpy(validation.x[start:end])
            y = torch.from_numpy(validation.binary[start:end])
            family = torch.from_numpy(validation.family[start:end])
            binary_logits, family_logits = model(x)
            loss = binary_loss(binary_logits, y.float())
            positive = family >= 0
            if positive.any():
                loss = loss + family_loss(family_logits[positive], family[positive])
            chunk_losses.append(float(loss))
    return float(np.mean(chunk_losses)) if chunk_losses else float("nan")


def _train(
    model: nn.Module, train: WindowSet, validation: WindowSet, *, epochs: int,
    seed: int,
) -> tuple[nn.Module, list[dict]]:
    torch.manual_seed(seed)
    labels = train.binary
    counts = np.bincount(labels, minlength=2).clip(min=1)
    weights = np.where(labels == 1, 1 / counts[1], 1 / counts[0])
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    dataset = TensorDataset(
        torch.from_numpy(train.x), torch.from_numpy(train.binary),
        torch.from_numpy(train.family),
    )
    loader = DataLoader(dataset, batch_size=128, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    binary_loss = nn.BCEWithLogitsLoss()
    family_counts = np.bincount(train.family[train.family >= 0], minlength=model.family_head.out_features).clip(min=1)
    family_weight = torch.tensor(family_counts.sum() / family_counts, dtype=torch.float32)
    family_loss = nn.CrossEntropyLoss(weight=family_weight)
    best_state = None
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y, family in loader:
            binary_logits, family_logits = model(x)
            loss = binary_loss(binary_logits, y.float())
            positive = family >= 0
            if positive.any():
                loss = loss + family_loss(family_logits[positive], family[positive])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_loss = _validation_loss(model, validation, binary_loss, family_loss)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("TCN did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history


def _temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    value = torch.tensor([1.0], requires_grad=True)
    x = torch.from_numpy(logits.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))
    optimizer = torch.optim.LBFGS([value], lr=0.05, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(x / value.clamp(0.05, 20), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(value.detach().clamp(0.05, 20))


def _evaluate(meta: pd.DataFrame, probability: np.ndarray, threshold: float) -> tuple[dict, pd.DataFrame]:
    data = meta.copy()
    data["probability"] = probability
    rows = []
    detected = events = false_events = 0
    flight_tp = flight_fn = flight_fp = flight_tn = 0
    normal_seconds = 0
    delays = []
    for canonical, flight in data.groupby("canonical_case_id", sort=False):
        flight = flight.sort_values("t_end_s")
        onsets = alarm_onsets(flight["probability"].to_numpy(), threshold)
        truth = flight["fault_active"].to_numpy(bool)
        role = str(flight["evaluation_role"].iloc[0])
        false_count = int((onsets & ~truth).sum())
        false_events += false_count
        normal_seconds += int((~truth).sum())
        hit = False
        predicted_alarm = bool(onsets.any())
        delay = math.nan
        if role == "fault_detection":
            events += 1
            inside = np.flatnonzero(onsets & truth)
            hit = bool(len(inside))
            detected += int(hit)
            flight_tp += int(hit)
            flight_fn += int(not hit)
            if hit:
                onset_time = float(flight.loc[truth, "t_end_s"].min())
                delay = float(flight["t_end_s"].iloc[inside[0]] - onset_time)
                delays.append(delay)
        elif role == 'normal_reference':
            flight_fp += int(predicted_alarm)
            flight_tn += int(not predicted_alarm)
        rows.append({
            "canonical_case_id": canonical, "domain": flight["domain"].iloc[0],
            "fault_family": flight["fault_family"].iloc[0],
            "evaluation_role": role, "detected": hit,
            "false_alarm_events": false_count, "detection_delay_s": delay,
            "normal_exposure_s": int((~truth).sum()),
        })
    metrics = {
        "events": events, "detected_events": detected,
        "event_recall": detected / events if events else None,
        "false_alarm_events": false_events,
        "normal_exposure_hours": normal_seconds / 3600,
        "false_alarms_per_hour": false_events / (normal_seconds / 3600) if normal_seconds else None,
        "median_detection_delay_s": float(np.median(delays)) if delays else None,
    }
    metrics['confusion_flight'] = {
        'tp': flight_tp, 'fn': flight_fn, 'fp': flight_fp, 'tn': flight_tn,
    }
    metrics.update(
        flight_tp=flight_tp, flight_fn=flight_fn,
        flight_fp=flight_fp, flight_tn=flight_tn,
    )
    return metrics, pd.DataFrame(rows)


def _fit_threshold(meta: pd.DataFrame, probabilities: np.ndarray, budget: float) -> tuple[float, dict]:
    finite = probabilities[np.isfinite(probabilities)]
    candidates = np.unique(np.quantile(finite, np.linspace(0.50, 0.999, 100)))
    feasible = []
    all_rows = []
    for threshold in candidates:
        metrics, _ = _evaluate(meta, probabilities, float(threshold))
        row = {"threshold": float(threshold), **metrics}
        all_rows.append(row)
        if metrics["false_alarms_per_hour"] is not None and metrics["false_alarms_per_hour"] <= budget:
            feasible.append(row)
    chosen = max(
        feasible or all_rows,
        key=lambda row: (row["event_recall"] or 0, -(row["false_alarms_per_hour"] or float("inf"))),
    )
    return float(chosen["threshold"]), chosen


def run(
    *, validation_fold: int = 0, held_out_family: str | None = None,
    epochs: int = 12, max_train_windows: int = 50_000, max_val_windows: int = 20_000,
    protocol: str = 'full', development_smoke_fold: int | None = None,
) -> Path:
    manifest = _available_manifest()
    if manifest.empty:
        raise RuntimeError("no v2 10 Hz flights; run the v2 parser first")
    development = manifest[manifest["split"].eq("development")]
    if development_smoke_fold == validation_fold:
        raise ValueError("development smoke fold must differ from validation fold")
    excluded_folds = [validation_fold]
    if development_smoke_fold is not None:
        excluded_folds.append(development_smoke_fold)
    train_manifest = development[~development["cv_fold"].isin(excluded_folds)]
    val_manifest = development[development["cv_fold"].eq(validation_fold)]
    test_manifest = (
        development[development["cv_fold"].eq(development_smoke_fold)]
        if development_smoke_fold is not None
        else manifest[manifest["split"].eq("locked_test")]
    )
    if protocol == 'simulation_to_real':
        train_manifest = train_manifest[train_manifest.domain.isin(['SIL', 'HIL'])]
        val_manifest = val_manifest[val_manifest.domain.isin(['SIL', 'HIL'])]
        test_manifest = test_manifest[test_manifest.domain.eq('Real')]
    elif protocol == 'real_only':
        train_manifest = train_manifest[train_manifest.domain.eq('Real')]
        val_manifest = val_manifest[val_manifest.domain.eq('Real')]
        test_manifest = test_manifest[test_manifest.domain.eq('Real')]
    elif protocol != 'full':
        raise ValueError(f'unknown protocol: {protocol}')
    if held_out_family:
        train_manifest = train_manifest[train_manifest["fault_family"].ne(held_out_family)]
        val_manifest = val_manifest[val_manifest["fault_family"].ne(held_out_family)]
    if train_manifest.empty or val_manifest.empty or test_manifest.empty:
        raise RuntimeError("train/validation/test manifests must all be non-empty")
    train_manifest = train_manifest[train_manifest["evaluation_role"].ne("environment_robustness")]
    families = sorted(train_manifest.loc[train_manifest["system_fault"], "fault_family"].unique())
    if not families:
        raise RuntimeError("training split has no active fault families")
    family_index = {family: index for index, family in enumerate(families)}
    scaler = _fit_scaler(train_manifest[train_manifest["evaluation_role"].eq("normal_reference")], list(V2_FEATURES))

    # Reservoir-capped: peak memory is bounded by max_train_windows /
    # max_val_windows regardless of how many flights the manifest holds.
    train = _build_training_windows(
        train_manifest, scaler, family_index,
        stride_seconds=TRAIN_STRIDE_SECONDS, max_windows=max_train_windows, seed=SEED,
    )
    validation_train = _build_training_windows(
        val_manifest, scaler, family_index,
        stride_seconds=TRAIN_STRIDE_SECONDS, max_windows=max_val_windows, seed=SEED + 1,
    )
    if not len(train.x) or not len(validation_train.x):
        raise RuntimeError("train/validation windows must both be non-empty")
    torch.manual_seed(SEED)
    model = TemporalFaultClassifier(train.x.shape[-1], len(families))
    model, history = _train(model, train, validation_train, epochs=epochs, seed=SEED)

    # Dense per-second scoring windows are built and scored flight by flight
    # (never all at once) since the locked-test partition alone can hold
    # hundreds of thousands of them.
    validation_meta, val_logits, val_family_logits = _score_streaming(
        val_manifest, model, scaler, stride_seconds=EVAL_STRIDE_SECONDS,
    )
    test_meta, test_logits, test_family_logits = _score_streaming(
        test_manifest, model, scaler, stride_seconds=EVAL_STRIDE_SECONDS,
    )
    if validation_meta.empty or test_meta.empty:
        raise RuntimeError("validation/test evaluation windows must both be non-empty")
    validation_binary = validation_meta["fault_active"].to_numpy(int)
    temperature = _temperature(val_logits, validation_binary)
    val_probability = 1 / (1 + np.exp(-val_logits / temperature))
    test_probability = 1 / (1 + np.exp(-test_logits / temperature))

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output = SUPERVISED_ROOT / run_name
    output.mkdir(parents=True, exist_ok=False)
    policies = {}
    metric_rows = []
    flight_frames = []
    for name, budget in (("critical", 2.0), ("advisory", 12.0)):
        threshold, validation_metrics = _fit_threshold(validation_meta, val_probability, budget)
        metrics, flights = _evaluate(test_meta, test_probability, threshold)
        policies[name] = {
            "threshold": threshold, "budget_fa_per_hour": budget,
            "validation": validation_metrics,
            "alarm": f"{ALARM_K}-of-{ALARM_N_SECONDS}s; refractory={REFRACTORY_SECONDS}s",
        }
        metric_rows.append({"policy": name, **metrics})
        flights["policy"] = name
        flight_frames.append(flights)
    metrics = pd.DataFrame(metric_rows)
    flights = pd.concat(flight_frames, ignore_index=True)
    family_breakdown = (
        flights[flights["evaluation_role"].eq("fault_detection")]
        .groupby(["policy", "domain", "fault_family"], dropna=False)
        .agg(flights=("canonical_case_id", "nunique"), detected=("detected", "sum"))
        .reset_index()
    )
    family_breakdown["recall"] = family_breakdown["detected"] / family_breakdown["flights"]
    test_binary = test_meta["fault_active"].to_numpy(int)
    test_family = np.where(
        test_meta["fault_active"].to_numpy(bool),
        test_meta["fault_family"].map(family_index).fillna(-1).to_numpy(),
        -1,
    ).astype(np.int64)
    active = test_binary == 1
    predicted_family = test_family_logits.argmax(axis=1)
    family_confusion = confusion_matrix(
        test_family[active & (test_family >= 0)],
        predicted_family[active & (test_family >= 0)],
        labels=np.arange(len(families)),
    )
    pd.DataFrame(family_confusion, index=families, columns=families).to_csv(output / "fault_family_confusion.csv")
    metrics.to_csv(output / "operational_metrics.csv", index=False)
    flights.to_csv(output / "per_flight_metrics.csv", index=False)
    family_breakdown.to_csv(output / "domain_family_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    _atomic_json(output / "policies.json", policies)
    checkpoint = {
        "state_dict": model.state_dict(), "features": scaler["features"],
        "families": families, "temperature": temperature,
        "sample_hz": SAMPLE_HZ, "window_seconds": WINDOW_SECONDS,
    }
    torch.save(checkpoint, output / "model.pt")
    summary = {
        "status": "complete" if len(test_manifest) else "development_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "supervised_tcn_multitask", "validation_fold": validation_fold,
        "held_out_family": held_out_family, "families": families,
        "normal_only_ae": False,
        "training_contract": "NoFault + wholly fault-active windows; transition/wind excluded",
        "calibration": "temperature scaling on labeled development validation",
        "split_contract": (
            "development-only disjoint validation/smoke folds; locked test unread"
            if development_smoke_fold is not None
            else "locked grouped test; development grouped five-fold"
        ),
        "development_smoke_fold": development_smoke_fold,
        "locked_test_features_read": development_smoke_fold is None,
        "training_flights": int(train_manifest["canonical_case_id"].nunique()),
        "validation_flights": int(val_manifest["canonical_case_id"].nunique()),
        "test_flights": int(test_manifest["canonical_case_id"].nunique()),
        "training_windows": int(len(train.x)),
        "validation_training_windows": int(len(validation_train.x)),
        "validation_eval_windows": int(len(validation_meta)),
        "test_eval_windows": int(len(test_meta)),
        "max_train_windows": max_train_windows, "max_val_windows": max_val_windows,
        "window_memory_contract": (
            "reservoir-sampled train/validation-loss windows capped at "
            "max_train_windows/max_val_windows; dense validation/test scoring "
            "windows are built and scored one flight at a time, never materialized "
            "for a whole split at once"
        ),
        "metrics": metric_rows,
    }
    _atomic_json(output / "summary.json", summary)
    parse_complete = bool(
        PARSE_STATE.exists()
        and json.loads(PARSE_STATE.read_text(encoding='utf-8')).get('remaining') == 0
    )
    summary['status'] = development_run_status(
        epochs=epochs,
        parse_complete=parse_complete,
        development_smoke_fold=development_smoke_fold,
    )
    summary['evaluation_scope'] = (
        'development_only'
        if development_smoke_fold is not None
        else 'locked_test'
    )
    summary['operational_claim_allowed'] = bool(
        epochs >= 12 and parse_complete and development_smoke_fold is None
    )
    summary['protocol'] = protocol
    _atomic_json(output / 'summary.json', summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-fold", type=int, default=0, choices=range(5))
    parser.add_argument("--held-out-family")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-train-windows", type=int, default=50_000)
    parser.add_argument("--max-val-windows", type=int, default=20_000)
    parser.add_argument(
        "--development-smoke-fold", type=int, choices=range(5),
        help=(
            "Use this disjoint development fold for evaluation; locked test stays "
            "unread. Runs below 12 epochs remain smoke_only."
        ),
    )
    parser.add_argument(
        '--protocol', choices=('full', 'simulation_to_real', 'real_only'),
        default='full',
    )
    args = parser.parse_args()
    output = run(
        protocol=args.protocol,
        validation_fold=args.validation_fold,
        held_out_family=args.held_out_family,
        epochs=args.epochs,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        development_smoke_fold=args.development_smoke_fold,
    )
    print(output)


if __name__ == "__main__":
    main()
