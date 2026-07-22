"""Domain-calibrated normal-only temporal autoencoder for RflyMAD-Full v2.

Only development NoFault flights enter scaler fitting, model training and
threshold calibration.  Development fault and Wind flights are consumed after
the policy is frozen for diagnostic evaluation.  Locked-test telemetry is not
loaded by this module.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from rfly_full.contract import V2_ROOT
from rfly_full.pipeline import _atomic_json
from rfly_full.supervised import (
    _available_manifest, _fit_scaler, _load_flight, _scaled_values, alarm_onsets,
)
from rfly_full.v2_parser import SAMPLE_HZ, V2_FEATURES


OUTPUT_ROOT = V2_ROOT / "normal_temporal_ae"
WINDOW_SECONDS = 10
TRAIN_STRIDE_SECONDS = 5
VALIDATION_STRIDE_SECONDS = 2
EVAL_STRIDE_SECONDS = 1
SEED = 20260721
DOMAINS = ("Real", "HIL", "SIL")
BUDGETS = {"critical": 2.0, "advisory": 12.0}


class TemporalConvAutoencoder(nn.Module):
    """Time-compressing convolutional AE; missingness is an input channel."""

    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels_in, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 16, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(64, channels_out, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values.transpose(1, 2))
        return self.decoder(encoded).transpose(1, 2)


def _ranked_groups(domain: str, groups: list[str]) -> list[str]:
    return sorted(
        groups,
        key=lambda value: hashlib.sha256(
            f"{SEED}:{domain}:{value}".encode("utf-8")
        ).hexdigest(),
    )


def _normal_split(
    available: pd.DataFrame, validation_rotation: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    normal = available.loc[
        available["split"].eq("development")
        & available["evaluation_role"].eq("normal_reference")
    ].copy()
    validation_groups: dict[str, str] = {}
    for domain in DOMAINS:
        groups = sorted(normal.loc[normal["domain"].eq(domain), "split_group_id"].dropna().astype(str).unique())
        if len(groups) < 2:
            raise RuntimeError(
                f"normal-only training needs at least two parsed {domain} groups; found {len(groups)}"
            )
        ranked = _ranked_groups(domain, groups)
        validation_groups[domain] = ranked[validation_rotation % len(ranked)]
    validation_mask = np.zeros(len(normal), dtype=bool)
    for domain, group in validation_groups.items():
        validation_mask |= normal["domain"].eq(domain) & normal["split_group_id"].eq(group)
    train = normal.loc[~validation_mask].copy()
    validation = normal.loc[validation_mask].copy()
    return train, validation, validation_groups


def _flight_windows(frame: pd.DataFrame, scaler: dict, stride_seconds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _scaled_values(frame, scaler)
    feature_count = len(scaler["features"])
    target = values[:, :feature_count]
    observed = values[:, feature_count:]
    length = WINDOW_SECONDS * SAMPLE_HZ
    stride = stride_seconds * SAMPLE_HZ
    indices = range(length - 1, len(frame), stride)
    x, y, mask = [], [], []
    for end in indices:
        start = end - length + 1
        x.append(values[start : end + 1])
        y.append(target[start : end + 1])
        mask.append(observed[start : end + 1])
    if not x:
        return (
            np.empty((0, length, feature_count * 2), np.float32),
            np.empty((0, length, feature_count), np.float32),
            np.empty((0, length, feature_count), np.float32),
        )
    return np.stack(x).astype(np.float32), np.stack(y).astype(np.float32), np.stack(mask).astype(np.float32)


def _normal_windows(manifest: pd.DataFrame, scaler: dict, stride_seconds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values, targets, masks, domains = [], [], [], []
    columns = ["t_rel_s", *scaler["features"]]
    for row in manifest.itertuples(index=False):
        frame = _load_flight(str(row.canonical_case_id), str(row.domain), columns)
        x, target, mask = _flight_windows(frame, scaler, stride_seconds)
        if len(x):
            x_values.append(x)
            targets.append(target)
            masks.append(mask)
            domains.extend([str(row.domain)] * len(x))
    if not x_values:
        raise RuntimeError("normal split produced no temporal windows")
    return (
        np.concatenate(x_values), np.concatenate(targets), np.concatenate(masks),
        np.asarray(domains, dtype=object),
    )


def _masked_loss(target: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = (target - reconstruction).square() * mask
    return error.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1.0)


def _validation_loss(model: nn.Module, x: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 512):
            xb = torch.from_numpy(x[start : start + 512])
            yb = torch.from_numpy(target[start : start + 512])
            mb = torch.from_numpy(mask[start : start + 512])
            values.append(_masked_loss(yb, model(xb), mb).numpy())
    return float(np.concatenate(values).mean())


def _train(
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *, epochs: int,
) -> tuple[nn.Module, list[dict]]:
    x_train, y_train, m_train, domains = train
    x_val, y_val, m_val, _ = validation
    counts = pd.Series(domains).value_counts().to_dict()
    weights = np.asarray([1.0 / counts[value] for value in domains], dtype=np.float64)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(m_train)
    )
    loader = DataLoader(dataset, batch_size=128, sampler=sampler)
    torch.manual_seed(SEED)
    model = TemporalConvAutoencoder(x_train.shape[-1], y_train.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb, mb in loader:
            loss = _masked_loss(yb, model(xb), mb).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_loss = _validation_loss(model, x_val, y_val, m_val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= 5:
            break
    if best_state is None:
        raise RuntimeError("normal temporal AE produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history


def _score_flight(model: nn.Module, frame: pd.DataFrame, scaler: dict) -> tuple[np.ndarray, np.ndarray]:
    x, target, mask = _flight_windows(frame, scaler, EVAL_STRIDE_SECONDS)
    if not len(x):
        return np.empty(0), np.empty(0)
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 512):
            xb = torch.from_numpy(x[start : start + 512])
            yb = torch.from_numpy(target[start : start + 512])
            mb = torch.from_numpy(mask[start : start + 512])
            scores.append(_masked_loss(yb, model(xb), mb).numpy())
    length = WINDOW_SECONDS * SAMPLE_HZ
    ends = np.arange(length - 1, len(frame), EVAL_STRIDE_SECONDS * SAMPLE_HZ)
    return np.concatenate(scores), ends


def _score_manifest(model: nn.Module, manifest: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    rows = []
    columns = ["t_rel_s", "fault_active", *scaler["features"]]
    for row in manifest.itertuples(index=False):
        frame = _load_flight(str(row.canonical_case_id), str(row.domain), columns)
        scores, ends = _score_flight(model, frame, scaler)
        for score, end in zip(scores, ends):
            rows.append({
                "canonical_case_id": str(row.canonical_case_id),
                "domain": str(row.domain),
                "fault_family": str(row.fault_family),
                "evaluation_role": str(row.evaluation_role),
                "t_end_s": float(frame["t_rel_s"].iloc[end]),
                "fault_active": bool(frame["fault_active"].iloc[end]),
                "score": float(score),
            })
    return pd.DataFrame(rows)


def _fa_rate(frame: pd.DataFrame, threshold: float) -> tuple[int, float]:
    alarms = 0
    exposure_seconds = 0
    for _, flight in frame.groupby("canonical_case_id", sort=False):
        alarms += int(alarm_onsets(flight["score"].to_numpy(), threshold).sum())
        exposure_seconds += len(flight)
    hours = exposure_seconds / 3600
    return alarms, alarms / hours if hours else float("inf")


def _fit_policies(validation_scores: pd.DataFrame) -> dict:
    policies = {}
    for domain in DOMAINS:
        subset = validation_scores.loc[validation_scores["domain"].eq(domain)]
        if subset.empty:
            raise RuntimeError(f"no normal validation scores for {domain}")
        candidates = np.unique(np.quantile(subset["score"], np.linspace(0.80, 1.0, 121)))
        candidates = np.r_[candidates, np.nextafter(candidates[-1], np.inf)]
        for name, budget in BUDGETS.items():
            rows = []
            for threshold in candidates:
                events, rate = _fa_rate(subset, float(threshold))
                rows.append((float(threshold), events, rate))
            feasible = [row for row in rows if row[2] <= budget]
            chosen = min(feasible, key=lambda row: row[0]) if feasible else max(rows, key=lambda row: row[0])
            policies[f"{domain}:{name}"] = {
                "domain": domain, "policy": name, "budget_fa_per_hour": budget,
                "threshold": chosen[0], "validation_alarm_events": chosen[1],
                "validation_fa_per_hour": chosen[2],
                "alarm": "4-of-6 seconds; 30-second refractory",
            }
    return policies


def _evaluate(scored: pd.DataFrame, policies: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    per_flight_rows = []
    for policy_name in BUDGETS:
        for canonical, flight in scored.groupby("canonical_case_id", sort=False):
            first = flight.iloc[0]
            policy = policies[f"{first['domain']}:{policy_name}"]
            onsets = alarm_onsets(flight["score"].to_numpy(), policy["threshold"])
            active = flight["fault_active"].to_numpy(bool)
            detected = bool(np.any(onsets & active)) if first["evaluation_role"] == "fault_detection" else None
            false_events = int(np.sum(onsets & ~active))
            per_flight_rows.append({
                "policy": policy_name, "canonical_case_id": canonical,
                "domain": first["domain"], "fault_family": first["fault_family"],
                "evaluation_role": first["evaluation_role"], "detected": detected,
                "alarm_events": int(onsets.sum()), "false_alarm_events": false_events,
                "normal_calibration_holdout": bool(first["normal_calibration_holdout"]),
                "exposure_hours": len(flight) / 3600,
                "normal_exposure_hours": float((~active).sum() / 3600),
            })
        result = pd.DataFrame([row for row in per_flight_rows if row["policy"] == policy_name])
        fault = result.loc[result["evaluation_role"].eq("fault_detection")]
        normal = result.loc[
            result["evaluation_role"].eq("normal_reference")
            & result["normal_calibration_holdout"]
        ]
        detected_flags = fault["detected"].fillna(False).astype(bool)
        nonfault = result.loc[
            result["evaluation_role"].eq("fault_detection")
            | (
                result["evaluation_role"].eq("normal_reference")
                & result["normal_calibration_holdout"]
            )
        ]
        environment = result.loc[result["evaluation_role"].eq("environment_robustness")]
        metric_rows.append({
            "policy": policy_name,
            "fault_flights": len(fault), "detected_fault_flights": int(detected_flags.sum()),
            "event_recall": float(detected_flags.mean()) if len(fault) else None,
            "flight_tp": int(detected_flags.sum()),
            "flight_fn": int((~detected_flags).sum()),
            "flight_fp": int(normal["alarm_events"].gt(0).sum()),
            "flight_tn": int(normal["alarm_events"].eq(0).sum()),
            "normal_validation_fa_per_hour": float(normal["false_alarm_events"].sum() / normal["normal_exposure_hours"].sum()) if normal["normal_exposure_hours"].sum() else None,
            "all_nonfault_fa_per_hour": float(nonfault["false_alarm_events"].sum() / nonfault["normal_exposure_hours"].sum()) if nonfault["normal_exposure_hours"].sum() else None,
            "environment_fa_per_hour": float(environment["false_alarm_events"].sum() / environment["normal_exposure_hours"].sum()) if environment["normal_exposure_hours"].sum() else None,
        })
    return pd.DataFrame(metric_rows), pd.DataFrame(per_flight_rows)


def run(
    *, epochs: int = 25, torch_threads: int = 4,
    validation_rotation: int = 0,
) -> Path:
    torch.set_num_threads(max(1, torch_threads))
    available = _available_manifest()
    train_manifest, validation_manifest, validation_groups = _normal_split(
        available, validation_rotation=validation_rotation
    )
    scaler = _fit_scaler(train_manifest, list(V2_FEATURES))
    train_windows = _normal_windows(train_manifest, scaler, TRAIN_STRIDE_SECONDS)
    validation_windows = _normal_windows(validation_manifest, scaler, VALIDATION_STRIDE_SECONDS)
    model, history = _train(train_windows, validation_windows, epochs=epochs)

    development = available.loc[available["split"].eq("development")].copy()
    scored = _score_manifest(model, development, scaler)
    validation_ids = set(validation_manifest["canonical_case_id"].astype(str))
    scored["normal_calibration_holdout"] = scored["canonical_case_id"].isin(validation_ids)
    validation_scores = scored.loc[scored["canonical_case_id"].isin(validation_ids)]
    policies = _fit_policies(validation_scores)
    metrics, per_flight = _evaluate(scored, policies)

    output = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "state_dict": model.state_dict(), "features": scaler["features"],
        "window_seconds": WINDOW_SECONDS, "sample_hz": SAMPLE_HZ,
        "model": "temporal_conv_autoencoder",
    }, output / "model.pt")
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    metrics.to_csv(output / "operational_metrics.csv", index=False)
    per_flight.to_csv(output / "per_flight_metrics.csv", index=False)
    family_metrics = (
        per_flight.loc[per_flight["evaluation_role"].eq("fault_detection")]
        .groupby(["policy", "domain", "fault_family"], observed=True)
        .agg(flights=("canonical_case_id", "nunique"), detected=("detected", "sum"))
        .reset_index()
    )
    family_metrics["recall"] = family_metrics["detected"] / family_metrics["flights"]
    family_metrics.to_csv(output / "domain_family_metrics.csv", index=False)
    scored.to_parquet(output / "development_scores.parquet", index=False)
    _atomic_json(output / "scaler.json", scaler)
    _atomic_json(output / "policies.json", policies)
    summary = {
        "status": "development_only_partial_parse",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "normal_only_temporal_conv_autoencoder",
        "training_contract": "development NoFault only; domain-balanced sampler",
        "threshold_contract": "domain-specific, held-out development NoFault groups only",
        "locked_test_features_read": False,
        "operational_claim_allowed": False,
        "validation_groups": validation_groups,
        "validation_rotation": validation_rotation,
        "train_normal_flights": int(train_manifest["canonical_case_id"].nunique()),
        "validation_normal_flights": int(validation_manifest["canonical_case_id"].nunique()),
        "train_windows": int(len(train_windows[0])),
        "validation_windows": int(len(validation_windows[0])),
        "available_development_flights": int(development["canonical_case_id"].nunique()),
        "epochs_completed": len(history),
        "metrics": metrics.to_dict(orient="records"),
    }
    _atomic_json(output / "summary.json", summary)
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--validation-rotation", type=int, default=0)
    args = parser.parse_args()
    print(run(
        epochs=args.epochs, torch_threads=args.torch_threads,
        validation_rotation=args.validation_rotation,
    ))


if __name__ == "__main__":
    main()
