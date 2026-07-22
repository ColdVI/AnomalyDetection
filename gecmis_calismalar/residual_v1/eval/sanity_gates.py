"""Threshold-independent RESIDUAL-V1 sanity gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

GateStatus = Literal["passed", "flagged", "failed", "not_evaluable"]


@dataclass(frozen=True)
class GateResult:
    gate: str
    dataset: str
    channel: str
    status: GateStatus
    reason: str | None
    metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)


class GateError(RuntimeError):
    """Raised when a downstream stage is attempted without a passed gate."""


def s1_magnitude_gate(
    frame: pd.DataFrame,
    *,
    dataset: str,
    channel: str,
    rho_threshold: float = 0.5,
) -> GateResult:
    """Compare per-flight mean |z| with the same rows' mean input magnitude."""

    required = {"flight_id", "z", "input_magnitude"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"{channel}: missing S-1 columns {missing}")
    if not 0.0 < rho_threshold < 1.0:
        raise ValueError("rho_threshold must be between zero and one")
    values = frame.loc[:, ["flight_id", "z", "input_magnitude"]].copy()
    values["abs_z"] = pd.to_numeric(values["z"], errors="coerce").abs()
    values["input_magnitude"] = pd.to_numeric(values["input_magnitude"], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna(subset=["abs_z", "input_magnitude"])
    flight = values.groupby("flight_id", sort=True).agg(
        mean_abs_z=("abs_z", "mean"),
        mean_input_magnitude=("input_magnitude", "mean"),
        rows=("abs_z", "size"),
    )
    if len(flight) < 3 or flight["mean_abs_z"].nunique() < 2 or flight["mean_input_magnitude"].nunique() < 2:
        return GateResult(
            gate="S-1",
            dataset=dataset,
            channel=channel,
            status="not_evaluable",
            reason="insufficient_nonconstant_flight_coverage",
            metrics={"flight_count": int(len(flight)), "rho_threshold": rho_threshold},
        )
    correlation = spearmanr(flight["mean_abs_z"], flight["mean_input_magnitude"])
    rho = float(correlation.statistic)
    pvalue = float(correlation.pvalue)
    flagged = bool(rho >= rho_threshold)
    return GateResult(
        gate="S-1",
        dataset=dataset,
        channel=channel,
        status="flagged" if flagged else "passed",
        reason="rho_at_or_above_threshold" if flagged else None,
        metrics={
            "spearman_rho": rho,
            "spearman_pvalue": pvalue,
            "rho_threshold": rho_threshold,
            "flight_count": int(len(flight)),
            "finite_row_count": int(len(values)),
            "aggregation": "per_flight_mean_abs_z_vs_mean_input_magnitude",
        },
    )


def s3_event_separation_gate(
    frame: pd.DataFrame,
    *,
    dataset: str,
    channel: str,
    fault_class: str,
    events: list[dict],
    pvalue_threshold: float = 0.01,
) -> GateResult:
    """Pool frozen pre/post |z| windows and run the threshold-free KS gate."""

    required = {"flight_id", "t", "z"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"{channel}: missing S-3 columns {missing}")
    if not 0.0 < pvalue_threshold < 1.0:
        raise ValueError("pvalue_threshold must be between zero and one")
    data = frame.loc[:, ["flight_id", "t", "z"]].copy()
    data["flight_id"] = data["flight_id"].astype(str)
    data["t"] = pd.to_numeric(data["t"], errors="coerce")
    data["abs_z"] = pd.to_numeric(data["z"], errors="coerce").abs()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["t", "abs_z"])
    pre_parts = []
    post_parts = []
    event_metrics = []
    for event in events:
        flight_id = str(event["flight_id"])
        onset = float(event["onset_s"])
        flight = data.loc[data["flight_id"] == flight_id]
        pre = flight.loc[(flight["t"] >= onset - 60.0) & (flight["t"] <= onset - 10.0), "abs_z"]
        post = flight.loc[(flight["t"] >= onset) & (flight["t"] <= onset + 15.0), "abs_z"]
        if not pre.empty:
            pre_parts.append(pre.to_numpy(float))
        if not post.empty:
            post_parts.append(post.to_numpy(float))
        event_metrics.append(
            {
                "flight_id": flight_id,
                "onset_s": onset,
                "pre_rows": int(len(pre)),
                "post_rows": int(len(post)),
                "pre_median_abs_z": float(pre.median()) if not pre.empty else None,
                "post_median_abs_z": float(post.median()) if not post.empty else None,
                "median_shift": (
                    float(post.median() - pre.median())
                    if not pre.empty and not post.empty
                    else None
                ),
            }
        )
    if not pre_parts or not post_parts:
        return GateResult(
            gate="S-3",
            dataset=dataset,
            channel=channel,
            status="not_evaluable",
            reason="empty_frozen_pre_or_post_window",
            metrics={
                "fault_class": fault_class,
                "event_count": len(events),
                "event_metrics": event_metrics,
                "pvalue_threshold": pvalue_threshold,
            },
        )
    pre_values = np.concatenate(pre_parts)
    post_values = np.concatenate(post_parts)
    ks = ks_2samp(pre_values, post_values, alternative="two-sided", method="auto")
    passed = bool(float(ks.pvalue) < pvalue_threshold)
    return GateResult(
        gate="S-3",
        dataset=dataset,
        channel=channel,
        status="passed" if passed else "failed",
        reason=None if passed else "ks_pvalue_not_below_threshold",
        metrics={
            "fault_class": fault_class,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "pvalue_threshold": pvalue_threshold,
            "alternative": "two-sided",
            "pre_window_s": [-60.0, -10.0],
            "post_window_s": [0.0, 15.0],
            "pre_rows": int(len(pre_values)),
            "post_rows": int(len(post_values)),
            "pre_median_abs_z": float(np.median(pre_values)),
            "post_median_abs_z": float(np.median(post_values)),
            "event_count": len(events),
            "event_metrics": event_metrics,
        },
    )


def require_s3_pass(class_results: dict[str, dict], required_classes: list[str]) -> None:
    """Programmatic calibration lock: every requested class must explicitly pass."""

    statuses = {}
    for fault_class in required_classes:
        value = class_results.get(fault_class)
        statuses[fault_class] = value if isinstance(value, str) else (value or {}).get("status", "missing")
    failures = {fault_class: status for fault_class, status in statuses.items() if status != "passed"}
    if failures:
        raise GateError(f"S-3 PASS required before calibration: {failures}")
