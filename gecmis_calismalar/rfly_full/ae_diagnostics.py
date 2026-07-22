"""Post-hoc diagnostics for the frozen 1 Hz Dense-AE baseline.

This module never retunes the frozen checkpoint or overwrites its original
reports.  It explains the operating trade-off and metric levels, while treating
Wind as environmental robustness rather than a system-fault true positive.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from rfly_dl.models import reconstruction_scores
from rfly_full.contract import taxonomy
from rfly_full.dl_worker import (
    EVAL_STRIDE, MODEL_PATH, _load_model, _normal_frames, _windows, split_of,
)
from rfly_full.pipeline import ARTIFACT_ROOT, PARSED_ROOT, _atomic_json, _k_of_n

OUTPUT_ROOT = ARTIFACT_ROOT / "v2" / "dense_ae_diagnostics"


def _validation_scores(model, checkpoint: dict) -> np.ndarray:
    normal = _normal_frames()
    validation = normal[normal.case_id.map(lambda value: split_of(str(value)) == "val")]
    x, mask, _ = _windows(validation, checkpoint["columns"], checkpoint["scaler"], EVAL_STRIDE)
    return reconstruction_scores("dense_ae", model, x, mask)


def _score_pool(model, checkpoint: dict) -> pd.DataFrame:
    rows = []
    for path in sorted(PARSED_ROOT.glob("*/*.parquet")):
        if path.parent.name == "bootstrap_normal":
            continue
        frame = pd.read_parquet(path)
        if frame.label.eq("normal").all():
            frame = frame[frame.case_id.map(lambda value: split_of(str(value)) == "test")]
        if frame.empty:
            continue
        x, mask, meta = _windows(frame, checkpoint["columns"], checkpoint["scaler"], EVAL_STRIDE)
        if not len(meta):
            continue
        meta["score"] = reconstruction_scores("dense_ae", model, x, mask)
        lookup = frame[["case_id", "package"]].drop_duplicates("case_id").set_index("case_id")["package"]
        meta["package"] = meta["case_id"].map(lookup)
        rows.append(meta)
    return pd.concat(rows, ignore_index=True)


def _decorate(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    fields = pool.apply(
        lambda row: taxonomy(str(row["package"]), str(row["case_id"])), axis=1,
        result_type="expand",
    )
    for column in fields:
        pool[column] = fields[column]
    pool["domain"] = pool["package"].str.extract(r"^(SIL|HIL|Real)", expand=False).fillna("Unknown")
    phase = pd.Series("normal", index=pool.index, dtype=object)
    for case, group in pool.groupby("case_id", sort=False):
        index = group.index
        role = str(group["evaluation_role"].iloc[0])
        active = group["fault_active"].to_numpy(bool)
        if role == "environment_robustness":
            phase.loc[index] = np.where(active, "environment_active", "environment_inactive")
        elif role == "fault_detection":
            if active.any():
                times = group["t_end"].to_numpy(float)
                active_times = times[active]
                values = np.where(
                    active, "fault_active",
                    np.where(times < active_times.min(), "pre_fault", "post_fault"),
                )
                phase.loc[index] = values
            else:
                phase.loc[index] = "missing_truth"
    pool["phase"] = phase
    return pool


def _metrics(pool: pd.DataFrame, threshold: float) -> tuple[dict, pd.DataFrame]:
    events = detected = 0
    nofault_false = nofault_seconds = 0
    contextual_false = contextual_seconds = 0
    environment_alarms = environment_seconds = 0
    tp = fn = fp = tn = 0
    flights = []
    for case, group in pool.groupby("case_id", sort=False):
        group = group.sort_values("t_end")
        alarm = _k_of_n(group["score"].to_numpy() > threshold)
        truth = group["fault_active"].to_numpy(bool)
        role = str(group["evaluation_role"].iloc[0])
        hit = False
        if role == "fault_detection":
            events += 1
            hit = bool((alarm & truth).any())
            detected += int(hit)
            tp += int(hit)
            fn += int(not hit)
            contextual_false += int((alarm & ~truth).sum())
            contextual_seconds += int((~truth).sum())
        elif role == "normal_reference":
            false_count = int(alarm.sum())
            nofault_false += false_count
            nofault_seconds += len(group)
            contextual_false += false_count
            contextual_seconds += len(group)
            fp += int(alarm.any())
            tn += int(not alarm.any())
        else:
            environment_alarms += int(alarm.sum())
            environment_seconds += len(group)
        flights.append({
            "case_id": case, "domain": group["domain"].iloc[0],
            "fault_family": group["fault_family"].iloc[0],
            "fault_subtype": group["fault_subtype"].iloc[0],
            "evaluation_role": role, "detected": hit,
            "alarm_events": int(alarm.sum()),
        })
    result = {
        "threshold": float(threshold), "events": events, "detected": detected,
        "event_recall": detected / events if events else None,
        "nofault_false_alarm_events": nofault_false,
        "nofault_exposure_hours": nofault_seconds / 3600,
        "nofault_fa_per_hour": nofault_false / (nofault_seconds / 3600) if nofault_seconds else None,
        "all_nonfault_false_alarm_events": contextual_false,
        "all_nonfault_exposure_hours": contextual_seconds / 3600,
        "all_nonfault_fa_per_hour": contextual_false / (contextual_seconds / 3600) if contextual_seconds else None,
        "environment_alarm_events": environment_alarms,
        "environment_exposure_hours": environment_seconds / 3600,
        "environment_alarms_per_hour": environment_alarms / (environment_seconds / 3600) if environment_seconds else None,
        "flight_tp": tp, "flight_fn": fn, "flight_fp": fp, "flight_tn": tn,
    }
    return result, pd.DataFrame(flights)


def _choose(curve: pd.DataFrame, budget: float) -> pd.Series:
    feasible = curve[curve["all_nonfault_fa_per_hour"] <= budget]
    if not len(feasible):
        chosen = curve.sort_values(
            ['all_nonfault_fa_per_hour', 'event_recall'], ascending=[True, False]
        ).iloc[0].copy()
        chosen['budget_target_fa_per_hour'] = budget
        chosen['budget_feasible'] = False
        return chosen
    candidates = feasible
    chosen = candidates.sort_values(
        ["event_recall", "all_nonfault_fa_per_hour"], ascending=[False, True]
    ).iloc[0]
    chosen = chosen.copy()
    chosen['budget_target_fa_per_hour'] = budget
    chosen['budget_feasible'] = bool(len(feasible))
    return chosen


def _family_rows(pool: pd.DataFrame, policies: dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, threshold in policies.items():
        _, flights = _metrics(pool, threshold)
        faults = flights[flights["evaluation_role"].eq("fault_detection")]
        for keys, group in faults.groupby(["domain", "fault_family", "fault_subtype"], dropna=False):
            domain, family, subtype = keys
            rows.append({
                "policy": name, "domain": domain, "fault_family": family,
                "fault_subtype": subtype, "flights": int(len(group)),
                "detected": int(group["detected"].sum()),
                "recall": float(group["detected"].mean()),
            })
    return pd.DataFrame(rows)


def _plot_curve(curve: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(curve["all_nonfault_fa_per_hour"], curve["event_recall"], marker=".")
    axis.axvline(2, color="#dc2626", linestyle="--", label="Critical 2 FA/h")
    axis.axvline(12, color="#f59e0b", linestyle="--", label="Advisory 12 FA/h")
    axis.set(xlabel="False alarm events / hour", ylabel="System-fault event recall",
             title="Frozen Dense AE: recall–false-alarm diagnostic")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_family(family: pd.DataFrame, path: Path) -> None:
    aggregate = family.groupby(["policy", "domain", "fault_family"], as_index=False).agg(
        flights=("flights", "sum"), detected=("detected", "sum")
    )
    aggregate["recall"] = aggregate["detected"] / aggregate["flights"]
    labels = aggregate[["domain", "fault_family"]].drop_duplicates().agg(" / ".join, axis=1)
    keys = aggregate[["domain", "fault_family"]].drop_duplicates().apply(tuple, axis=1).tolist()
    policies = list(aggregate["policy"].drop_duplicates())
    x = np.arange(len(keys))
    width = 0.8 / max(1, len(policies))
    figure, axis = plt.subplots(figsize=(max(9, len(keys) * 0.65), 5))
    for index, policy in enumerate(policies):
        lookup = aggregate[aggregate["policy"].eq(policy)].set_index(["domain", "fault_family"])["recall"]
        axis.bar(x + index * width, [lookup.get(key, 0) for key in keys], width, label=policy)
    axis.set_xticks(x + width * (len(policies) - 1) / 2, labels, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Event recall")
    axis.set_title("Dense AE recall by domain and fault family")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run() -> Path:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model, checkpoint = _load_model()
    validation = _validation_scores(model, checkpoint)
    score_cache = OUTPUT_ROOT / 'scored_windows.parquet'
    if score_cache.exists():
        pool = pd.read_parquet(score_cache)
    else:
        pool = _decorate(_score_pool(model, checkpoint))
        pool.to_parquet(score_cache, index=False)
    quantiles = np.unique(np.r_[
        np.linspace(0.80, 0.995, 18), 0.997, 0.999, 0.9995,
        0.9999, 0.99995, 0.99999, 1.0,
    ])
    thresholds = np.quantile(validation, quantiles)
    rows = []
    for quantile, threshold in zip(quantiles, thresholds):
        metrics, _ = _metrics(pool, float(threshold))
        rows.append({"validation_normal_quantile": float(quantile), **metrics})
    current_metrics, current_flights = _metrics(pool, float(checkpoint["threshold"]))
    curve = pd.DataFrame(rows).sort_values("threshold")
    critical = _choose(curve, 2.0)
    advisory = _choose(curve, 12.0)
    policies = {
        "frozen_q99.5": float(checkpoint["threshold"]),
        "diagnostic_at_2_fa_h": float(critical["threshold"]),
        "diagnostic_at_12_fa_h": float(advisory["threshold"]),
    }
    family = _family_rows(pool, policies)
    distributions = (
        pool.groupby(["domain", "fault_family", "phase"])["score"]
        .agg(count="size", mean="mean", median="median", q90=lambda value: value.quantile(0.90),
             q95=lambda value: value.quantile(0.95), q99=lambda value: value.quantile(0.99))
        .reset_index()
    )
    curve.to_csv(OUTPUT_ROOT / "threshold_recall_fa_curve.csv", index=False)
    family.to_csv(OUTPUT_ROOT / "domain_family_recall.csv", index=False)
    distributions.to_csv(OUTPUT_ROOT / "score_distributions_by_phase.csv", index=False)
    current_flights.to_csv(OUTPUT_ROOT / "frozen_policy_per_flight.csv", index=False)
    _plot_curve(curve, OUTPUT_ROOT / "threshold_recall_fa_curve.png")
    _plot_family(family, OUTPUT_ROOT / "domain_family_recall.png")
    rank_mask = pool.evaluation_role.ne('environment_robustness') & pool.phase.ne('missing_truth')
    rank_truth = pool.loc[rank_mask, 'phase'].eq('fault_active').to_numpy()
    rank_scores = pool.loc[rank_mask, 'score'].to_numpy(float)
    ranking = {
        'windows': int(len(rank_scores)),
        'positive_windows': int(rank_truth.sum()),
        'auroc': float(roc_auc_score(rank_truth, rank_scores)),
        'auprc': float(average_precision_score(rank_truth, rank_scores)),
        'positive_prevalence': float(rank_truth.mean()),
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "posthoc_diagnostic_not_model_selection",
        "wind_contract": "environment robustness, excluded from system-fault recall",
        "frozen_policy": current_metrics,
        "recall_at_2_fa_h": critical.to_dict(),
        "recall_at_12_fa_h": advisory.to_dict(),
        "metric_level_explanation": {
            "flight_fp_tn": "whole NoFault flight contains any alarm onset",
            "nofault_fa_per_hour": "alarm onsets on held-out NoFault flights only",
            "all_nonfault_fa_per_hour": "NoFault plus pre/post-fault normal exposure",
            "environment_alarms_per_hour": "Wind robustness stream, reported separately",
        },
    }
    _atomic_json(OUTPUT_ROOT / "summary.json", summary)
    summary['window_ranking_diagnostic'] = ranking
    _atomic_json(OUTPUT_ROOT / 'summary.json', summary)
    return OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(run())


if __name__ == "__main__":
    main()
