"""Development-only five-fold sweep for the supervised RflyMAD v2 TCN."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gecmis_calismalar.rfly_full.pipeline import _atomic_json
from gecmis_calismalar.rfly_full.supervised import SUPERVISED_ROOT, run


ROOT = Path(__file__).resolve().parents[2]
SWEEP_NAME = "development_5fold_20260722_v1"
SWEEP_ROOT = SUPERVISED_ROOT / SWEEP_NAME
CONTRACT_PATH = ROOT / "docs" / "RFLYMAD_V2_TCN_DEVELOPMENT_SOZLESMESI_20260722.md"
DEFAULT_CAPS = (12, 25, 50)
EXTENSION_MIN_DELTA = 1e-4


def validation_fold_for_outer(outer_fold: int) -> int:
    """Use the preceding fold for calibration and keep the other three for train."""
    if outer_fold not in range(5):
        raise ValueError("outer fold must be in 0..4")
    return (outer_fold - 1) % 5


def extension_decision(
    history: pd.DataFrame,
    *,
    cap: int,
    caps: tuple[int, ...] = DEFAULT_CAPS,
    min_delta: float = EXTENSION_MIN_DELTA,
) -> dict:
    """Decide a preregistered cap extension from validation loss only."""
    if history.empty or not {"epoch", "validation_loss"}.issubset(history.columns):
        raise ValueError("training history is missing epoch/validation_loss")
    ordered = history.sort_values("epoch")
    best_index = ordered["validation_loss"].astype(float).idxmin()
    best_epoch = int(ordered.loc[best_index, "epoch"])
    best_loss = float(ordered.loc[best_index, "validation_loss"])
    earlier = ordered.loc[ordered["epoch"].astype(int) <= cap - 2, "validation_loss"]
    earlier_best = float(earlier.min()) if len(earlier) else float("inf")
    improvement = earlier_best - best_loss
    has_next_cap = cap in caps and caps.index(cap) < len(caps) - 1
    extend = bool(
        has_next_cap
        and best_epoch >= cap - 1
        and np.isfinite(improvement)
        and improvement >= min_delta
    )
    return {
        "cap": cap,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_before_last_two": earlier_best if np.isfinite(earlier_best) else None,
        "boundary_improvement": improvement if np.isfinite(improvement) else None,
        "extend": extend,
        "next_cap": caps[caps.index(cap) + 1] if extend else None,
        "rule": "best epoch in final two and improvement >= 1e-4",
    }


def _contract_sha256() -> str:
    if not CONTRACT_PATH.exists():
        raise RuntimeError(f"frozen contract is missing: {CONTRACT_PATH}")
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _fa_rate(frame: pd.DataFrame) -> float | None:
    exposure = float(frame["normal_exposure_s"].sum())
    if exposure <= 0:
        return None
    return float(frame["false_alarm_events"].sum()) / (exposure / 3600.0)


def _recall(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    return float(frame["detected"].astype(bool).mean())


def _outer_rows(run_dir: Path, outer_fold: int, validation_fold: int, cap: int) -> list[dict]:
    operational = pd.read_csv(run_dir / "operational_metrics.csv").set_index("policy")
    flights = pd.read_csv(run_dir / "per_flight_metrics.csv")
    rows = []
    for policy in ("critical", "advisory"):
        subset = flights.loc[flights["policy"].eq(policy)].copy()
        real_fault = subset.loc[
            subset["domain"].eq("Real")
            & subset["evaluation_role"].eq("fault_detection")
        ]
        real_motor = real_fault.loc[real_fault["fault_family"].eq("Motor")]
        real_sensor = real_fault.loc[real_fault["fault_family"].eq("Sensor")]
        motor_recall = _recall(real_motor)
        sensor_recall = _recall(real_sensor)
        real_macro = (
            float(np.mean([motor_recall, sensor_recall]))
            if motor_recall is not None and sensor_recall is not None
            else None
        )
        real_normal = subset.loc[
            subset["domain"].eq("Real")
            & subset["evaluation_role"].eq("normal_reference")
        ]
        wind = subset.loc[subset["evaluation_role"].eq("environment_robustness")]
        op = operational.loc[policy]
        rows.append({
            "outer_fold": outer_fold,
            "validation_fold": validation_fold,
            "epoch_cap": cap,
            "policy": policy,
            "event_recall": float(op["event_recall"]),
            "all_nonfault_fa_per_hour": float(op["false_alarms_per_hour"]),
            "median_detection_delay_s": float(op["median_detection_delay_s"]),
            "wind_fa_per_hour": _fa_rate(wind),
            "real_motor_recall": motor_recall,
            "real_sensor_recall": sensor_recall,
            "real_macro_recall": real_macro,
            "real_normal_fa_per_hour": _fa_rate(real_normal),
            "real_normal_alarm_flight_rate": (
                float(real_normal["false_alarm_events"].gt(0).mean())
                if len(real_normal) else None
            ),
            "run_dir": _relative(run_dir),
        })
    return rows


def _aggregate(outer: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "event_recall", "all_nonfault_fa_per_hour", "wind_fa_per_hour",
        "real_motor_recall", "real_sensor_recall", "real_macro_recall",
        "real_normal_fa_per_hour", "real_normal_alarm_flight_rate",
        "median_detection_delay_s",
    ]
    rows = []
    for policy, frame in outer.groupby("policy", sort=False):
        row = {"policy": policy}
        for metric in metrics:
            values = frame[metric].dropna().astype(float)
            row[f"{metric}_n"] = int(len(values))
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else None
            row[f"{metric}_min"] = float(values.min()) if len(values) else None
            row[f"{metric}_max"] = float(values.max()) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _gate_summary(aggregate: pd.DataFrame) -> dict:
    critical = aggregate.loc[aggregate["policy"].eq("critical")].iloc[0]
    advisory = aggregate.loc[aggregate["policy"].eq("advisory")].iloc[0]
    critical_conditions = {
        "recall_mean_ge_0_50": critical["event_recall_mean"] >= 0.50,
        "recall_min_ge_0_40": critical["event_recall_min"] >= 0.40,
        "fa_mean_le_2": critical["all_nonfault_fa_per_hour_mean"] <= 2.0,
        "fa_max_le_4": critical["all_nonfault_fa_per_hour_max"] <= 4.0,
    }
    advisory_conditions = {
        "recall_mean_ge_0_60": advisory["event_recall_mean"] >= 0.60,
        "recall_min_ge_0_50": advisory["event_recall_min"] >= 0.50,
        "fa_mean_le_12": advisory["all_nonfault_fa_per_hour_mean"] <= 12.0,
        "fa_max_le_15": advisory["all_nonfault_fa_per_hour_max"] <= 15.0,
    }
    real_conditions = {
        "macro_mean_ge_0_40": critical["real_macro_recall_mean"] >= 0.40,
        "motor_mean_ge_0_30": critical["real_motor_recall_mean"] >= 0.30,
        "sensor_mean_ge_0_30": critical["real_sensor_recall_mean"] >= 0.30,
        "macro_min_ge_0_25": critical["real_macro_recall_min"] >= 0.25,
        "real_fa_mean_le_4": critical["real_normal_fa_per_hour_mean"] <= 4.0,
        "real_fa_max_le_8": critical["real_normal_fa_per_hour_max"] <= 8.0,
    }
    wind_conditions = {
        "wind_fa_mean_le_15": critical["wind_fa_per_hour_mean"] <= 15.0,
        "wind_fa_max_le_20": critical["wind_fa_per_hour_max"] <= 20.0,
        "critical_recall_mean_ge_0_50": critical["event_recall_mean"] >= 0.50,
        "all_nonfault_fa_mean_le_2": critical["all_nonfault_fa_per_hour_mean"] <= 2.0,
    }
    def gate(conditions: dict) -> dict:
        normalized = {name: bool(value) for name, value in conditions.items()}
        return {"passed": all(normalized.values()), "conditions": normalized}

    return {
        "critical_development_gate": {
            **gate(critical_conditions),
        },
        "advisory_development_gate": {
            **gate(advisory_conditions),
        },
        "real_research_gate": {
            **gate(real_conditions),
        },
        "wind_intermediate_gate": {
            **gate(wind_conditions),
        },
        "operational_claim_allowed": False,
    }


def _write_outputs(state: dict) -> None:
    outer_rows = []
    history_frames = []
    for outer_fold in range(5):
        selected = state["completed_outer_folds"][str(outer_fold)]
        run_dir = ROOT / selected["run_dir"]
        outer_rows.extend(_outer_rows(
            run_dir, outer_fold, selected["validation_fold"], selected["epoch_cap"],
        ))
        history = pd.read_csv(run_dir / "training_history.csv")
        history.insert(0, "outer_fold", outer_fold)
        history.insert(1, "validation_fold", selected["validation_fold"])
        history.insert(2, "epoch_cap", selected["epoch_cap"])
        history_frames.append(history)
    outer = pd.DataFrame(outer_rows)
    aggregate = _aggregate(outer)
    outer.to_csv(SWEEP_ROOT / "outer_fold_metrics.csv", index=False)
    aggregate.to_csv(SWEEP_ROOT / "aggregate_metrics.csv", index=False)
    pd.concat(history_frames, ignore_index=True).to_csv(
        SWEEP_ROOT / "training_history.csv", index=False,
    )
    gates = _gate_summary(aggregate)
    _atomic_json(SWEEP_ROOT / "gate_summary.json", gates)
    _atomic_json(SWEEP_ROOT / "summary.json", {
        "status": "development_only_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": _relative(CONTRACT_PATH),
        "contract_sha256": state["contract_sha256"],
        "outer_folds": list(range(5)),
        "completed_outer_folds": state["completed_outer_folds"],
        "gate_summary": gates,
        "locked_test_features_read": False,
        "operational_claim_allowed": False,
    })


def run_sweep(*, caps: tuple[int, ...] = DEFAULT_CAPS) -> Path:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = SWEEP_ROOT / "sweep_state.json"
    contract_hash = _contract_sha256()
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["contract_sha256"] != contract_hash:
            raise RuntimeError("frozen contract changed after sweep creation")
    else:
        state = {
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": contract_hash,
            "caps": list(caps),
            "completed_outer_folds": {},
            "attempts": [],
            "locked_test_features_read": False,
            "operational_claim_allowed": False,
        }
        _atomic_json(state_path, state)

    for outer_fold in range(5):
        if str(outer_fold) in state["completed_outer_folds"]:
            continue
        validation_fold = validation_fold_for_outer(outer_fold)
        for cap in caps:
            output = run(
                validation_fold=validation_fold,
                development_smoke_fold=outer_fold,
                epochs=cap,
                max_train_windows=50_000,
                max_val_windows=20_000,
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            if summary["locked_test_features_read"]:
                raise RuntimeError("locked test was read")
            if summary["operational_claim_allowed"]:
                raise RuntimeError("development run granted an operational claim")
            if summary["status"] != "development_only":
                raise RuntimeError(f"unexpected development status: {summary['status']}")
            history = pd.read_csv(output / "training_history.csv")
            decision = extension_decision(history, cap=cap, caps=caps)
            attempt = {
                "outer_fold": outer_fold,
                "validation_fold": validation_fold,
                "epoch_cap": cap,
                "run_dir": _relative(output),
                "extension_decision": decision,
            }
            state["attempts"].append(attempt)
            state["active_outer_fold"] = outer_fold
            _atomic_json(state_path, state)
            if decision["extend"]:
                continue
            state["completed_outer_folds"][str(outer_fold)] = attempt
            state["active_outer_fold"] = None
            _atomic_json(state_path, state)
            break
        else:
            raise RuntimeError(f"outer fold {outer_fold} exhausted caps without selection")

    state["status"] = "complete"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(state_path, state)
    _write_outputs(state)
    return SWEEP_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", type=int, nargs="+", default=list(DEFAULT_CAPS))
    args = parser.parse_args()
    caps = tuple(args.caps)
    if not caps or caps[0] != 12 or tuple(sorted(set(caps))) != caps:
        raise ValueError("caps must be unique ascending values starting at 12")
    print(run_sweep(caps=caps))


if __name__ == "__main__":
    main()
