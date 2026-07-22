"""Aggregate checkpointed RflyMAD batch evidence without mixing metric levels."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gecmis_calismalar.rfly_full.pipeline import ARTIFACT_ROOT, PARSED_ROOT, REPORT_ROOT, RUN_STATE


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def build_summary() -> dict:
    state = json.loads(RUN_STATE.read_text(encoding="utf-8"))
    expansion_state_path = ARTIFACT_ROOT / "expansion_state.json"
    expansion_state = (
        json.loads(expansion_state_path.read_text(encoding="utf-8"))
        if expansion_state_path.exists()
        else {}
    )
    package_rows = []
    total_flights = 0
    total_rows = 0
    for package_dir in sorted(path for path in PARSED_ROOT.iterdir() if path.is_dir() and path.name != "bootstrap_normal"):
        frames = [pd.read_parquet(path, columns=["case_id", "label", "fault_active", "truth_source"]) for path in sorted(package_dir.glob("*.parquet"))]
        if not frames:
            continue
        telemetry = pd.concat(frames, ignore_index=True)
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((REPORT_ROOT / package_dir.name).glob("*.json"))]
        flights = telemetry.drop_duplicates("case_id")
        confusion = {
            key: int(sum(report.get("confusion_flight", {}).get(key, 0) for report in reports))
            for key in ("tp", "fn", "fp", "tn")
        }
        false_events = int(sum(sum(row.get("false_alarm_events", 0) for row in report.get("flights", [])) for report in reports))
        normal_hours = float((~telemetry["fault_active"].astype(bool)).sum() / 3600.0)
        fault_flights = flights[flights["label"].ne("normal")]
        truth_missing = int(
            (~fault_flights["truth_source"].isin({"rfly_ctrl_lxl", "test_info"})).sum()
        )
        events = confusion["tp"] + confusion["fn"]
        row = {
            "package": package_dir.name,
            "flights": int(flights.case_id.nunique()),
            "telemetry_rows_1hz": int(len(telemetry)),
            "truth_missing_flights": truth_missing,
            "tp": confusion["tp"], "fn": confusion["fn"],
            "fp": confusion["fp"], "tn": confusion["tn"],
            "event_recall": confusion["tp"] / events if events else None,
            "false_alarm_events": false_events,
            "normal_exposure_hours": normal_hours,
            "false_alarms_per_hour": false_events / normal_hours if normal_hours else None,
            "batch_row_auroc_mean": _finite_mean([report.get("row_auroc") for report in reports]),
            "batch_row_auprc_mean": _finite_mean([report.get("row_auprc") for report in reports]),
            "batches": len(reports),
        }
        package_rows.append(row)
        total_flights += row["flights"]
        total_rows += row["telemetry_rows_1hz"]
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_stop_reason": state.get("stop_reason"),
        "expansion_stop_reason": expansion_state.get("stop_reason"),
        "expansion_failed_batches": expansion_state.get("failed_batches", {}),
        "failed_cases": state.get("failed_cases", {}),
        "total_flights": total_flights,
        "total_telemetry_rows_1hz": total_rows,
        "model": "Isolation Forest (80 trees), normal-only training",
        "decision": "4-of-6 consecutive 1 Hz score exceedances",
        "metric_contract": {
            "event_recall": "flight/event level",
            "confusion": "flight level",
            "false_alarms_per_hour": "alarm onset count / non-fault telemetry exposure",
            "row_auroc_auprc": "1 Hz row ranking; batch means, not operational metrics",
        },
        "limitations": [
            "Kaggle mirror contains only HIL/SIL Wind simulation packages.",
            "No simulation-domain normal package exists in the mirror; Wind results conflate domain shift and fault response.",
            "Batch smoke results are diagnostic and do not replace a fixed blind holdout evaluation.",
        ],
        "packages": package_rows,
    }


def main() -> None:
    summary = build_summary()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "overnight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(summary["packages"]).to_csv(
        ARTIFACT_ROOT / "overnight_package_metrics.csv", index=False
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
