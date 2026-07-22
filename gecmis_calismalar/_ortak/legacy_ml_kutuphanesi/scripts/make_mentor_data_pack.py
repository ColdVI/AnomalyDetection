"""Create a compact, human-readable mentor packet for ML-14/RFLY status."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ML14 = ROOT / "artifacts/ml14/uav_sead"
RUN = ML14 / "full_matrix"
RFLY = ROOT / "artifacts/rfly0/rflymad"
OUT = ML14 / "mentor_pack"
FIG = OUT / "figures"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = metrics.copy()
    rows["budget_limit"] = rows["budget"].map({"critical": 2.0, "advisory": 12.0})
    rows["target_recall"] = rows["budget"].map({"critical": 0.30, "advisory": 0.50})
    rows["passes_fa_budget"] = rows["false_alarms_per_hour"] <= rows["budget_limit"]
    rows["passes_recall_target"] = rows["event_onset_recall"] >= rows["target_recall"]
    summary = (
        rows.groupby(["score_source", "decision", "budget"], as_index=False)
        .agg(
            mean_event_onset_recall=("event_onset_recall", "mean"),
            mean_false_alarms_per_hour=("false_alarms_per_hour", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    summary["budget_limit"] = summary["budget"].map({"critical": 2.0, "advisory": 12.0})
    summary["target_recall"] = summary["budget"].map({"critical": 0.30, "advisory": 0.50})
    summary["passes_gate"] = (
        (summary["mean_false_alarms_per_hour"] <= summary["budget_limit"])
        & (summary["mean_event_onset_recall"] >= summary["target_recall"])
    )
    return summary.sort_values(
        ["passes_gate", "mean_event_onset_recall", "mean_false_alarms_per_hour"],
        ascending=[False, False, True],
    )


def _old_new_category_compare() -> pd.DataFrame:
    old_path = ROOT / "artifacts/ml13/uav_sead/full_matrix/category_metrics.csv"
    new_path = RUN / "category_metrics.csv"
    if not old_path.exists() or not new_path.exists():
        return pd.DataFrame()

    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    key = ["annotation_category"]
    old_best = (
        old[(old["decision"] == "cusum") & (old["budget"] == "advisory")]
        .groupby(key, as_index=False)
        .agg(
            old_recall=("event_onset_recall", "mean"),
            old_fa_per_hour=("false_alarms_per_hour", "mean"),
        )
    )
    new_best = (
        new[(new["score_source"] == "ml14_fusion") & (new["decision"] == "cusum")
            & (new["budget"] == "advisory")]
        .groupby(key, as_index=False)
        .agg(
            new_recall=("event_onset_recall", "mean"),
            new_fa_per_hour=("false_alarms_per_hour", "mean"),
        )
    )
    merged = old_best.merge(new_best, on=key, how="outer")
    merged["recall_delta"] = merged["new_recall"] - merged["old_recall"]
    return merged.sort_values("new_recall", ascending=False)


def _write_rfly_feature_report(rfly_silver: pd.DataFrame) -> dict:
    subset_counts = (
        rfly_silver[["source_id", "rflymad_subdataset"]]
        .drop_duplicates()
        .groupby("rflymad_subdataset")
        .size()
        .sort_index()
        .to_dict()
    )
    label_counts = (
        rfly_silver[["source_id", "label"]]
        .drop_duplicates()
        .groupby("label")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    normal = int(label_counts.get("normal", 0))
    report = {
        "source": "rflymad",
        "status": "parsed_but_gate_not_ready" if normal < 61 else "ready_for_split_build",
        "reason": (
            "registered RFLY split quota requires at least 61 normal flights "
            "for 30 val + 30 test-normal + train"
        ),
        "silver_flights": int(rfly_silver["source_id"].nunique()),
        "silver_rows": int(len(rfly_silver)),
        "case_label_counts": label_counts,
        "case_subset_counts": subset_counts,
        "required_normal_for_registered_quota": 61,
        "current_normal_flights": normal,
        "normal_flights_missing": max(0, 61 - normal),
        "next_action": (
            "build RFLY split/features and run R-gates"
            if normal >= 61 else
            "continue download until Real-No_Fault normal >=61; then build RFLY split/features"
        ),
    }
    RFLY.mkdir(parents=True, exist_ok=True)
    (RFLY / "feature_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    gates = _read_json(RUN / "gates.json")
    rebuild = _read_json(ML14 / "rebuild_report.json")
    manifest = _read_json(RUN / "manifest.json")
    metrics = pd.read_csv(RUN / "metrics.csv")
    best = _best_rows(metrics)
    best.to_csv(OUT / "ml14_best_operational_rows.csv", index=False)

    drift = pd.DataFrame(gates["gate_d2"]["cells"])
    drift.to_csv(OUT / "ml14_drift_shift_cells.csv", index=False)

    categories = _old_new_category_compare()
    if not categories.empty:
        categories.to_csv(OUT / "ml13_vs_ml14_category_recall.csv", index=False)

    rfly_silver = pd.read_parquet(ROOT / "data/silver/rflymad_silver.parquet")
    rfly_report = _write_rfly_feature_report(rfly_silver)
    rfly_labels = (
        rfly_silver[["source_id", "label"]]
        .drop_duplicates()
        .groupby("label")
        .size()
        .sort_values(ascending=False)
    )
    rfly_labels.to_csv(OUT / "rflymad_label_counts.csv", header=["flight_count"])

    plt.figure(figsize=(7.2, 4.0))
    x = range(len(drift))
    labels = [f"{r.score_source}\n{r.budget}" for r in drift.itertuples()]
    plt.bar([i - 0.18 for i in x], drift["old_shift_ratio"], width=0.36, label="old")
    plt.bar([i + 0.18 for i in x], drift["new_shift_ratio"], width=0.36, label="ML-14")
    plt.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    plt.xticks(list(x), labels, fontsize=8)
    plt.ylabel("test FA / budget")
    plt.title("FA drift dropped after SEAD refresh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "ml14_fa_drift_old_vs_new.png")
    plt.close()

    plt.figure(figsize=(7.0, 4.2))
    for budget, marker in [("critical", "o"), ("advisory", "s")]:
        subset = best[best["budget"] == budget]
        plt.scatter(
            subset["mean_false_alarms_per_hour"],
            subset["mean_event_onset_recall"],
            marker=marker,
            label=budget,
            s=70,
            alpha=0.8,
        )
    plt.axvline(2.0, color="tab:red", linestyle="--", linewidth=1, label="critical FA")
    plt.axvline(12.0, color="tab:orange", linestyle="--", linewidth=1, label="advisory FA")
    plt.axhline(0.30, color="tab:red", linestyle=":", linewidth=1)
    plt.axhline(0.50, color="tab:orange", linestyle=":", linewidth=1)
    plt.xlabel("false alarms per hour")
    plt.ylabel("event onset recall")
    plt.title("ML-14 operating points: FA improved, recall still low")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "ml14_recall_vs_false_alarm.png")
    plt.close()

    plt.figure(figsize=(7.2, 4.0))
    subset_counts = pd.Series(rfly_report["case_subset_counts"]).sort_index()
    subset_counts.plot(kind="bar", color=["#4c78a8", "#f58518", "#54a24b"])
    plt.ylabel("parsed flights")
    plt.title("RFLYMAD parsed real-flight pool")
    plt.tight_layout()
    plt.savefig(FIG / "rflymad_parsed_pool.png")
    plt.close()

    gate_rows = [
        {"gate": "D1 rebuild", "status": gates["gate_d1"]["status"]},
        {"gate": "D2 FA drift", "status": gates["gate_d2"]["status"]},
        {"gate": "D3 operating target", "status": gates["gate_d3"]["status"]},
        {"gate": "RFLY split readiness", "status": rfly_report["status"]},
    ]
    pd.DataFrame(gate_rows).to_csv(OUT / "gate_status_summary.csv", index=False)

    best_critical = best[
        (best["budget"] == "critical")
        & (best["mean_false_alarms_per_hour"] <= 2.0)
    ].head(1)
    best_advisory = best[
        (best["budget"] == "advisory")
        & (best["mean_false_alarms_per_hour"] <= 12.0)
    ].head(1)
    critical_txt = "n/a"
    advisory_txt = "n/a"
    if not best_critical.empty:
        row = best_critical.iloc[0]
        critical_txt = (
            f"{row.score_source}/{row.decision}: recall={row.mean_event_onset_recall:.3f}, "
            f"FA/h={row.mean_false_alarms_per_hour:.2f}"
        )
    if not best_advisory.empty:
        row = best_advisory.iloc[0]
        advisory_txt = (
            f"{row.score_source}/{row.decision}: recall={row.mean_event_onset_recall:.3f}, "
            f"FA/h={row.mean_false_alarms_per_hour:.2f}"
        )

    md = f"""# Mentor Packet: ML-14 + RFLYMAD Status

## One-minute summary

We are building UAV anomaly detection, but the model must be judged by two
numbers together: how many true anomaly events it catches, and how many false
operator notifications it creates per hour.

ML-14 improved the false-alarm drift problem strongly, but did not yet restore
enough recall to pass the operating gate.

## Key numbers

- SEAD labels: {rebuild['parse_coverage']['labels_flights']}
- SEAD parsed flights: {rebuild['parse_coverage']['parsed_feature_flights']}
- SEAD normal flights parsed: {rebuild['parse_coverage']['parsed_class_counts'].get('Normal')}
- Development normal sessions: {rebuild['split_summary']['development_normal_sessions']}
- Holdout flights kept closed: {manifest['blind_holdout_flights']}
- D1 rebuild gate: {gates['gate_d1']['status']}
- D2 FA drift gate: {gates['gate_d2']['status']}
- Median FA shift ratio: {gates['gate_d2']['old_median_shift_ratio']:.2f} -> {gates['gate_d2']['new_median_shift_ratio']:.2f}
- Relative FA drift drop: {100 * gates['gate_d2']['relative_drop']:.1f}%
- D3 operating gate: {gates['gate_d3']['status']}
- Best critical under FA<=2: {critical_txt}
- Best advisory under FA<=12: {advisory_txt}

## RFLYMAD status

- Parsed RFLYMAD flights: {rfly_report['silver_flights']}
- Real-Motor: {rfly_report['case_subset_counts'].get('Real-Motor', 0)}
- Real-Sensors: {rfly_report['case_subset_counts'].get('Real-Sensors', 0)}
- Real-No_Fault normal: {rfly_report['case_subset_counts'].get('Real-No_Fault', 0)}
- Normal needed for registered split: {rfly_report['required_normal_for_registered_quota']}
- Missing normal flights: {rfly_report['normal_flights_missing']}
- RFLY status: {rfly_report['status']}

## How recall is computed

1. Each anomaly interval is converted into a boolean timeline: `y_true=True`
   inside the annotated anomaly interval.
2. The model produces an anomaly score for each time bucket.
3. A decision policy turns scores into alarms. We count only new alarm starts,
   not every row where the alarm remains active.
4. One anomaly event is counted as detected only if a new alarm starts inside
   that event interval.
5. `event_onset_recall = detected_events / n_events`.
6. `false_alarms_per_hour = false_alarm_events / normal_hours`, where false
   alarms are new alarm starts outside anomaly intervals.

So recall 0.126 means: among all true anomaly events in the evaluated splits,
about 12.6% produced a timely new operator alarm.

## Mentor-ready sentence

The refresh fixed a major measurement problem: false alarms no longer explode
from validation to test as badly. Median FA drift dropped from
{gates['gate_d2']['old_median_shift_ratio']:.2f}x budget to
{gates['gate_d2']['new_median_shift_ratio']:.2f}x budget. But the model became
too conservative, so recall is still far below the operating target. RFLYMAD is
the right next data source for motor/sensor faults, but it still needs
{rfly_report['normal_flights_missing']} more normal real flights before a
registered split/evaluation is honest.

## Files

- `gate_status_summary.csv`
- `ml14_best_operational_rows.csv`
- `ml14_drift_shift_cells.csv`
- `ml13_vs_ml14_category_recall.csv`
- `rflymad_label_counts.csv`
- `figures/ml14_fa_drift_old_vs_new.png`
- `figures/ml14_recall_vs_false_alarm.png`
- `figures/rflymad_parsed_pool.png`
"""
    (OUT / "mentor_summary.md").write_text(md, encoding="utf-8")

    print(f"Mentor packet written: {OUT}")


if __name__ == "__main__":
    main()
