import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rfly_full.supervised import alarm_onsets

EXPERIMENT_ROOT = (
    ROOT / "artifacts" / "rfly_full" / "v2" / "normal_temporal_ae"
    / "robustness" / "approved_20260722_nested_v1"
)
# Base (frozen, un-fine-tuned) rotation-0 AE scores, paired with R1's
# threshold-only recalibration on the same model/scaler.
SCORES_PATH = EXPERIMENT_ROOT / "base" / "rotation_0" / "development_scores.parquet"
OUTPUT = EXPERIMENT_ROOT / "candidates" / "R4"

INK = "#1b1b1b"
SCORE_COLOR = "#4c78a8"
CRITICAL_COLOR = "#e45756"
ADVISORY_COLOR = "#f2cf5b"
ADVISORY_EDGE = "#8a6d00"
TRUTH_COLOR = "#bab0ac"

CASES = [
    {
        "flight_id": "rfly_02274966d49a5cabb82c",
        "title": "Real / Motor — arıza penceresinde geç ve marjinal tespit",
        "critical_threshold": 20.08138341903688,
        "advisory_threshold": 12.019485449790954,
        "show_truth": True,
    },
    {
        "flight_id": "rfly_402c8be3d17c32a4c13d",
        "title": "Real / NoFault — arıza yokken üretilen yanlış alarmlar",
        "critical_threshold": 20.08138341903688,
        "advisory_threshold": 12.019485449790954,
        "show_truth": False,
    },
    {
        "flight_id": "rfly_097e5b35cd9e4bef962b",
        "title": "HIL / Wind (Environment) — sistem arızası yok, tekrarlayan nuisance alarm",
        "critical_threshold": 6.1543798200289395,
        "advisory_threshold": 4.167800505161288,
        "show_truth": False,
    },
]


def _plot_case(axis, frame: pd.DataFrame, case: dict) -> None:
    flight = frame.loc[frame["canonical_case_id"] == case["flight_id"]].sort_values("t_end_s")
    t = flight["t_end_s"].to_numpy()
    score = flight["score"].to_numpy()
    critical_thr = case["critical_threshold"]
    advisory_thr = case["advisory_threshold"]

    if case["show_truth"] and flight["fault_active"].any():
        active = flight["fault_active"].to_numpy()
        onset = t[np.flatnonzero(active)[0]]
        offset = t[np.flatnonzero(active)[-1]]
        axis.axvspan(
            onset, offset, color=TRUTH_COLOR, alpha=0.35, zorder=1,
            label="Gerçek arıza aralığı (truth)",
        )

    axis.plot(t, score, color=SCORE_COLOR, linewidth=1.6, zorder=3, label="Reconstruction score")
    axis.axhline(
        critical_thr, color=CRITICAL_COLOR, linestyle="--", linewidth=1.2, zorder=2,
        label=f"Critical eşiği ({critical_thr:.1f})",
    )
    axis.axhline(
        advisory_thr, color=ADVISORY_EDGE, linestyle="--", linewidth=1.2, zorder=2,
        label=f"Advisory eşiği ({advisory_thr:.1f})",
    )

    critical_onsets = alarm_onsets(score, critical_thr)
    advisory_onsets = alarm_onsets(score, advisory_thr)
    if critical_onsets.any():
        axis.scatter(
            t[critical_onsets], score[critical_onsets], marker="^", s=130,
            color=CRITICAL_COLOR, edgecolor=INK, linewidth=0.8, zorder=4,
            label="Critical alarm (4-of-6s, 30s refractory)",
        )
    if advisory_onsets.any():
        axis.scatter(
            t[advisory_onsets], score[advisory_onsets], marker="^", s=90,
            color=ADVISORY_COLOR, edgecolor=ADVISORY_EDGE, linewidth=0.8, zorder=4,
            label="Advisory alarm (4-of-6s, 30s refractory)",
        )

    axis.set_title(case["title"], fontsize=11, loc="left")
    axis.set_xlabel("Uçuş içi zaman (s)")
    axis.set_ylabel("Skor")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7.5, loc="upper left", ncol=2)


def main() -> None:
    frame = pd.read_parquet(SCORES_PATH)
    fig, axes = plt.subplots(len(CASES), 1, figsize=(11, 11.5))
    for axis, case in zip(axes, CASES):
        _plot_case(axis, frame, case)
    fig.suptitle(
        "RflyMAD-Full v2 — dondurulmuş temel AE: örnek uçuş bazında alarm davranışı\n"
        "(model/eşik: R1 threshold-only, rotation 0; yalnız görsel örnek, kapı kararı değildir)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.subplots_adjust(hspace=0.4)
    output_path = OUTPUT / "08_alarm_timeseries_case_studies.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
