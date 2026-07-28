"""Create executive ADS-B research progression and anomaly-detector map plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _progression(output: Path) -> None:
    stages = [
        ("ADR-025/032", "Rule baseline > several NN trials\nAE / USAD: NO-GO", "#8D99AE"),
        ("ADR-029", "Two-axis causal\nPage-CUSUM core", "#457B9D"),
        ("ADR-035", "Contextual v1 training\nMagnitude gate: PASS", "#2A9D8F"),
        ("ADR-042", "Truth-v2: CUSUM 49.72% @ V=5\nMost model profiles <6%\nDevelopment rejected", "#E9C46A"),
        ("v2 prereg", "Grid 0.1–500\nCumulative persistence_v2\n275.5M natural rows", "#F4A261"),
        ("ADR-046", "Speed persistence 57.18% @ V=0.1\nPosition CUSUM 51.27% @ V=5\nUniversal NO-GO / scoped research GO", "#E76F51"),
    ]
    fig, axis = plt.subplots(figsize=(18, 7))
    x = np.arange(len(stages), dtype=float)
    axis.plot(x, np.zeros_like(x), color="#343A40", lw=3, zorder=1)
    for index, (name, text, color) in enumerate(stages):
        axis.scatter(index, 0, s=520, color=color, edgecolor="white", linewidth=2, zorder=3)
        offset = 0.28 if index % 2 == 0 else -0.28
        valign = "bottom" if offset > 0 else "top"
        axis.plot([index, index], [0, offset * 0.82], color=color, lw=2)
        axis.text(index, offset, f"{name}\n{text}", ha="center", va=valign, fontsize=10, weight="bold" if index in {0, 5} else "normal")
    axis.set_xlim(-0.55, len(stages) - 0.45)
    axis.set_ylim(-0.75, 0.75)
    axis.axis("off")
    axis.set_title("ADS-B anomaly detection research progression — evidence carried forward, not reset", fontsize=16, weight="bold", pad=18)
    axis.text(0.5, 0.02, "Each redesign was pre-registered; failed gates were retained as evidence rather than tuned away.", transform=axis.transAxes, ha="center", fontsize=10, color="#444444")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _detector_map(output: Path) -> None:
    rows = ["Vertical-rate freeze", "Ground-speed bias", "Track freeze", "Position stealth ramp", "Altitude dropout"]
    columns = ["Instant model", "Persistence_v2", "Two-axis CUSUM"]
    values = np.asarray([
        [0.0000, 0.263426, np.nan],
        [0.122537, 0.571801, np.nan],
        [np.nan, 0.148183, np.nan],
        [np.nan, np.nan, 0.512653],
        [np.nan, np.nan, np.nan],
    ])
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#E9ECEF")
    fig, axis = plt.subplots(figsize=(12, 7))
    image = axis.imshow(masked, vmin=0.0, vmax=0.6, cmap=cmap, aspect="auto")
    axis.set_xticks(range(len(columns)), columns)
    axis.set_yticks(range(len(rows)), rows)
    axis.set_title("Frozen V=5 — anomaly mechanism × detector assignment and event recall", fontsize=14, weight="bold")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                label = f"{100.0 * value:.2f}% recall"
                color = "white" if value > 0.35 else "#111111"
            elif row == 4:
                label = "S2 data-quality\nout of scope"
                color = "#555555"
            else:
                label = "not assigned"
                color = "#777777"
            axis.text(column, row, label, ha="center", va="center", fontsize=10, color=color, weight="bold" if np.isfinite(value) else "normal")
    bar = fig.colorbar(image, ax=axis, shrink=0.82)
    bar.set_label("observable-event recall")
    axis.text(0.0, -0.14, "V=5 means a total calibration budget of 5 episodes per 100 scoreable flight-hours; it is not a confidence score.", transform=axis.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(out_dir: Path) -> list[Path]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    outputs = [out_dir / "adsb_research_progression.png", out_dir / "anomaly_detector_map_V5.png"]
    _progression(outputs[0])
    _detector_map(outputs[1])
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/adsb/plots/reporting_summary"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps({"outputs": [str(path) for path in run(args.out_dir.resolve())]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
