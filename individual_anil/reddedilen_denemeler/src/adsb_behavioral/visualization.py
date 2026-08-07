"""Static diagnostic plots for ADS-B Stage-1 results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.adsb_behavioral.physics_residuals import MODEL_FEATURES


def plot_correlation(train: pd.DataFrame, output: str | Path) -> None:
    quality = train.loc[train["quality_good"], MODEL_FEATURES]
    sample = quality.sample(n=min(25_000, len(quality)), random_state=20260710)
    corr = sample.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [name.replace("abs_", "") for name in corr]
    ax.set_xticks(range(len(corr)), labels, rotation=70, ha="right")
    ax.set_yticks(range(len(corr)), labels)
    ax.set_title("ADS-B physics residual Spearman correlation (train-normal)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_injected_example(
    flight: pd.DataFrame,
    *,
    score_col: str,
    threshold: float,
    output: str | Path,
) -> None:
    ordered = flight.sort_values("timestamp_utc")
    t0 = ordered["timestamp_utc"].min()
    t = ordered["timestamp_utc"] - t0
    start = float(ordered["event_start_utc"].iloc[0] - t0)
    end = float(ordered["event_end_utc"].iloc[0] - t0)
    fig, axes = plt.subplots(5, 1, figsize=(12, 13))
    anomaly = ordered["is_injected_anomaly"].fillna(False)
    axes[0].plot(ordered["lon"], ordered["lat"], color="steelblue", linewidth=1.0)
    axes[0].scatter(ordered.loc[anomaly, "lon"], ordered.loc[anomaly, "lat"], s=9, color="crimson")
    axes[0].set_title(f"Route — {ordered['injection_type'].iloc[0]} / {ordered['severity'].iloc[0]}")
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    axes[1].plot(t, ordered["alt"], label="barometric")
    axes[1].plot(t, ordered["alt_geom_m"], label="geometric", alpha=0.7)
    axes[1].set_ylabel("altitude m")
    axes[2].plot(t, ordered["ground_speed_ms"], label="reported")
    axes[2].plot(t, ordered["position_speed_mps"], label="position-derived", alpha=0.8)
    axes[2].set_ylabel("speed m/s")
    axes[3].plot(t, ordered["vertical_rate_ms"], label="reported")
    axes[3].plot(t, ordered["derived_vrate_mps"], label="derived", alpha=0.8)
    axes[3].set_ylabel("vertical m/s")
    axes[4].plot(t, ordered[score_col], label=score_col)
    axes[4].axhline(threshold, linestyle="--", color="black", label="threshold")
    axes[4].set_ylabel("anomaly score")
    axes[4].set_xlabel("seconds from flight start")
    for ax in axes[1:]:
        ax.axvspan(start, end, color="crimson", alpha=0.15)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
