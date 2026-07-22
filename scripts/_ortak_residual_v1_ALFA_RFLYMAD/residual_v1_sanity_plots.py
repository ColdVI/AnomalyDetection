from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gecmis_calismalar.residual_v1.features.align import align_to_clock, default_tolerances, observed_tolerances
from gecmis_calismalar.residual_v1.features.physics import finite_difference
from gecmis_calismalar.residual_v1.ingest.common import write_json
from gecmis_calismalar.residual_v1.run import create_run_dir, update_manifest
from gecmis_calismalar.residual_v1.tracking import log_run


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _spearman_or_none(left: pd.Series, right: pd.Series) -> tuple[float | None, str]:
    valid = pd.concat([left, right], axis=1).dropna()
    if len(valid) < 3:
        return None, "insufficient_samples"
    if valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return None, "constant_input"
    statistic = float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)
    if not np.isfinite(statistic):
        return None, "undefined"
    return statistic, "defined"


def _metadata(silver_root: Path, flight_id: str) -> dict:
    return json.loads((silver_root / flight_id / "flight.json").read_text(encoding="utf-8"))


def _events(silver_root: Path, flight_id: str) -> list[dict]:
    return json.loads((silver_root / flight_id / "events.json").read_text(encoding="utf-8"))


def _topics(silver_root: Path, flight_id: str, names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    root = silver_root / flight_id
    return {name: pd.read_parquet(root / f"{name}.parquet") for name in names}


def _engine_plot(
    silver_root: Path,
    flight_id: str,
    plot_dir: Path,
) -> dict:
    flight = _topics(
        silver_root,
        flight_id,
        ("mavros-nav_info-airspeed", "mavros-vfr_hud"),
    )
    aligned = align_to_clock(
        flight,
        "mavros-nav_info-airspeed",
        observed_tolerances(flight, default_tolerances("alfa")),
    )
    derivative = finite_difference(aligned["t"], aligned["airspeed"], window_s=0.5)
    events = _events(silver_root, flight_id)
    onset = float(events[0]["onset_s"])
    post = (aligned["t"] >= onset) & (aligned["t"] <= onset + 5.0)
    post_values = derivative.loc[post].dropna()
    median_derivative = float(post_values.median()) if not post_values.empty else float("nan")
    negative_fraction = float((post_values < 0).mean()) if not post_values.empty else float("nan")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(aligned["t"], aligned["throttle_cmd"], linewidth=0.9, color="#d95f02")
    axes[0].set_ylabel("throttle ratio")
    axes[1].plot(aligned["t"], aligned["airspeed"], linewidth=0.9, color="#1b9e77")
    axes[1].set_ylabel("airspeed m/s")
    axes[2].plot(aligned["t"], derivative, linewidth=0.9, color="#7570b3")
    axes[2].axhline(0.0, color="black", linewidth=0.6)
    axes[2].set_ylabel("dV/dt m/s²")
    axes[2].set_xlabel("flight time s")
    for axis in axes:
        axis.axvline(onset, color="#e7298a", linestyle="--", linewidth=1.2, label="fault onset")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    fig.suptitle(f"R4 raw command/response sanity — {flight_id}")
    fig.tight_layout()
    path = plot_dir / f"engine_R4_{_safe(flight_id)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "flight_id": flight_id,
        "plot": str(path),
        "onset_s": onset,
        "post_5s_median_airspeed_derivative_m_s2": median_derivative,
        "post_5s_negative_derivative_fraction": negative_fraction,
        "sentences": [
            f"Onset sonrası ilk 5 saniyede ortanca hava-hızı türevi {median_derivative:.3f} m/s² ve negatif örnek oranı %{100.0 * negative_fraction:.1f}.",
            "Grafik yalnız throttle komutu, ölçülen airspeed ve gözlenen-örnek fark türevini gösterir; model tahmini veya kalibre skor içermez.",
        ],
    }


def _aggressiveness(silver_root: Path, flight_id: str) -> tuple[float, pd.DataFrame]:
    flight = _topics(
        silver_root,
        flight_id,
        ("mavros-nav_info-roll", "mavros-rc-out", "mavros-imu-data"),
    )
    aligned = align_to_clock(
        flight,
        "mavros-nav_info-roll",
        observed_tolerances(flight, default_tolerances("alfa")),
    )
    roll_deg = np.rad2deg(pd.to_numeric(aligned["roll"], errors="coerce")).abs()
    rate_deg = np.rad2deg(pd.to_numeric(aligned["roll_rate"], errors="coerce")).abs()
    metric = float(np.nanquantile(np.maximum(roll_deg, rate_deg), 0.99))
    aligned["aggressiveness"] = np.maximum(roll_deg, rate_deg)
    return metric, aligned


def _maneuver_plot(
    flight_id: str,
    aligned: pd.DataFrame,
    metric: float,
    plot_dir: Path,
) -> dict:
    peak_index = int(pd.to_numeric(aligned["aggressiveness"], errors="coerce").idxmax())
    peak_time = float(aligned.loc[peak_index, "t"])
    window = aligned.loc[(aligned["t"] >= peak_time - 15.0) & (aligned["t"] <= peak_time + 15.0)].copy()
    rho, correlation_status = _spearman_or_none(
        window["aileron_cmd"], window["roll_rate"]
    )
    rho_text = f"{rho:.3f}" if rho is not None else f"tanımsız ({correlation_status})"

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(window["t"], window["aileron_cmd"], color="#d95f02", linewidth=0.9)
    axes[0].set_ylabel("aileron PWM delta")
    axes[1].plot(window["t"], np.rad2deg(window["roll_rate"]), color="#1b9e77", linewidth=0.9)
    axes[1].set_ylabel("roll rate deg/s")
    axes[1].set_xlabel("flight time s")
    for axis in axes:
        axis.axvline(peak_time, color="#7570b3", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.25)
    fig.suptitle(f"R1 aggressive normal maneuver sanity — {flight_id}")
    fig.tight_layout()
    path = plot_dir / f"normal_R1_{_safe(flight_id)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "flight_id": flight_id,
        "plot": str(path),
        "aggressiveness_q99": metric,
        "peak_time_s": peak_time,
        "command_response_spearman_rho": rho,
        "command_response_correlation_status": correlation_status,
        "sentences": [
            f"En agresif 30 saniyelik pencerede aileron komutu–roll-rate eşzamanlı Spearman ρ={rho_text}; agresiflik q99={metric:.2f}.",
            "Bu grafik arızasız uçuşta yalnız R1 girdisi ve tepkisini gösterir; residual/model skoru sonraki insan onayından önce hesaplanmamıştır.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RESIDUAL-V1 Phase-C visual sanity STOP")
    parser.add_argument("--silver-root", default="artifacts/residual_v1/silver/alfa")
    parser.add_argument("--split-manifest", default="artifacts/residual_v1/splits/alfa_seed11.json")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    silver_root = Path(args.silver_root)
    split_path = Path(args.split_manifest)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development = split["partitions"]["development"]["flight_ids"]
    engine_ids = [
        flight_id
        for flight_id in development
        if _metadata(silver_root, flight_id)["fault_class"] == "engine"
        and _events(silver_root, flight_id)
    ]
    normal_ids = [
        flight_id
        for flight_id in development
        if _metadata(silver_root, flight_id)["fault_class"] == "normal"
    ]
    if not engine_ids or len(normal_ids) < 3:
        raise RuntimeError("sanity STOP requires one development engine event and three normal flights")

    run_dir, _ = create_run_dir(
        "sanity",
        seed=args.seed,
        config_paths=["configs/residual_v1_phases.json"],
        input_paths=[split_path],
    )
    plot_dir = run_dir / "plots"
    plot_dir.mkdir()
    engine_result = _engine_plot(silver_root, engine_ids[0], plot_dir)
    ranked = []
    for flight_id in normal_ids:
        metric, aligned = _aggressiveness(silver_root, flight_id)
        ranked.append((metric, flight_id, aligned))
    ranked.sort(key=lambda value: value[0], reverse=True)
    maneuver_results = [
        _maneuver_plot(flight_id, aligned, metric, plot_dir)
        for metric, flight_id, aligned in ranked[:3]
    ]
    payload = {
        "stop": "task_3.5_visual_approval_required",
        "split_role": "development_only",
        "engine": engine_result,
        "aggressive_normal": maneuver_results,
    }
    write_json(run_dir / "sanity_metrics.json", payload, fail_if_exists=True)
    lines = [
        "# RESIDUAL-V1 Görev 3.5 Sanity Raporu",
        "",
        "Durum: **STOP — insan görsel onayı gerekli.** Tüm uçuşlar yalnız development bölmesinden seçildi.",
        "",
        "## Engine / R4",
        "",
        f"- Uçuş: `{engine_result['flight_id']}`",
        f"- Grafik: `{Path(engine_result['plot']).name}`",
        f"- {engine_result['sentences'][0]}",
        f"- {engine_result['sentences'][1]}",
        "",
        "## Agresif normal / R1",
        "",
    ]
    for result in maneuver_results:
        lines.extend(
            [
                f"### {result['flight_id']}",
                "",
                f"- Grafik: `{Path(result['plot']).name}`",
                f"- {result['sentences'][0]}",
                f"- {result['sentences'][1]}",
                "",
            ]
        )
    (run_dir / "SANITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    tracking = log_run(
        run_dir,
        run_name="phaseC_sanity_stop",
        metrics={
            "engine_post_negative_fraction": engine_result["post_5s_negative_derivative_fraction"],
            "normal_flights_plotted": len(maneuver_results),
        },
        params={"split_role": "development"},
    )
    update_manifest(
        run_dir,
        status="STOP_visual_approval_required",
        split_role="development_only",
        mlflow_status=tracking["status"],
    )
    print(run_dir)


if __name__ == "__main__":
    main()
