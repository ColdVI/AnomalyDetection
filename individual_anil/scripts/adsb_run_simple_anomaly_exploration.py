"""Run the two frozen, rule-only ADS-B discovery analyses on one Silver part."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adsb.features import build_feature_table
from adsb.segmentation import segment_flights
from adsb.simple_anomaly import (
    ALTITUDE_DEVIATION_M,
    ALTITUDE_MIN_DURATION_S,
    ROUTE_CONSECUTIVE_SAMPLES,
    ROUTE_DEVIATION_DEG,
    detect_altitude_deviation_events,
    detect_route_deviation_events,
    flight_phase,
)


CALIBRATION_PATH = ROOT / "adsb" / "reports" / "simple_phase_calibration_20260722.json"
CONTRACT_PATH = ROOT / "docs" / "ADSB_BASIT_ANOMALI_ONKAYIT_20260722.md"
OUTPUT_ROOT = ROOT / "artifacts" / "adsb" / "simple_anomaly_20260722"
ASSET_ROOT = ROOT / "docs" / "assets" / "adsb_simple_anomaly"
ALTITUDE_REPORT = ROOT / "docs" / "ADSB_BASIT_IRTIFA_KESIF_RAPORU_20260722.md"
ROUTE_REPORT = ROOT / "docs" / "ADSB_BASIT_ROTA_KESIF_RAPORU_20260722.md"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"

ALTITUDE_MANUAL_REVIEW = {
    "407ebf_002_alt_001": {
        "classification": "phase_boundary_descent_candidate",
        "confirmed_anomaly": False,
        "note": (
            "Altitude descends from about 548.6 m to 243.8 m before the strict "
            "phase rule starts landing; this is a phase-boundary false-positive candidate."
        ),
    },
    "a510bf_000_alt_001": {
        "classification": "multi_level_cruise_candidate",
        "confirmed_anomaly": False,
        "note": (
            "A stable 1402.1 m level is followed by a stable 1158.2 m level; the "
            "whole-flight cruise median makes the first level look anomalous, but "
            "the pattern is compatible with a legitimate altitude change."
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: pd.Series, points=(0.25, 0.50, 0.75, 0.95)) -> dict:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        f"p{int(round(point * 100)):02d}": (
            float(finite.quantile(point)) if len(finite) else None
        )
        for point in points
    }


def _load_exact_sample() -> tuple[pd.DataFrame, dict]:
    profile = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    if profile.get("anomaly_triggers_computed") is not False:
        raise RuntimeError("Calibration profile is not trigger-free")
    part = ROOT / profile["silver_part"]
    if _sha256(part) != profile["silver_part_sha256"]:
        raise RuntimeError("Frozen Silver part SHA-256 changed")
    columns = [
        "source_id", "timestamp_utc", "lat", "lon", "alt", "alt_geom_m",
        "ground_speed_ms", "track_deg", "vertical_rate_ms", "roll_deg",
        "flags_new_leg", "_source_file",
    ]
    frame = pq.read_table(part, columns=columns).to_pandas()
    segmented = segment_flights(frame, gap_s=1800.0)
    selection = profile["selection"]
    grouped = segmented.groupby("flight_id", sort=False)
    eligibility = grouped.agg(
        n_rows=("flight_id", "size"),
        start_time=("timestamp_utc", "min"),
        end_time=("timestamp_utc", "max"),
        alt_coverage=("alt", lambda values: float(values.notna().mean())),
        vr_coverage=("vertical_rate_ms", lambda values: float(values.notna().mean())),
    )
    eligibility["duration_s"] = eligibility["end_time"] - eligibility["start_time"]
    eligible = eligibility.loc[
        eligibility["n_rows"].ge(selection["min_rows"])
        & eligibility["duration_s"].ge(selection["min_duration_s"])
        & eligibility["alt_coverage"].ge(selection["min_signal_coverage"])
        & eligibility["vr_coverage"].ge(selection["min_signal_coverage"])
    ].copy()
    seed = int(selection["seed"])
    eligible["stable_rank"] = [
        hashlib.sha256(f"{seed}|{flight_id}".encode("utf-8")).hexdigest()
        for flight_id in eligible.index.astype(str)
    ]
    selected_ids = (
        eligible.sort_values(["stable_rank", "start_time"])
        .head(int(selection["sample_flights"]))
        .index.astype(str)
        .tolist()
    )
    identity_hash = hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest()
    if identity_hash != selection["flight_ids_sha256"]:
        raise RuntimeError("Stable calibration flight selection changed")
    sample = segmented.loc[
        segmented["flight_id"].astype(str).isin(selected_ids)
    ].copy()
    return sample, profile


def _stable_event_sample(events: pd.DataFrame, *, limit: int = 6) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    ranked = events.copy()
    ranked["_rank"] = ranked["event_id"].map(
        lambda value: hashlib.sha256(f"20260722|{value}".encode("utf-8")).hexdigest()
    )
    return ranked.sort_values("_rank").head(limit).drop(columns="_rank")


def _save(fig: plt.Figure, filename: str) -> Path:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    path = ASSET_ROOT / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_phase_distribution(features: pd.DataFrame) -> Path:
    order = ["takeoff", "cruise", "landing", "uncertain"]
    counts = features["flight_phase"].value_counts().reindex(order, fill_value=0)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(order, counts.values, color=["#f2cf5b", "#4c78a8", "#e45756", "#bab0ac"])
    axis.bar_label(bars, fmt="%d", padding=3)
    axis.set_title("Dondurulmuş üç-faz kuralı — satır kapsamı")
    axis.set_ylabel("Satır")
    axis.grid(axis="y", alpha=0.25)
    return _save(fig, "01_phase_distribution.png")


def _plot_altitude_summary(events: pd.DataFrame, evaluable_flights: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    triggered = int(events["flight_id"].nunique()) if len(events) else 0
    bars = axes[0].bar(
        ["Triggerlı", "Triggersız"],
        [triggered, max(evaluable_flights - triggered, 0)],
        color=["#e45756", "#72b7b2"],
    )
    axes[0].bar_label(bars, fmt="%d", padding=3)
    axes[0].set_title("Cruise-değerlendirilebilir uçuşlar")
    axes[0].set_ylabel("Uçuş")
    if len(events):
        axes[1].scatter(
            events["duration_s"] / 60.0,
            events["peak_abs_deviation_m"],
            c=events["data_quality_suspect"].map({True: "#e45756", False: "#4c78a8"}),
            alpha=0.75,
        )
    axes[1].axhline(ALTITUDE_DEVIATION_M, color="#e45756", linestyle="--")
    axes[1].axvline(ALTITUDE_MIN_DURATION_S / 60.0, color="#e45756", linestyle="--")
    axes[1].set_xlabel("Olay süresi (dakika)")
    axes[1].set_ylabel("Peak mutlak irtifa sapması (m)")
    axes[1].set_title("İrtifa olay şiddeti ve süre")
    for axis in axes:
        axis.grid(alpha=0.25)
    return _save(fig, "02_altitude_summary.png")


def _plot_altitude_examples(features: pd.DataFrame, sample: pd.DataFrame) -> Path:
    rows = max(len(sample), 1)
    fig, axes = plt.subplots(rows, 1, figsize=(13, 3.2 * rows), squeeze=False)
    if sample.empty:
        axes[0, 0].text(0.5, 0.5, "Dondurulmuş kuralda irtifa olayı yok", ha="center", va="center")
        axes[0, 0].set_axis_off()
    for axis, (_, event) in zip(axes[:, 0], sample.iterrows()):
        flight = features.loc[features["flight_id"].eq(event["flight_id"])].copy()
        window = flight.loc[
            flight["timestamp_utc"].between(event["start_time"] - 300, event["end_time"] + 300)
        ]
        minutes = (window["timestamp_utc"] - event["start_time"]) / 60.0
        axis.plot(minutes, window["alt"], color="#4c78a8", linewidth=1.2)
        axis.axvspan(0, event["duration_s"] / 60.0, color="#e45756", alpha=0.16)
        median = event["cruise_median_alt_m"]
        axis.axhline(median, color="#54a24b", linestyle="--", linewidth=1)
        axis.axhline(median + ALTITUDE_DEVIATION_M, color="#e45756", linestyle=":", linewidth=1)
        axis.axhline(median - ALTITUDE_DEVIATION_M, color="#e45756", linestyle=":", linewidth=1)
        axis.set_title(f"{event['event_id']} — {event['direction']}, {event['duration_s']:.0f} s")
        axis.set_xlabel("Olay başlangıcına göre dakika")
        axis.set_ylabel("alt (m)")
        axis.grid(alpha=0.25)
    return _save(fig, "03_altitude_examples.png")


def _plot_route_summary(events: pd.DataFrame, evaluable_flights: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    triggered = int(events["flight_id"].nunique()) if len(events) else 0
    bars = axes[0].bar(
        ["Triggerlı", "Triggersız"],
        [triggered, max(evaluable_flights - triggered, 0)],
        color=["#e45756", "#72b7b2"],
    )
    axes[0].bar_label(bars, fmt="%d", padding=3)
    axes[0].set_title("Heading-residual değerlendirilebilir uçuşlar")
    axes[0].set_ylabel("Uçuş")
    if len(events):
        axes[1].scatter(
            events["n_samples"], events["peak_abs_heading_residual_deg"],
            c=events["low_speed_context"].map({True: "#f2cf5b", False: "#4c78a8"}),
            alpha=0.65,
        )
    axes[1].axhline(ROUTE_DEVIATION_DEG, color="#e45756", linestyle="--")
    axes[1].axvline(ROUTE_CONSECUTIVE_SAMPLES, color="#e45756", linestyle="--")
    axes[1].set_xlabel("Ardışık örnek")
    axes[1].set_ylabel("Peak |heading residual| (derece)")
    axes[1].set_title("Rota olay şiddeti ve süreklilik")
    for axis in axes:
        axis.grid(alpha=0.25)
    return _save(fig, "04_route_summary.png")


def _plot_route_examples(features: pd.DataFrame, sample: pd.DataFrame) -> Path:
    rows = max(len(sample), 1)
    fig, axes = plt.subplots(rows, 1, figsize=(13, 3.2 * rows), squeeze=False)
    if sample.empty:
        axes[0, 0].text(0.5, 0.5, "Dondurulmuş kuralda rota olayı yok", ha="center", va="center")
        axes[0, 0].set_axis_off()
    for axis, (_, event) in zip(axes[:, 0], sample.iterrows()):
        flight = features.loc[features["flight_id"].eq(event["flight_id"])].copy()
        window = flight.loc[
            flight["timestamp_utc"].between(event["start_time"] - 120, event["end_time"] + 120)
        ]
        seconds = window["timestamp_utc"] - event["start_time"]
        axis.plot(seconds, window["heading_residual"], color="#4c78a8", linewidth=1.1)
        axis.axvspan(0, event["duration_s"], color="#e45756", alpha=0.16)
        axis.axhline(ROUTE_DEVIATION_DEG, color="#e45756", linestyle="--", linewidth=1)
        axis.axhline(-ROUTE_DEVIATION_DEG, color="#e45756", linestyle="--", linewidth=1)
        axis.set_title(
            f"{event['event_id']} — {event['n_samples']} örnek, "
            f"low-speed={event['low_speed_context']}"
        )
        axis.set_xlabel("Olay başlangıcına göre saniye")
        axis.set_ylabel("heading residual (°)")
        axis.grid(alpha=0.25)
    return _save(fig, "05_route_examples.png")


def _event_table(events: pd.DataFrame, columns: list[str]) -> str:
    if events.empty:
        return "Dondurulmuş stable-hash örnekleminde olay yok."
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for _, event in events.iterrows():
        values = []
        for column in columns:
            value = event[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _write_reports(summary: dict, altitude_sample: pd.DataFrame, route_sample: pd.DataFrame) -> None:
    phase = summary["phase"]
    altitude = summary["altitude"]
    route = summary["route"]
    common = (
        f"- Silver: `{summary['silver_part']}`\n"
        f"- Silver SHA-256: `{summary['silver_part_sha256']}`\n"
        f"- Ön-kayıt SHA-256: `{summary['contract_sha256']}`\n"
        f"- Sabit örnek: {summary['flights']} uçuş / {summary['rows']:,} satır\n"
        f"- Üç-faz çözülen: {phase['resolved_flights']}/{summary['flights']} uçuş\n"
        "- Ground truth yok; bu bir keşif/karakterizasyon raporudur.\n"
    )
    altitude_text = f"""# ADS-B Basit Anomali — İrtifa Keşif Raporu

> Dondurulmuş kural: cruise medyanından `±150 m`, `>120 s`.
> Operasyonel başarı/recall iddiası yoktur.

{common}
## Sonuç

- Cruise-değerlendirilebilir uçuş: {altitude['evaluable_flights']}
- En az bir olaylı uçuş: {altitude['triggered_flights']} ({altitude['triggered_flight_rate']:.2%})
- Olay sayısı: {altitude['events']}
- Data-quality-suspect olay: {altitude['data_quality_suspect_events']}
- Süre dağılımı (s): `{altitude['duration_s']}`
- Peak sapma dağılımı (m): `{altitude['peak_abs_deviation_m']}`
- Nitel incelemede doğrulanmış anomaly: {altitude['manual_review_confirmed_anomalies']}

![Faz kapsamı](assets/adsb_simple_anomaly/01_phase_distribution.png)

![İrtifa özeti](assets/adsb_simple_anomaly/02_altitude_summary.png)

![Stable-hash irtifa olay örnekleri](assets/adsb_simple_anomaly/03_altitude_examples.png)

## Stable-hash nitel inceleme örneği

{_event_table(altitude_sample, ['event_id', 'duration_s', 'direction', 'peak_abs_deviation_m', 'data_quality_suspect'])}

- `407ebf_002_alt_001`: cruise etiketi içindeki geç iniş/descent bölümü; faz
  sınırı false-positive adayı.
- `a510bf_000_alt_001`: yaklaşık 1402 m ve 1158 m'de iki stabil seviye; tüm-cruise
  medyanı ilk seviyeyi sapma sayıyor. Meşru seviye değişimiyle uyumlu.
- İki olayda da barometrik/geometrik kaynak uyuşmazlığı bayrağı yok. Bu nedenle
  ikisi de doğrulanmış anomaly değildir.

## Sınır

Kural yalnız üç fazı tam çözülen trace'lerin cruise bölümünü değerlendirir.
`uncertain` uçuşlar normal sayılmamış, kapsam dışı bırakılmıştır. Trigger bir
fiziksel olay adayıdır; doğrulanmış anomaly etiketi değildir.
"""
    route_text = f"""# ADS-B Basit Anomali — GPS/Rota Keşif Raporu

> Dondurulmuş kural: `|heading_residual| >=20°`, en az 4 ardışık örnek.
> Operasyonel başarı/recall iddiası yoktur.

{common}
## Sonuç

- Heading-residual değerlendirilebilir uçuş: {route['evaluable_flights']}
- En az bir olaylı uçuş: {route['triggered_flights']} ({route['triggered_flight_rate']:.2%})
- Olay sayısı: {route['events']}
- Düşük-hız bağlamlı olay: {route['low_speed_context_events']}
- Tamamı düşük-hız satırlarından oluşan olay: {route['all_samples_low_speed_events']}
- Cruise fazındaki olay: {route['cruise_events']}
- Ardışık örnek dağılımı: `{route['n_samples']}`
- Peak heading-residual dağılımı (derece): `{route['peak_abs_heading_residual_deg']}`
- Olay faz dağılımı: `{route['events_by_phase']}`

![Rota özeti](assets/adsb_simple_anomaly/04_route_summary.png)

![Stable-hash rota olay örnekleri](assets/adsb_simple_anomaly/05_route_examples.png)

## Stable-hash nitel inceleme örneği

{_event_table(route_sample, ['event_id', 'n_samples', 'duration_s', 'phase', 'peak_abs_heading_residual_deg', 'low_speed_context'])}

## Sınır

Kural bildirilen track ile iki konumdan türetilen bearing arasındaki iç
tutarsızlığı bulur; planlanan rotadan sapmayı doğrudan ölçmez. Düşük hız olayları
silinmemiş, bearing kararsızlığı bağlamı olarak ayrıca işaretlenmiştir. Trigger
doğrulanmış spoofing/anomaly etiketi değildir.

{route['events']} olayın {route['all_samples_low_speed_events']} tanesinde bütün
örnekler `<30 m/s`; en düşük olay-içi düşük-hız oranı
`{route['minimum_low_speed_fraction']:.2%}`dir. Cruise olayı
`{route['cruise_events']}`. Bu turdaki rota triggerları
GPS/rota anomaly kanıtından çok düşük-hız konum/bearing kararsızlığıdır.
"""
    ALTITUDE_REPORT.write_text(altitude_text, encoding="utf-8")
    ROUTE_REPORT.write_text(route_text, encoding="utf-8")


def main() -> None:
    if not CONTRACT_PATH.exists():
        raise RuntimeError("Frozen preregistration document is missing")
    sample, profile = _load_exact_sample()
    features = build_feature_table(sample)
    features["flight_phase"] = flight_phase(features)
    altitude_events = detect_altitude_deviation_events(features)
    route_events = detect_route_deviation_events(features)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_ROOT / "scored_rows.parquet", index=False)
    altitude_events.to_csv(OUTPUT_ROOT / "altitude_events.csv", index=False)
    route_events.to_csv(OUTPUT_ROOT / "route_events.csv", index=False)

    flight_phase_counts = features.groupby("flight_id")["flight_phase"].value_counts().unstack(fill_value=0)
    resolved_flights = int((flight_phase_counts.drop(columns="uncertain", errors="ignore").sum(axis=1) > 0).sum())
    altitude_evaluable = int(features.loc[features["flight_phase"].eq("cruise"), "flight_id"].nunique())
    route_evaluable = int(features.loc[features["heading_residual"].notna(), "flight_id"].nunique())
    altitude_triggered = int(altitude_events["flight_id"].nunique()) if len(altitude_events) else 0
    route_triggered = int(route_events["flight_id"].nunique()) if len(route_events) else 0
    altitude_sample = _stable_event_sample(altitude_events)
    route_sample = _stable_event_sample(route_events)

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "discovery_only",
        "silver_part": profile["silver_part"],
        "silver_part_sha256": profile["silver_part_sha256"],
        "contract": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "flights": int(features["flight_id"].nunique()),
        "rows": int(len(features)),
        "phase": {
            "resolved_flights": resolved_flights,
            "uncertain_flights": int(features["flight_id"].nunique() - resolved_flights),
            "rows_by_phase": {
                str(key): int(value)
                for key, value in features["flight_phase"].value_counts().items()
            },
        },
        "altitude": {
            "evaluable_flights": altitude_evaluable,
            "triggered_flights": altitude_triggered,
            "triggered_flight_rate": altitude_triggered / altitude_evaluable if altitude_evaluable else 0.0,
            "events": int(len(altitude_events)),
            "data_quality_suspect_events": int(altitude_events["data_quality_suspect"].sum()) if len(altitude_events) else 0,
            "duration_s": _quantiles(altitude_events["duration_s"]),
            "peak_abs_deviation_m": _quantiles(altitude_events["peak_abs_deviation_m"]),
            "review_event_ids": altitude_sample["event_id"].tolist(),
            "manual_review": ALTITUDE_MANUAL_REVIEW,
            "manual_review_confirmed_anomalies": int(sum(
                bool(item["confirmed_anomaly"])
                for item in ALTITUDE_MANUAL_REVIEW.values()
            )),
        },
        "route": {
            "evaluable_flights": route_evaluable,
            "triggered_flights": route_triggered,
            "triggered_flight_rate": route_triggered / route_evaluable if route_evaluable else 0.0,
            "events": int(len(route_events)),
            "low_speed_context_events": int(route_events["low_speed_context"].sum()) if len(route_events) else 0,
            "all_samples_low_speed_events": int(route_events["low_speed_fraction"].eq(1.0).sum()) if len(route_events) else 0,
            "minimum_low_speed_fraction": float(route_events["low_speed_fraction"].min()) if len(route_events) else None,
            "cruise_events": int(route_events["phase"].eq("cruise").sum()) if len(route_events) else 0,
            "n_samples": _quantiles(route_events["n_samples"]),
            "peak_abs_heading_residual_deg": _quantiles(route_events["peak_abs_heading_residual_deg"]),
            "events_by_phase": {
                str(key): int(value)
                for key, value in route_events["phase"].value_counts().items()
            },
            "review_event_ids": route_sample["event_id"].tolist(),
        },
        "operational_claim_allowed": False,
        "ground_truth_available": False,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _plot_phase_distribution(features)
    _plot_altitude_summary(altitude_events, altitude_evaluable)
    _plot_altitude_examples(features, altitude_sample)
    _plot_route_summary(route_events, route_evaluable)
    _plot_route_examples(features, route_sample)
    _write_reports(summary, altitude_sample, route_sample)
    print(SUMMARY_PATH.relative_to(ROOT))
    print(ALTITUDE_REPORT.relative_to(ROOT))
    print(ROUTE_REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
