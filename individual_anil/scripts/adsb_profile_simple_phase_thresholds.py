"""Profile one real Silver part for the simple ADS-B phase preregistration.

This script deliberately does not run either anomaly rule.  It reads only the
columns needed for segmentation plus altitude/vertical-rate, chooses a stable
100-flight sample, and records natural measurement distributions so thresholds
can be frozen before trigger counts are inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adsb.segmentation import segment_flights


SILVER_DIR = ROOT / "data" / "objectstore" / "silver" / "adsblol_historical"
REPORT_JSON = ROOT / "adsb" / "reports" / "simple_phase_calibration_20260722.json"
REPORT_MD = ROOT / "adsb" / "reports" / "simple_phase_calibration_20260722.md"
SEED = 20260722
SAMPLE_FLIGHTS = 100
MIN_ROWS = 20
MIN_DURATION_S = 300.0
MIN_SIGNAL_COVERAGE = 0.60


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: pd.Series, points: tuple[float, ...]) -> dict[str, float | None]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    result: dict[str, float | None] = {}
    for point in points:
        label = f"p{int(round(point * 100)):02d}"
        result[label] = float(finite.quantile(point)) if len(finite) else None
    return result


def _stable_rank(flight_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{flight_id}".encode("utf-8")).hexdigest()


def _select_part(path: Path | None) -> Path:
    if path is not None:
        selected = path.resolve()
    else:
        candidates = sorted(SILVER_DIR.glob("*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No Silver parquet files under {SILVER_DIR}")
        selected = candidates[0].resolve()
    if selected.suffix.lower() != ".parquet" or not selected.is_file():
        raise ValueError(f"Expected an existing parquet file: {selected}")
    try:
        selected.relative_to(SILVER_DIR.resolve())
    except ValueError as exc:
        raise ValueError("Calibration input must stay inside the open Silver directory") from exc
    return selected


def build_profile(part: Path) -> dict:
    columns = [
        "source_id", "timestamp_utc", "lat", "lon", "alt",
        "vertical_rate_ms", "flags_new_leg", "_source_file",
    ]
    table = pq.read_table(part, columns=columns)
    frame = table.to_pandas()
    segmented = segment_flights(frame, gap_s=1800.0)

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
        eligibility["n_rows"].ge(MIN_ROWS)
        & eligibility["duration_s"].ge(MIN_DURATION_S)
        & eligibility["alt_coverage"].ge(MIN_SIGNAL_COVERAGE)
        & eligibility["vr_coverage"].ge(MIN_SIGNAL_COVERAGE)
    ].copy()
    eligible["stable_rank"] = [_stable_rank(str(value)) for value in eligible.index]
    selected_ids = (
        eligible.sort_values(["stable_rank", "start_time"])
        .head(SAMPLE_FLIGHTS)
        .index.astype(str)
        .tolist()
    )
    if len(selected_ids) < SAMPLE_FLIGHTS:
        raise RuntimeError(
            f"Only {len(selected_ids)} eligible flights; expected {SAMPLE_FLIGHTS}"
        )
    sample = segmented.loc[segmented["flight_id"].astype(str).isin(selected_ids)].copy()

    sample["dt_s"] = sample.groupby("flight_id", sort=False)["timestamp_utc"].diff()
    positive_dt = sample.loc[sample["dt_s"].gt(0), "dt_s"]
    finite_vr = pd.to_numeric(sample["vertical_rate_ms"], errors="coerce")
    finite_alt = pd.to_numeric(sample["alt"], errors="coerce")
    abs_vr = finite_vr.abs()

    flight_altitude = sample.groupby("flight_id", sort=False)["alt"].agg(
        q05=lambda values: values.quantile(0.05),
        median="median",
        q95=lambda values: values.quantile(0.95),
    )
    flight_altitude["robust_range_m"] = flight_altitude["q95"] - flight_altitude["q05"]

    source_files = sorted(sample["_source_file"].dropna().astype(str).unique().tolist())
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "threshold_calibration_only",
        "anomaly_triggers_computed": False,
        "silver_part": part.relative_to(ROOT).as_posix(),
        "silver_part_sha256": _sha256(part),
        "silver_footer_rows": int(table.num_rows),
        "source_files": source_files,
        "selection": {
            "seed": SEED,
            "sample_flights": len(selected_ids),
            "eligible_flights": int(len(eligible)),
            "min_rows": MIN_ROWS,
            "min_duration_s": MIN_DURATION_S,
            "min_signal_coverage": MIN_SIGNAL_COVERAGE,
            "flight_ids_sha256": hashlib.sha256(
                "\n".join(selected_ids).encode("utf-8")
            ).hexdigest(),
        },
        "sample": {
            "rows": int(len(sample)),
            "duration_s": _quantiles(
                eligibility.loc[selected_ids, "duration_s"],
                (0.05, 0.25, 0.50, 0.75, 0.95),
            ),
            "sampling_interval_s": _quantiles(
                positive_dt, (0.50, 0.75, 0.90, 0.95, 0.99),
            ),
            "vertical_rate_ms": _quantiles(
                finite_vr, (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
            ),
            "absolute_vertical_rate_ms": _quantiles(
                abs_vr, (0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
            ),
            "altitude_m": _quantiles(
                finite_alt, (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
            ),
            "flight_robust_altitude_range_m": _quantiles(
                flight_altitude["robust_range_m"],
                (0.05, 0.25, 0.50, 0.75, 0.95),
            ),
            "candidate_threshold_occupancy": {
                "abs_vr_lt_1_0_fraction": float(abs_vr.lt(1.0).mean()),
                "vr_gt_2_5_fraction": float(finite_vr.gt(2.5).mean()),
                "vr_lt_minus_2_5_fraction": float(finite_vr.lt(-2.5).mean()),
            },
        },
    }


def _markdown(profile: dict) -> str:
    sample = profile["sample"]
    occupancy = sample["candidate_threshold_occupancy"]
    lines = [
        "# ADS-B Basit Anomali — Faz Eşiği Kalibrasyon Profili",
        "",
        "> Bu rapor yalnız `vertical_rate_ms`/`alt` doğal dağılımını profiller.",
        "> Anomali tetik sayısı hesaplanmamıştır.",
        "",
        f"- Silver parçası: `{profile['silver_part']}`",
        f"- SHA-256: `{profile['silver_part_sha256']}`",
        f"- Kaynak: `{', '.join(profile['source_files'])}`",
        f"- Örnek: {profile['selection']['sample_flights']} uçuş / {sample['rows']:,} satır",
        f"- Eligible havuz: {profile['selection']['eligible_flights']} uçuş",
        "",
        "## Dağılımlar",
        "",
        f"- Örnekleme aralığı (s): `{sample['sampling_interval_s']}`",
        f"- Dikey hız (m/s): `{sample['vertical_rate_ms']}`",
        f"- Mutlak dikey hız (m/s): `{sample['absolute_vertical_rate_ms']}`",
        f"- İrtifa (m): `{sample['altitude_m']}`",
        f"- Uçuş robust irtifa aralığı (m): `{sample['flight_robust_altitude_range_m']}`",
        "",
        "## Taslak eşiklerin doğal kapsama oranı",
        "",
        f"- `|vertical_rate_ms| < 1.0`: {occupancy['abs_vr_lt_1_0_fraction']:.3%}",
        f"- `vertical_rate_ms > 2.5`: {occupancy['vr_gt_2_5_fraction']:.3%}",
        f"- `vertical_rate_ms < -2.5`: {occupancy['vr_lt_minus_2_5_fraction']:.3%}",
        "",
        "Bu oranlar anomali sonucu veya başarı metriği değildir; yalnız faz sınırı ön-kayıt girdisidir.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver-part", type=Path)
    parser.add_argument("--json-output", type=Path, default=REPORT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=REPORT_MD)
    args = parser.parse_args()

    selected = _select_part(args.silver_part)
    profile = build_profile(selected)
    for output in (args.json_output, args.markdown_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(_markdown(profile), encoding="utf-8")
    print(args.json_output.relative_to(ROOT))
    print(args.markdown_output.relative_to(ROOT))


if __name__ == "__main__":
    main()
