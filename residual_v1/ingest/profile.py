"""Per-flight telemetry hygiene profiling."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from residual_v1.ingest.alfa_channels import CHANNELS as ALFA_CHANNELS
from residual_v1.ingest.common import write_json
from residual_v1.ingest.rfly_channels import CHANNELS as RFLY_CHANNELS

DT_BINS_S = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, float("inf"))


def find_flight_roots(silver_root: str | Path) -> list[Path]:
    return sorted(path.parent for path in Path(silver_root).rglob("flight.json"))


def stale_segments(
    time: pd.Series, values: pd.Series, *, minimum_duration_s: float = 2.0
) -> list[dict[str, float]]:
    t = pd.to_numeric(time, errors="coerce").to_numpy(float)
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = np.isfinite(t) & np.isfinite(x)
    if len(t) == 0:
        return []
    continues = np.zeros(len(t), dtype=bool)
    continues[1:] = (
        finite[1:]
        & finite[:-1]
        & np.isclose(x[1:], x[:-1], rtol=0.0, atol=1e-12)
    )
    starts = np.flatnonzero(~continues)
    ends = np.r_[starts[1:] - 1, len(t) - 1]
    retained = finite[starts] & ((t[ends] - t[starts]) >= minimum_duration_s)
    return [
        {"start_s": float(t[start]), "end_s": float(t[end])}
        for start, end in zip(starts[retained], ends[retained], strict=True)
    ]


def _dt_histogram(time: pd.Series) -> dict[str, int]:
    values = pd.to_numeric(time, errors="coerce").dropna().to_numpy(float)
    dt = np.diff(values)
    counts, _ = np.histogram(dt[np.isfinite(dt)], bins=np.asarray(DT_BINS_S))
    return {
        f"[{DT_BINS_S[index]},{DT_BINS_S[index + 1]})": int(count)
        for index, count in enumerate(counts)
    }


def profile_flight(flight_root: str | Path, *, dataset: str) -> dict:
    root = Path(flight_root)
    specs = ALFA_CHANNELS if dataset == "alfa" else RFLY_CHANNELS
    spec_by_name = {spec.name: spec for spec in specs}
    metadata = json.loads((root / "flight.json").read_text(encoding="utf-8"))
    topics: dict[str, dict] = {}
    quarantine_reasons: list[str] = []
    duration_s = 0.0
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if "t" not in frame:
            continue
        time = pd.to_numeric(frame["t"], errors="coerce")
        if time.notna().any():
            duration_s = max(duration_s, float(time.max() - time.min()))
        channels: dict[str, dict] = {}
        for column in frame.columns.difference(["t"]):
            values = pd.to_numeric(frame[column], errors="coerce")
            valid_count = int(values.notna().sum())
            spec = spec_by_name.get(column)
            violation_count = 0
            if spec is not None and valid_count:
                violation_count = int(((values < spec.valid_min) | (values > spec.valid_max)).sum())
                if violation_count / valid_count > 0.01:
                    quarantine_reasons.append(
                        f"{path.stem}.{column}: range violation ratio {violation_count / valid_count:.6f}"
                    )
            sensor_stale_segments = (
                stale_segments(time, values)
                if spec is not None and spec.nominal_hz >= 20.0 and spec.role != "command"
                else []
            )
            channels[column] = {
                "null_ratio": float(values.isna().mean()),
                "valid_count": valid_count,
                "range_violation_count": violation_count,
                "stale_segments": sensor_stale_segments,
            }
        topics[path.stem] = {
            "rows": int(len(frame)),
            "dt_histogram_s": _dt_histogram(time),
            "channels": channels,
        }
    return {
        "dataset": dataset,
        "flight_id": metadata["flight_id"],
        "duration_s": duration_s,
        "topics": topics,
        "quarantined": bool(quarantine_reasons),
        "quarantine_reasons": sorted(set(quarantine_reasons)),
    }


def _profile_html(profile: dict) -> str:
    rows = []
    for topic, topic_payload in profile["topics"].items():
        for channel, payload in topic_payload["channels"].items():
            rows.append(
                "<tr>"
                f"<td>{html.escape(topic)}</td><td>{html.escape(channel)}</td>"
                f"<td>{payload['null_ratio']:.6f}</td>"
                f"<td>{payload['range_violation_count']}</td>"
                f"<td>{len(payload['stale_segments'])}</td></tr>"
            )
    return (
        "<!doctype html><meta charset='utf-8'><title>RESIDUAL-V1 profile</title>"
        f"<h1>{html.escape(profile['flight_id'])}</h1>"
        f"<p>duration_s={profile['duration_s']:.3f}; quarantined={profile['quarantined']}</p>"
        "<table><thead><tr><th>topic</th><th>channel</th><th>null ratio</th>"
        "<th>range violations</th><th>stale segments</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def profile_dataset(
    silver_root: str | Path,
    output_root: str | Path,
    *,
    dataset: str,
) -> dict:
    if dataset not in {"alfa", "rfly"}:
        raise ValueError("dataset must be alfa or rfly")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    profiles = []
    for flight_root in find_flight_roots(silver_root):
        profile = profile_flight(flight_root, dataset=dataset)
        profiles.append(profile)
        relative = Path(profile["flight_id"])
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target.with_suffix(".json"), profile, fail_if_exists=True)
        target.with_suffix(".html").write_text(_profile_html(profile), encoding="utf-8")
    quarantined = {
        profile["flight_id"]: profile["quarantine_reasons"]
        for profile in profiles
        if profile["quarantined"]
    }
    summary = {
        "dataset": dataset,
        "flight_count": len(profiles),
        "quarantine_count": len(quarantined),
        "quarantined_flights": quarantined,
    }
    write_json(output / "summary.json", summary, fail_if_exists=True)
    write_json(Path(silver_root) / "quarantine.json", quarantined)
    return summary
