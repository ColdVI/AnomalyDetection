"""Deterministic, memory-bounded sampling from readsb globe-history tar archives."""

from __future__ import annotations

import hashlib
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.silver.parse_adsblol_historical import parse_trace_bytes


DATE_RE = re.compile(r"v(\d{4}\.\d{2}\.\d{2})-planes-readsb")


@dataclass(frozen=True)
class SampleResult:
    rows: pd.DataFrame
    selected_members: dict[str, list[str]]
    archive_stats: list[dict]


def archive_date(path: str | Path) -> str:
    """Return YYYY-MM-DD encoded in an adsb.lol archive filename."""

    match = DATE_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Archive name has no readsb date: {path}")
    return match.group(1).replace(".", "-")


def _is_trace_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return "/traces/" in f"/{normalized}" and normalized.endswith((".json", ".json.gz"))


def _member_rank(archive_name: str, member_name: str, seed: int) -> str:
    value = f"{seed}:{archive_name}:{member_name}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _segment_trace(
    frame: pd.DataFrame,
    *,
    archive_day: str,
    gap_s: float,
    min_points: int,
    min_duration_s: float,
) -> list[pd.DataFrame]:
    if frame.empty or frame["source_id"].isna().all():
        return []
    frame = frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").copy()
    dt = frame["timestamp_utc"].diff()
    new_leg = frame["flags_new_leg"].fillna(False).astype(bool)
    boundary = dt.isna() | (dt > gap_s) | new_leg
    frame["_segment"] = boundary.cumsum().astype(int) - 1
    results: list[pd.DataFrame] = []
    icao = str(frame["source_id"].iloc[0])
    for segment_id, group in frame.groupby("_segment", sort=True):
        group = group.drop(columns="_segment").copy()
        duration = float(group["timestamp_utc"].max() - group["timestamp_utc"].min())
        if len(group) < min_points or duration < min_duration_s:
            continue
        group["archive_date"] = archive_day
        group["flight_id"] = f"{archive_day}:{icao}:{int(segment_id):03d}"
        results.append(group)
    return results


def sample_archives(
    archive_paths: list[str | Path],
    *,
    members_per_archive: int = 250,
    seed: int = 20260710,
    gap_s: float = 1800.0,
    min_points: int = 60,
    min_duration_s: float = 600.0,
    max_flights_per_archive: int | None = 500,
) -> SampleResult:
    """Hash-sample aircraft members and return qualifying flight segments.

    Selection depends only on archive/member names and ``seed``. No anomaly result is
    consulted, making repeated pilot runs reproducible without scanning every payload.
    """

    all_flights: list[pd.DataFrame] = []
    selected_members: dict[str, list[str]] = {}
    archive_stats: list[dict] = []

    for archive in sorted((Path(path) for path in archive_paths), key=archive_date):
        day = archive_date(archive)
        with tarfile.open(archive, mode="r:*") as tar:
            candidates = [member for member in tar.getmembers() if member.isfile() and _is_trace_member(member.name)]
            candidates.sort(key=lambda member: _member_rank(archive.name, member.name, seed))
            chosen = candidates[:members_per_archive]
            selected_members[archive.name] = [member.name for member in chosen]

            flights_before = len(all_flights)
            parsed_members = 0
            parse_errors = 0
            for member in chosen:
                try:
                    handle = tar.extractfile(member)
                    if handle is None:
                        continue
                    frame = parse_trace_bytes(handle.read())
                    parsed_members += 1
                    if not frame.empty:
                        frame = frame[
                            frame["ads_source_type"].eq("adsb_icao")
                            & frame["lat"].between(-90, 90)
                            & frame["lon"].between(-180, 180)
                        ]
                    all_flights.extend(
                        _segment_trace(
                            frame,
                            archive_day=day,
                            gap_s=gap_s,
                            min_points=min_points,
                            min_duration_s=min_duration_s,
                        )
                    )
                except Exception:
                    parse_errors += 1

            new_flights = all_flights[flights_before:]
            if max_flights_per_archive is not None and len(new_flights) > max_flights_per_archive:
                ranked = sorted(
                    new_flights,
                    key=lambda group: hashlib.sha256(
                        f"{seed}:{group['flight_id'].iloc[0]}".encode("utf-8")
                    ).hexdigest(),
                )[:max_flights_per_archive]
                all_flights[flights_before:] = ranked
                new_flights = ranked

            archive_stats.append(
                {
                    "archive": archive.name,
                    "archive_date": day,
                    "trace_members_total": len(candidates),
                    "trace_members_selected": len(chosen),
                    "trace_members_parsed": parsed_members,
                    "parse_errors": parse_errors,
                    "qualifying_flights": len(new_flights),
                    "qualifying_rows": int(sum(len(group) for group in new_flights)),
                }
            )

    rows = pd.concat(all_flights, ignore_index=True) if all_flights else pd.DataFrame()
    return SampleResult(rows=rows, selected_members=selected_members, archive_stats=archive_stats)
