"""Vectorized, bounded-memory natural-burden summaries for S2 reason channels."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from adsb.s2 import CRITICAL_SQUAWK_TYPES, INACTIVE_EMERGENCY_VALUES, QUALITY_SPECS


FRESHNESS_FIELDS = ("squawk", "emergency", "nic", "nac_p", "sil", "adsb_version")
FRESHNESS_COLUMNS = tuple(
    f"{field}_{suffix}"
    for field in FRESHNESS_FIELDS
    for suffix in ("updated", "update_timestamp_utc", "update_age_s")
)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normal_emergency(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.lower().str.replace(r"[\s\-/]+", "_", regex=True)
    return out.replace(
        {
            "unlawful_interference": "unlawful",
            "minimum_fuel": "minfuel",
            "radio_failure": "nordo",
            "general_emergency": "general",
            **{value: pd.NA for value in INACTIVE_EMERGENCY_VALUES},
        }
    )


def _normal_squawk(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    numeric = text.str.fullmatch(r"\d{1,4}", na=False)
    text.loc[numeric] = text.loc[numeric].str.zfill(4)
    return text


def _reason_token(value: str) -> str:
    token = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
    return token or "UNSPECIFIED"


def _reason_episode_summary(
    flight_ids: pd.Series,
    timestamps: pd.Series,
    reasons: Iterable[tuple[str, np.ndarray]],
    *,
    point_event_reasons: frozenset[str] = frozenset(),
) -> dict[str, dict[str, int]]:
    """Count state rising edges; point-event reasons count every selected row.

    A sparse cadence gap does not by itself create a new state episode.  The
    caller has already segmented flights, and an inactive row between two
    active rows is an explicit rising edge.  MESSAGE_GAP is different: every
    post-gap row represents one distinct interval event.
    """

    result: dict[str, dict[str, int]] = {}
    flight = flight_ids.astype("string").reset_index(drop=True)
    time = _numeric(timestamps).reset_index(drop=True)
    if flight.isna().any():
        raise ValueError("flight_id cannot be null in an S2 part")
    if not np.isfinite(time.to_numpy(dtype=float)).all():
        raise ValueError("timestamp_utc must be finite in an S2 part")
    positions = np.arange(len(flight))
    for reason, raw_mask in reasons:
        mask = np.asarray(raw_mask, dtype=bool)
        if len(mask) != len(flight):
            raise ValueError(f"reason mask length mismatch: {reason}")
        selected = np.flatnonzero(mask)
        if not len(selected):
            result[reason] = {"rows": 0, "episodes": 0, "flights": 0}
            continue
        f = flight.iloc[selected].reset_index(drop=True)
        p = positions[selected]
        if reason in point_event_reasons:
            new_episode = np.ones(len(selected), dtype=bool)
        else:
            new_episode = (
                np.r_[True, f.iloc[1:].to_numpy() != f.iloc[:-1].to_numpy()]
                | np.r_[True, np.diff(p) != 1]
            )
        result[reason] = {
            "rows": int(len(selected)),
            "episodes": int(np.count_nonzero(new_episode)),
            "flights": int(f.nunique()),
        }
    return result


def _freshness_states(frame: pd.DataFrame, field: str, *, max_age_s: float) -> pd.Series:
    needed = {f"{field}_updated", f"{field}_update_timestamp_utc", f"{field}_update_age_s"}
    present = needed & set(frame.columns)
    if not present:
        return pd.Series("freshness_unknown", index=frame.index, dtype="string")
    if present != needed:
        raise KeyError(
            f"Partial freshness schema for {field}: missing {sorted(needed - present)}"
        )
    updated = frame[f"{field}_updated"].astype("boolean")
    age = _numeric(frame[f"{field}_update_age_s"])
    updated_at = frame[f"{field}_update_timestamp_utc"]
    states = pd.Series("freshness_unknown", index=frame.index, dtype="string")
    states.loc[updated.eq(True)] = "fresh"
    not_updated = updated.eq(False)
    finite_age = pd.Series(
        np.isfinite(age.to_numpy(dtype=float)), index=frame.index, dtype="boolean"
    )
    states.loc[not_updated & finite_age & age.ge(0) & age.le(max_age_s)] = "fresh"
    states.loc[not_updated & finite_age & age.gt(max_age_s)] = "stale"
    states.loc[not_updated & finite_age & age.lt(0)] = "invalid_update_metadata"
    states.loc[not_updated & ~finite_age & updated_at.isna()] = "never_observed"
    return states


def _quality_status(series: pd.Series, *, maximum: int) -> pd.Series:
    numeric = _numeric(series)
    status = pd.Series("reported_valid_outside_faa_reference_scope", index=series.index, dtype="string")
    status.loc[series.isna()] = "missing"
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)), index=series.index, dtype="boolean"
    )
    safe_numeric = numeric.where(finite, 0.0)
    integer = numeric.notna() & finite & np.equal(safe_numeric, np.floor(safe_numeric))
    status.loc[series.notna() & (~integer | numeric.lt(0) | numeric.gt(maximum))] = "schema_invalid"
    status.loc[integer & numeric.eq(0)] = "reported_unknown_or_unavailable"
    return status


def summarize_s2_part(
    frame: pd.DataFrame,
    *,
    scoreable_max_gap_s: float = 60.0,
    message_gap_threshold_s: float = 60.0,
    freshness_max_age_s: float = 60.0,
) -> dict:
    """Summarize one already segmented/sorted part without retaining row output."""

    for name, value in (
        ("scoreable_max_gap_s", scoreable_max_gap_s),
        ("message_gap_threshold_s", message_gap_threshold_s),
        ("freshness_max_age_s", freshness_max_age_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0")

    required = {
        "flight_id", "timestamp_utc", "squawk", "emergency", "nic", "nac_p", "sil",
        "adsb_version", "alt", "alt_geom_m", "on_ground",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"S2 part missing columns: {sorted(missing)}")
    if frame["flight_id"].isna().any():
        raise ValueError("flight_id cannot be null in an S2 part")
    ordered = frame.sort_values(
        ["flight_id", "timestamp_utc"], kind="mergesort"
    ).reset_index(drop=True)
    flight = ordered["flight_id"]
    time = _numeric(ordered["timestamp_utc"])
    if not np.isfinite(time.to_numpy(dtype=float)).all():
        raise ValueError("timestamp_utc must be finite in an S2 part")
    same_flight = flight.eq(flight.shift(1))
    dt = time - time.shift(1)
    exposure_mask = same_flight & dt.gt(0) & dt.le(scoreable_max_gap_s)
    exposure_s = float(dt.where(exposure_mask, 0.0).sum())

    squawk = _normal_squawk(ordered["squawk"])
    emergency = _normal_emergency(ordered["emergency"])
    freshness_states = {
        field: _freshness_states(ordered, field, max_age_s=freshness_max_age_s)
        for field in FRESHNESS_FIELDS
    }
    reasons: dict[str, np.ndarray] = {}

    def add_reason(name: str, mask: pd.Series | np.ndarray) -> None:
        values = np.asarray(mask, dtype=bool)
        if len(values) != len(ordered):
            raise ValueError(f"reason mask length mismatch: {name}")
        reasons[name] = reasons.get(name, np.zeros(len(ordered), dtype=bool)) | values

    for code, event_type in CRITICAL_SQUAWK_TYPES.items():
        add_reason(
            f"DECLARED_SQUAWK_{event_type.upper()}",
            squawk.eq(code).fillna(False).to_numpy(),
        )
    for event_type in sorted(set(emergency.dropna().astype(str))):
        add_reason(
            f"DECLARED_EMERGENCY_{_reason_token(event_type)}",
            emergency.eq(event_type).fillna(False).to_numpy(),
        )
    declared_active = squawk.isin(CRITICAL_SQUAWK_TYPES) | emergency.notna()
    squawk_type = squawk.map(CRITICAL_SQUAWK_TYPES)
    both_fresh = freshness_states["squawk"].eq("fresh") & freshness_states[
        "emergency"
    ].eq("fresh")
    both_critical = squawk_type.notna() & emergency.isin(
        frozenset(CRITICAL_SQUAWK_TYPES.values())
    )
    corroborated = (both_fresh & both_critical & squawk_type.eq(emergency)).fillna(False)
    contradictory = (
        both_fresh & both_critical & squawk_type.ne(emergency)
    ).fillna(False)
    not_corroborated = declared_active & ~(corroborated | contradictory)
    add_reason("DECLARED_STATUS_CORROBORATED", corroborated.to_numpy())
    add_reason("DECLARED_STATUS_CONTRADICTORY", contradictory.to_numpy())
    add_reason("DECLARED_STATUS_NOT_CORROBORATED", not_corroborated.to_numpy())

    quality_counts: dict[str, dict[str, int]] = {}
    for field, (maximum, _reference) in QUALITY_SPECS.items():
        status = _quality_status(ordered[field], maximum=maximum)
        quality_counts[field] = {str(k): int(v) for k, v in status.value_counts().items()}
        for flagged in ("schema_invalid", "missing", "reported_unknown_or_unavailable"):
            add_reason(
                f"POSITION_QUALITY_{field.upper()}_{flagged.upper()}",
                status.eq(flagged).to_numpy(),
            )
    version_numeric = _numeric(ordered["adsb_version"])
    version_status = pd.Series("valid", index=ordered.index, dtype="string")
    version_status.loc[ordered["adsb_version"].isna()] = "missing"
    version_finite = pd.Series(
        np.isfinite(version_numeric.to_numpy(dtype=float)),
        index=ordered.index,
        dtype="boolean",
    )
    safe_version = version_numeric.where(version_finite, 0.0)
    version_integer = (
        version_numeric.notna()
        & version_finite
        & np.equal(safe_version, np.floor(safe_version))
    )
    version_status.loc[
        ordered["adsb_version"].notna()
        & (~version_integer | version_numeric.lt(0) | version_numeric.gt(2))
    ] = "schema_invalid"
    quality_counts["adsb_version"] = {str(k): int(v) for k, v in version_status.value_counts().items()}
    add_reason(
        "POSITION_QUALITY_ADSB_VERSION_SCHEMA_INVALID",
        version_status.eq("schema_invalid").to_numpy(),
    )

    ground = ordered["on_ground"].astype("boolean").eq(True).fillna(False)
    baro_missing = ordered["alt"].isna()
    geom_missing = ordered["alt_geom_m"].isna()
    altitude = pd.Series("AVAILABLE", index=ordered.index, dtype="string")
    altitude.loc[ground] = "GROUND_ALT_NOT_APPLICABLE"
    altitude.loc[~ground & baro_missing & ~geom_missing] = "BARO_ALT_DROPOUT"
    altitude.loc[~ground & baro_missing & geom_missing] = "ALL_ALTITUDE_UNAVAILABLE"
    altitude_counts = {str(k): int(v) for k, v in altitude.value_counts().items()}
    altitude_exposure_s = {
        state: float(dt.where(exposure_mask & altitude.eq(state), 0.0).sum())
        for state in altitude_counts
    }
    for state in ("BARO_ALT_DROPOUT", "ALL_ALTITUDE_UNAVAILABLE"):
        add_reason(f"ALTITUDE_AVAILABILITY_{state}", altitude.eq(state).to_numpy())

    message_gap = same_flight & dt.gt(message_gap_threshold_s)
    add_reason("MESSAGE_GAP", message_gap.fillna(False).to_numpy())
    reason_summary = _reason_episode_summary(
        flight,
        time,
        reasons.items(),
        point_event_reasons=frozenset({"MESSAGE_GAP"}),
    )

    matching = (
        (squawk.eq("7500") & emergency.eq("unlawful"))
        | (squawk.eq("7600") & emergency.eq("nordo"))
        | (squawk.eq("7700") & emergency.eq("general"))
    ).fillna(False)
    legacy_matching_not_corroborated = (
        matching
        & not_corroborated
        & freshness_states["squawk"].eq("freshness_unknown")
        & freshness_states["emergency"].eq("freshness_unknown")
    )
    return {
        "contract": {
            "scoreable_max_gap_s": float(scoreable_max_gap_s),
            "message_gap_threshold_s": float(message_gap_threshold_s),
            "freshness_max_age_s": float(freshness_max_age_s),
            "state_episode_semantics": "rising_edge_or_new_flight",
            "message_gap_episode_semantics": "one_point_event_per_post_gap_row",
        },
        "n_rows": int(len(ordered)),
        "n_flights": int(flight.nunique()),
        "scoreable_flight_hours": exposure_s / 3600.0,
        "scoreable_exposure_definition": (
            "sum positive within-flight intervals <= scoreable_max_gap_s; "
            "interval state is assigned to its ending row"
        ),
        "freshness": {
            field: dict(Counter(freshness_states[field].astype(str)))
            for field in FRESHNESS_FIELDS
        },
        "declared_status": {
            "active_rows": int(declared_active.sum()),
            "matching_value_rows": int(matching.sum()),
            "legacy_matching_rows_not_corroboration": int(
                legacy_matching_not_corroborated.sum()
            ),
            "consistency_row_counts": {
                "corroborated": int(corroborated.sum()),
                "contradictory": int(contradictory.sum()),
                "not_corroborated": int(not_corroborated.sum()),
            },
            "freshness_policy": (
                "matching values corroborate only when both fields are fresh; "
                "legacy rows remain not_corroborated"
            ),
        },
        "position_quality": {
            "faa_reference_scope": "not_asserted; no below_faa_reference advisory emitted",
            "status_counts": quality_counts,
        },
        "altitude_availability": {
            "row_counts": altitude_counts,
            "exposure_seconds_by_state": altitude_exposure_s,
        },
        "reason_burden": reason_summary,
    }


def merge_s2_summaries(parts: Iterable[dict]) -> dict:
    """Merge disjoint-part summaries (caller must enforce source non-overlap)."""

    parts = list(parts)
    contracts = [item["contract"] for item in parts]
    if contracts and any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Cannot merge S2 summaries with different contracts")
    total_rows = sum(item["n_rows"] for item in parts)
    total_flights = sum(item["n_flights"] for item in parts)
    total_hours = sum(item["scoreable_flight_hours"] for item in parts)
    if total_rows < 0 or total_flights < 0 or not math.isfinite(total_hours) or total_hours < 0:
        raise ValueError("Invalid S2 summary totals")
    reason_names = sorted({name for item in parts for name in item["reason_burden"]})
    burden: dict[str, dict] = {}
    for name in reason_names:
        rows = sum(item["reason_burden"].get(name, {}).get("rows", 0) for item in parts)
        episodes = sum(item["reason_burden"].get(name, {}).get("episodes", 0) for item in parts)
        flights = sum(item["reason_burden"].get(name, {}).get("flights", 0) for item in parts)
        burden[name] = {
            "rows": rows,
            "episodes": episodes,
            "flights": flights,
            "row_fraction": rows / total_rows if total_rows else None,
            "rows_per_scoreable_flight_hour": rows / total_hours if total_hours else None,
            "episodes_per_scoreable_flight_hour": episodes / total_hours if total_hours else None,
            "flight_fraction": flights / total_flights if total_flights else None,
        }

    freshness: dict[str, dict[str, int]] = {}
    for field in FRESHNESS_FIELDS:
        counter: Counter = Counter()
        for item in parts:
            counter.update(item["freshness"][field])
        freshness[field] = dict(counter)
    altitude_rows: Counter = Counter()
    altitude_exposure: Counter = Counter()
    declared_active_rows = 0
    declared_matching_value_rows = 0
    declared_matching_rows = 0
    declared_consistency: Counter = Counter()
    quality_status: dict[str, Counter] = {
        field: Counter() for field in (*QUALITY_SPECS, "adsb_version")
    }
    for item in parts:
        altitude_rows.update(item["altitude_availability"]["row_counts"])
        altitude_exposure.update(item["altitude_availability"]["exposure_seconds_by_state"])
        declared_active_rows += item["declared_status"]["active_rows"]
        declared_matching_value_rows += item["declared_status"]["matching_value_rows"]
        declared_matching_rows += item["declared_status"]["legacy_matching_rows_not_corroboration"]
        declared_consistency.update(item["declared_status"]["consistency_row_counts"])
        for field, counts in item["position_quality"]["status_counts"].items():
            quality_status[field].update(counts)
    return {
        "contract": contracts[0] if contracts else None,
        "n_rows": total_rows,
        "n_flights": total_flights,
        "scoreable_flight_hours": total_hours,
        "scoreable_exposure_definition": (
            "sum positive within-flight intervals <= scoreable_max_gap_s; "
            "interval state is assigned to its ending row"
        ),
        "burden_units": {
            "rows": "row count and fraction of all rows",
            "episodes": "state rising edges; MESSAGE_GAP counts one point event per gap row",
            "flights": "flight-segment count and fraction of all flight segments",
            "rate": "per scoreable flight-hour",
        },
        "freshness": freshness,
        "declared_status": {
            "active_rows": declared_active_rows,
            "active_row_fraction": (
                declared_active_rows / total_rows if total_rows else None
            ),
            "matching_value_rows": declared_matching_value_rows,
            "legacy_matching_rows_not_corroboration": declared_matching_rows,
            "consistency_row_counts": dict(declared_consistency),
            "ground_truth_claim": "none; declarations are operational states, not attacks",
        },
        "position_quality": {
            "faa_reference_scope": "not_asserted; no below_faa_reference advisory emitted",
            "status_counts": {field: dict(counts) for field, counts in quality_status.items()},
        },
        "altitude_availability": {
            "row_counts": dict(altitude_rows),
            "exposure_seconds_by_state": dict(altitude_exposure),
        },
        "reason_burden": burden,
    }
