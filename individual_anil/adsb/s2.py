"""S2 ADS-B status and data-quality reason codes.

This module is deliberately independent from the residual penalty scorer.  It
does not turn operational declarations (for example, a genuine 7700 squawk)
into attack ground truth and it never adds NIC/NACp/SIL values to a physics
penalty.  Instead it emits explicit row-level reason codes that can be
eventized and reported as natural operational burden.

Silver v2 carries ``<field>_updated``, ``<field>_update_timestamp_utc`` and
``<field>_update_age_s`` for sparse ``ac_dict`` fields.  Silver v1 lacks those
columns; such rows are classified as ``freshness_unknown`` rather than being
silently treated as fresh.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd


CRITICAL_SQUAWK_TYPES: dict[str, str] = {
    "7500": "unlawful",
    "7600": "nordo",
    "7700": "general",
}

CRITICAL_EMERGENCY_TYPES = frozenset(CRITICAL_SQUAWK_TYPES.values())
STANDALONE_EMERGENCY_TYPES = frozenset({"lifeguard", "minfuel", "downed", "reserved"})
INACTIVE_EMERGENCY_VALUES = frozenset({"", "none", "no_emergency", "noemergency"})


@dataclass(frozen=True)
class S2Config:
    """Explicit, serializable candidate semantics for S2 row classification.

    ``freshness_max_age_s`` is not learned from anomaly results.  Sixty seconds
    is the already accepted maximum continuous-observation gap used by the ADS-B
    feature/evaluation path.  Callers must serialize any override in their run
    manifest.

    FAA-reference advisories are opt-in per row.  A row is eligible only when
    ``faa_reference_scope_col`` exists and is exactly true, it is airborne, its
    source is ADS-B, and its ADS-B version is in ``faa_eligible_versions``.
    This prevents US-specific reference values from becoming global anomaly
    thresholds.
    """

    freshness_max_age_s: float = 60.0
    message_gap_threshold_s: float = 60.0
    faa_reference_scope_col: str = "faa_reference_scope"
    faa_eligible_versions: tuple[int, ...] = (2,)

    def __post_init__(self) -> None:
        if not math.isfinite(self.freshness_max_age_s) or self.freshness_max_age_s < 0:
            raise ValueError("freshness_max_age_s must be finite and >= 0")
        if not math.isfinite(self.message_gap_threshold_s) or self.message_gap_threshold_s <= 0:
            raise ValueError("message_gap_threshold_s must be finite and > 0")
        if not self.faa_reference_scope_col:
            raise ValueError("faa_reference_scope_col must be non-empty")
        if not self.faa_eligible_versions:
            raise ValueError("faa_eligible_versions must be non-empty")


def _column(frame: pd.DataFrame, name: str, default: object = None) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index, dtype=object)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _finite_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer_code(value: object) -> int | None:
    number = _finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _strict_true(value: object) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _normalise_emergency(value: object) -> str | None:
    if _is_missing(value):
        return None
    normalised = str(value).strip().casefold()
    normalised = re.sub(r"[\s\-/]+", "_", normalised)
    aliases = {
        "unlawful_interference": "unlawful",
        "minimum_fuel": "minfuel",
        "radio_failure": "nordo",
        "general_emergency": "general",
    }
    normalised = aliases.get(normalised, normalised)
    return None if normalised in INACTIVE_EMERGENCY_VALUES else normalised


def _normalise_squawk(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def _reason_token(value: str) -> str:
    token = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
    return token or "UNSPECIFIED"


def field_freshness(
    frame: pd.DataFrame,
    field: str,
    *,
    max_age_s: float,
) -> pd.Series:
    """Return per-row freshness without guessing for legacy Silver.

    Result values are ``fresh``, ``stale``, ``never_observed``,
    ``invalid_update_metadata`` or ``freshness_unknown``.  A current-row update
    is fresh even if its timestamp is unavailable; key presence itself proves
    that the value (including an explicit null clear) was reported on this row.
    """

    updated_col = f"{field}_updated"
    timestamp_col = f"{field}_update_timestamp_utc"
    age_col = f"{field}_update_age_s"
    if not {updated_col, timestamp_col, age_col}.issubset(frame.columns):
        return pd.Series(["freshness_unknown"] * len(frame), index=frame.index, dtype=object)

    states: list[str] = []
    for updated, updated_at, age in zip(
        frame[updated_col], frame[timestamp_col], frame[age_col]
    ):
        if _is_missing(updated):
            states.append("freshness_unknown")
            continue
        if _strict_true(updated):
            states.append("fresh")
            continue

        numeric_age = _finite_float(age)
        if numeric_age is not None:
            if numeric_age < 0:
                states.append("invalid_update_metadata")
            elif numeric_age <= max_age_s:
                states.append("fresh")
            else:
                states.append("stale")
        elif _is_missing(updated_at):
            states.append("never_observed")
        else:
            states.append("freshness_unknown")
    return pd.Series(states, index=frame.index, dtype=object)


def _declared_status(
    frame: pd.DataFrame,
    freshness: dict[str, pd.Series],
) -> dict[str, list[object]]:
    output: dict[str, list[object]] = {
        "declared_status_squawk_type": [],
        "declared_status_emergency_type": [],
        "declared_status_consistency": [],
        "declared_status_active": [],
        "declared_status_reason_codes": [],
    }

    squawks = _column(frame, "squawk")
    emergencies = _column(frame, "emergency")
    sq_fresh = freshness["squawk"]
    em_fresh = freshness["emergency"]

    for squawk, emergency, squawk_freshness, emergency_freshness in zip(
        squawks, emergencies, sq_fresh, em_fresh
    ):
        squawk_type = CRITICAL_SQUAWK_TYPES.get(_normalise_squawk(squawk) or "")
        emergency_type = _normalise_emergency(emergency)
        active = squawk_type is not None or emergency_type is not None
        reasons: list[str] = []

        if squawk_type is not None:
            reasons.append(f"DECLARED_SQUAWK_{_reason_token(squawk_type)}")
        if emergency_type is not None:
            # Known critical, standalone and provider-specific non-none values
            # all remain independent declarations; none is suppressed.
            reasons.append(f"DECLARED_EMERGENCY_{_reason_token(emergency_type)}")

        if not active:
            consistency = "not_applicable"
        elif (
            squawk_type is not None
            and emergency_type in CRITICAL_EMERGENCY_TYPES
            and squawk_freshness == "fresh"
            and emergency_freshness == "fresh"
        ):
            consistency = "corroborated" if squawk_type == emergency_type else "contradictory"
        else:
            consistency = "not_corroborated"

        if active:
            reasons.append(f"DECLARED_STATUS_{consistency.upper()}")
        output["declared_status_squawk_type"].append(squawk_type)
        output["declared_status_emergency_type"].append(emergency_type)
        output["declared_status_consistency"].append(consistency)
        output["declared_status_active"].append(active)
        output["declared_status_reason_codes"].append(tuple(reasons))
    return output


QUALITY_SPECS: dict[str, tuple[int, int]] = {
    # field: (largest provider-domain integer, FAA advisory reference)
    "nic": (11, 7),
    "nac_p": (11, 8),
    "sil": (3, 3),
}


def _adsb_version_status(value: object) -> str:
    if _is_missing(value):
        return "missing"
    code = _integer_code(value)
    if code is None or not 0 <= code <= 2:
        return "schema_invalid"
    return "valid"


def _faa_scope_status(row: dict[str, object], config: S2Config) -> str:
    if _strict_true(row.get("on_ground")):
        return "ground_not_applicable"

    source = row.get("ads_source_type")
    if _is_missing(source) or not str(source).strip().casefold().startswith("adsb"):
        return "not_adsb_source"

    version_status = _adsb_version_status(row.get("adsb_version"))
    if version_status == "missing":
        return "version_missing"
    if version_status == "schema_invalid":
        return "version_schema_invalid"
    version = _integer_code(row.get("adsb_version"))
    if version not in config.faa_eligible_versions:
        return "version_ineligible"

    if config.faa_reference_scope_col not in row:
        return "scope_not_asserted"
    scope = row.get(config.faa_reference_scope_col)
    if _is_missing(scope):
        return "scope_unknown"
    if not _strict_true(scope):
        return "outside_asserted_scope"
    return "eligible"


def _quality_value_status(field: str, value: object, *, faa_eligible: bool) -> str:
    if _is_missing(value):
        return "missing"
    code = _integer_code(value)
    maximum, reference = QUALITY_SPECS[field]
    if code is None or not 0 <= code <= maximum:
        return "schema_invalid"
    if code == 0:
        return "reported_unknown_or_unavailable"
    if faa_eligible:
        return "below_faa_reference" if code < reference else "meets_faa_reference"
    return "reported_valid_outside_faa_reference_scope"


def _position_quality(frame: pd.DataFrame, config: S2Config) -> dict[str, list[object]]:
    output: dict[str, list[object]] = {
        "position_quality_scope": [],
        "position_quality_adsb_version_status": [],
        "position_quality_advisory": [],
        "position_quality_reason_codes": [],
    }
    for field in QUALITY_SPECS:
        output[f"position_quality_{field}_status"] = []

    # Records preserve column names, including the explicit scope column when
    # present.  No geographic inference is attempted from lat/lon.
    for row in frame.to_dict(orient="records"):
        scope_status = _faa_scope_status(row, config)
        version_status = _adsb_version_status(row.get("adsb_version"))
        statuses = {
            field: _quality_value_status(
                field,
                row.get(field),
                faa_eligible=scope_status == "eligible",
            )
            for field in QUALITY_SPECS
        }

        reasons: list[str] = []
        if version_status == "schema_invalid":
            reasons.append("POSITION_QUALITY_ADSB_VERSION_SCHEMA_INVALID")
        for field, status in statuses.items():
            if status in {
                "schema_invalid",
                "missing",
                "reported_unknown_or_unavailable",
                "below_faa_reference",
            }:
                reasons.append(f"POSITION_QUALITY_{field.upper()}_{status.upper()}")

        output["position_quality_scope"].append(scope_status)
        output["position_quality_adsb_version_status"].append(version_status)
        output["position_quality_advisory"].append(
            any(status == "below_faa_reference" for status in statuses.values())
        )
        output["position_quality_reason_codes"].append(tuple(reasons))
        for field, status in statuses.items():
            output[f"position_quality_{field}_status"].append(status)
    return output


def _altitude_availability(frame: pd.DataFrame) -> list[str]:
    states: list[str] = []
    for on_ground, baro_alt, geom_alt in zip(
        _column(frame, "on_ground"),
        _column(frame, "alt"),
        _column(frame, "alt_geom_m"),
    ):
        if _strict_true(on_ground):
            state = "GROUND_ALT_NOT_APPLICABLE"
        elif _is_missing(baro_alt) and not _is_missing(geom_alt):
            state = "BARO_ALT_DROPOUT"
        elif _is_missing(baro_alt) and _is_missing(geom_alt):
            state = "ALL_ALTITUDE_UNAVAILABLE"
        else:
            state = "AVAILABLE"
        states.append(state)
    return states


def _group_key(value: object) -> tuple[str, str]:
    if _is_missing(value):
        return ("missing", "")
    return (type(value).__name__, str(value))


def _message_intervals(
    frame: pd.DataFrame,
    *,
    threshold_s: float,
    entity_col: str,
) -> dict[str, list[object]]:
    timestamps = _column(frame, "timestamp_utc")
    entities = _column(frame, entity_col, "__single_entity__")
    previous: dict[tuple[str, str], float] = {}

    result: dict[str, list[object]] = {
        "message_interval_status": [],
        "message_gap": [],
        "message_gap_s": [],
        "message_gap_interval_start": [],
        "message_gap_interval_end": [],
        "message_gap_reason_code": [],
    }
    for entity, timestamp in zip(entities, timestamps):
        key = _group_key(entity)
        current = _finite_float(timestamp)
        prior = previous.get(key)
        status: str
        gap = False
        duration: float | None = None
        start: float | None = None
        end: float | None = None
        reason: str | None = None

        if current is None:
            status = "TIMESTAMP_MISSING"
        elif prior is None:
            status = "FIRST_OBSERVATION"
            previous[key] = current
        else:
            delta = current - prior
            previous[key] = current
            if delta < 0:
                status = "NON_MONOTONIC_TIMESTAMP"
            elif delta > threshold_s:
                status = "MESSAGE_GAP"
                gap = True
                duration, start, end = delta, prior, current
                reason = "MESSAGE_GAP"
            else:
                status = "CONTIGUOUS"

        result["message_interval_status"].append(status)
        result["message_gap"].append(gap)
        result["message_gap_s"].append(duration)
        result["message_gap_interval_start"].append(start)
        result["message_gap_interval_end"].append(end)
        result["message_gap_reason_code"].append(reason)
    return result


def classify_s2_rows(
    frame: pd.DataFrame,
    *,
    config: S2Config = S2Config(),
    entity_col: str | None = None,
) -> pd.DataFrame:
    """Classify S2 declarations/quality/availability without an anomaly label.

    Input order is causal and is preserved.  ``entity_col`` controls message-gap
    resets; when omitted, ``flight_id`` is preferred, then ``source_id``, then a
    single-stream fallback.  The returned frame has the same index and contains
    deterministic tuples of reason codes suitable for long-form eventization.
    """

    if entity_col is None:
        entity_col = "flight_id" if "flight_id" in frame else "source_id"

    freshness = {
        field: field_freshness(frame, field, max_age_s=config.freshness_max_age_s)
        for field in ("squawk", "emergency", "nic", "nac_p", "sil", "adsb_version")
    }
    result = pd.DataFrame(index=frame.index)
    for field, values in freshness.items():
        result[f"{field}_freshness"] = values

    for name, values in _declared_status(frame, freshness).items():
        result[name] = values
    for name, values in _position_quality(frame, config).items():
        result[name] = values

    altitude_states = _altitude_availability(frame)
    result["altitude_availability"] = altitude_states
    result["altitude_availability_reason_code"] = [
        f"ALTITUDE_AVAILABILITY_{state}" for state in altitude_states
    ]

    for name, values in _message_intervals(
        frame,
        threshold_s=config.message_gap_threshold_s,
        entity_col=entity_col,
    ).items():
        result[name] = values

    combined: list[tuple[str, ...]] = []
    for declared, quality, altitude, gap in zip(
        result["declared_status_reason_codes"],
        result["position_quality_reason_codes"],
        result["altitude_availability_reason_code"],
        result["message_gap_reason_code"],
    ):
        codes = [*declared, *quality, altitude]
        if not _is_missing(gap):
            codes.append(gap)
        combined.append(tuple(codes))
    result["s2_reason_codes"] = combined

    # Retain only identity/time columns needed to produce independent natural
    # episode and burden tables.  Physics feature columns are not copied.
    for column in (entity_col, "flight_id", "source_id", "timestamp_utc"):
        if column in frame and column not in result:
            result[column] = frame[column]
    return result


def explode_s2_reasons(classified: pd.DataFrame) -> pd.DataFrame:
    """Convert row reason tuples to a stable long table for event/burden summaries."""

    if "s2_reason_codes" not in classified:
        raise KeyError("classified frame lacks s2_reason_codes")

    identity_columns = [
        column
        for column in ("flight_id", "source_id", "timestamp_utc")
        if column in classified.columns
    ]
    rows: list[dict[str, object]] = []
    for row_position, (_, row) in enumerate(classified.iterrows()):
        identity = {column: row[column] for column in identity_columns}
        for reason in row["s2_reason_codes"]:
            if reason.startswith("DECLARED_"):
                channel = "declared_status"
            elif reason.startswith("POSITION_QUALITY_"):
                channel = "position_quality"
            elif reason.startswith("ALTITUDE_AVAILABILITY_"):
                channel = "altitude_availability"
            elif reason == "MESSAGE_GAP":
                channel = "message_gap"
            else:
                channel = "unknown"
            rows.append(
                {
                    "row_position": row_position,
                    **identity,
                    "channel": channel,
                    "reason_code": reason,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["row_position", *identity_columns, "channel", "reason_code"],
    )


def reason_episodes(
    reason_rows: pd.DataFrame,
    *,
    entity_col: str,
    merge_gap_s: float = 60.0,
) -> pd.DataFrame:
    """Eventize equal reason codes without anomaly/false-positive claims.

    State reasons continue only across consecutive original row_position
    values. Thus an explicit inactive source row creates a new rising edge,
    while sparse message cadence alone does not. MESSAGE_GAP is a point event
    and every selected source row is its own episode.

    merge_gap_s remains part of the public API for compatibility and is
    validated, but it does not merge or split state episodes. Callers must
    provide row_position; timestamp-only merging cannot distinguish a sparse
    cadence from an intervening inactive row and therefore fails closed.
    """

    required = {entity_col, "row_position", "timestamp_utc", "channel", "reason_code"}
    missing = required - set(reason_rows.columns)
    if missing:
        raise KeyError(f"reason rows lack required columns: {sorted(missing)}")
    if not math.isfinite(merge_gap_s) or merge_gap_s < 0:
        raise ValueError("merge_gap_s must be finite and >= 0")

    episodes: list[dict[str, object]] = []
    groups: Iterable[tuple[tuple[object, object, object], pd.DataFrame]] = reason_rows.groupby(
        [entity_col, "channel", "reason_code"],
        sort=False,
        dropna=False,
    )
    for (entity, channel, reason), group in groups:
        positions = pd.to_numeric(group["row_position"], errors="coerce")
        times = pd.to_numeric(group["timestamp_utc"], errors="coerce")
        if (
            positions.isna().any()
            or times.isna().any()
            or not np.isfinite(positions.to_numpy(dtype=float)).all()
            or not np.isfinite(times.to_numpy(dtype=float)).all()
            or not np.equal(positions, np.floor(positions)).all()
        ):
            raise ValueError(
                "reason rows require finite integer row_position and finite timestamp_utc"
            )
        ordered = pd.DataFrame(
            {"row_position": positions.astype("int64"), "timestamp_utc": times}
        ).sort_values("row_position", kind="stable")

        if isinstance(reason, str) and reason == "MESSAGE_GAP":
            for timestamp in ordered["timestamp_utc"]:
                point = float(timestamp)
                episodes.append(
                    {
                        entity_col: entity,
                        "channel": channel,
                        "reason_code": reason,
                        "episode_start": point,
                        "episode_end": point,
                        "n_rows": 1,
                    }
                )
            continue

        first = ordered.iloc[0]
        start = end = float(first["timestamp_utc"])
        previous_position = int(first["row_position"])
        n_rows = 1
        for row in ordered.iloc[1:].itertuples(index=False):
            position = int(row.row_position)
            timestamp = float(row.timestamp_utc)
            if position == previous_position + 1:
                end = timestamp
                n_rows += 1
            else:
                episodes.append(
                    {
                        entity_col: entity,
                        "channel": channel,
                        "reason_code": reason,
                        "episode_start": start,
                        "episode_end": end,
                        "n_rows": n_rows,
                    }
                )
                start = end = timestamp
                n_rows = 1
            previous_position = position
        episodes.append(
            {
                entity_col: entity,
                "channel": channel,
                "reason_code": reason,
                "episode_start": start,
                "episode_end": end,
                "n_rows": n_rows,
            }
        )
    return pd.DataFrame(
        episodes,
        columns=[
            entity_col,
            "channel",
            "reason_code",
            "episode_start",
            "episode_end",
            "n_rows",
        ],
    )
