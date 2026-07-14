"""Tests for the independent S2 status/data-quality channel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adsb.s2 import (
    S2Config,
    classify_s2_rows,
    explode_s2_reasons,
    field_freshness,
    reason_episodes,
)


FRESH_FIELDS = ("squawk", "emergency", "nic", "nac_p", "sil", "adsb_version")


def _with_fresh_metadata(frame: pd.DataFrame, *, updated: bool = True, age_s: float = 0.0):
    result = frame.copy()
    for field in FRESH_FIELDS:
        result[f"{field}_updated"] = updated
        result[f"{field}_update_timestamp_utc"] = 1000.0 if updated or age_s >= 0 else np.nan
        result[f"{field}_update_age_s"] = age_s
    return result


def _base_row(**overrides):
    row = {
        "flight_id": "f1",
        "source_id": "abc123",
        "timestamp_utc": 1000.0,
        "on_ground": False,
        "alt": 1000.0,
        "alt_geom_m": 1010.0,
        "ads_source_type": "adsb_icao",
        "squawk": "1200",
        "emergency": "none",
        "nic": 8,
        "nac_p": 9,
        "sil": 3,
        "adsb_version": 2,
        "faa_reference_scope": True,
    }
    row.update(overrides)
    return row


def test_matching_fresh_critical_declarations_are_corroborated_and_independent():
    frame = _with_fresh_metadata(pd.DataFrame([_base_row(squawk="7700", emergency="general")]))

    result = classify_s2_rows(frame)

    assert result.loc[0, "declared_status_squawk_type"] == "general"
    assert result.loc[0, "declared_status_emergency_type"] == "general"
    assert result.loc[0, "declared_status_consistency"] == "corroborated"
    assert result.loc[0, "declared_status_reason_codes"] == (
        "DECLARED_SQUAWK_GENERAL",
        "DECLARED_EMERGENCY_GENERAL",
        "DECLARED_STATUS_CORROBORATED",
    )
    assert bool(result.loc[0, "declared_status_active"]) is True


def test_only_two_fresh_conflicting_critical_states_are_contradictory():
    rows = [
        _base_row(squawk="7500", emergency="nordo"),
        _base_row(squawk="7700", emergency="lifeguard", timestamp_utc=1010.0),
    ]
    frame = _with_fresh_metadata(pd.DataFrame(rows))

    result = classify_s2_rows(frame)

    assert result.loc[0, "declared_status_consistency"] == "contradictory"
    assert "DECLARED_STATUS_CONTRADICTORY" in result.loc[0, "declared_status_reason_codes"]
    # Lifeguard is a separate declaration, not an explicitly conflicting
    # member of the three critical squawk/emergency mappings.
    assert result.loc[1, "declared_status_consistency"] == "not_corroborated"
    assert "DECLARED_EMERGENCY_LIFEGUARD" in result.loc[1, "declared_status_reason_codes"]


@pytest.mark.parametrize("emergency", ["lifeguard", "minfuel", "downed", "reserved"])
def test_standalone_emergency_types_remain_separate_declarations(emergency):
    frame = _with_fresh_metadata(
        pd.DataFrame([_base_row(squawk="1200", emergency=emergency)])
    )

    result = classify_s2_rows(frame)

    assert result.loc[0, "declared_status_emergency_type"] == emergency
    assert f"DECLARED_EMERGENCY_{emergency.upper()}" in result.loc[
        0, "declared_status_reason_codes"
    ]
    assert result.loc[0, "declared_status_consistency"] == "not_corroborated"


def test_stale_or_legacy_freshness_cannot_corroborate():
    stale = _with_fresh_metadata(
        pd.DataFrame([_base_row(squawk="7600", emergency="nordo")]),
        updated=False,
        age_s=61.0,
    )
    stale_result = classify_s2_rows(stale, config=S2Config(freshness_max_age_s=60.0))
    assert stale_result.loc[0, "squawk_freshness"] == "stale"
    assert stale_result.loc[0, "emergency_freshness"] == "stale"
    assert stale_result.loc[0, "declared_status_consistency"] == "not_corroborated"

    # Existing Silver v1 has values but no update columns.  It is never silently
    # promoted to fresh/corroborated.
    legacy = pd.DataFrame([_base_row(squawk="7600", emergency="nordo")])
    legacy_result = classify_s2_rows(legacy)
    assert legacy_result.loc[0, "squawk_freshness"] == "freshness_unknown"
    assert legacy_result.loc[0, "emergency_freshness"] == "freshness_unknown"
    assert legacy_result.loc[0, "declared_status_consistency"] == "not_corroborated"


def test_field_freshness_distinguishes_never_observed_and_invalid_age():
    frame = pd.DataFrame(
        {
            "squawk_updated": [False, False, True],
            "squawk_update_timestamp_utc": [np.nan, 1000.0, np.nan],
            "squawk_update_age_s": [np.nan, -1.0, np.nan],
        }
    )

    states = field_freshness(frame, "squawk", max_age_s=60.0)

    assert states.tolist() == ["never_observed", "invalid_update_metadata", "fresh"]


def test_position_quality_schema_missing_zero_and_scoped_advisory_are_distinct():
    rows = [
        _base_row(nic=6, nac_p=7, sil=2),
        _base_row(nic=6, nac_p=7, sil=2, faa_reference_scope=False, timestamp_utc=1010.0),
        _base_row(nic=12, nac_p=12, sil=4, adsb_version=3, timestamp_utc=1020.0),
        _base_row(nic=0, nac_p=0, sil=0, timestamp_utc=1030.0),
        _base_row(nic=None, nac_p=None, sil=None, timestamp_utc=1040.0),
    ]
    frame = _with_fresh_metadata(pd.DataFrame(rows))

    result = classify_s2_rows(frame)

    assert result.loc[0, "position_quality_scope"] == "eligible"
    assert result.loc[0, "position_quality_nic_status"] == "below_faa_reference"
    assert result.loc[0, "position_quality_nac_p_status"] == "below_faa_reference"
    assert result.loc[0, "position_quality_sil_status"] == "below_faa_reference"
    assert bool(result.loc[0, "position_quality_advisory"]) is True

    assert result.loc[1, "position_quality_scope"] == "outside_asserted_scope"
    assert result.loc[1, "position_quality_nic_status"] == (
        "reported_valid_outside_faa_reference_scope"
    )
    assert bool(result.loc[1, "position_quality_advisory"]) is False

    assert result.loc[2, "position_quality_scope"] == "version_schema_invalid"
    assert result.loc[2, "position_quality_adsb_version_status"] == "schema_invalid"
    assert result.loc[2, "position_quality_nic_status"] == "schema_invalid"
    assert result.loc[2, "position_quality_nac_p_status"] == "schema_invalid"
    assert result.loc[2, "position_quality_sil_status"] == "schema_invalid"
    assert "POSITION_QUALITY_ADSB_VERSION_SCHEMA_INVALID" in result.loc[
        2, "position_quality_reason_codes"
    ]

    for field in ("nic", "nac_p", "sil"):
        assert result.loc[3, f"position_quality_{field}_status"] == (
            "reported_unknown_or_unavailable"
        )
        assert result.loc[4, f"position_quality_{field}_status"] == "missing"


def test_faa_reference_requires_airborne_adsb_version_and_explicit_scope():
    rows = [
        _base_row(nic=6, nac_p=7, sil=2, on_ground=True),
        _base_row(nic=6, nac_p=7, sil=2, ads_source_type="mlat", timestamp_utc=1010.0),
        _base_row(nic=6, nac_p=7, sil=2, adsb_version=1, timestamp_utc=1020.0),
    ]
    without_scope = _base_row(nic=6, nac_p=7, sil=2, timestamp_utc=1030.0)
    without_scope.pop("faa_reference_scope")
    rows.append(without_scope)
    frame = _with_fresh_metadata(pd.DataFrame(rows))

    result = classify_s2_rows(frame)

    assert result["position_quality_scope"].tolist() == [
        "ground_not_applicable",
        "not_adsb_source",
        "version_ineligible",
        "scope_unknown",  # column exists due to other rows, but this value is null
    ]
    assert not result["position_quality_advisory"].any()


def test_altitude_availability_and_message_gap_are_separate_channels():
    rows = [
        _base_row(timestamp_utc=0.0, on_ground=True, alt=None, alt_geom_m=None),
        _base_row(timestamp_utc=10.0, on_ground=False, alt=None, alt_geom_m=1000.0),
        _base_row(timestamp_utc=80.0, on_ground=False, alt=None, alt_geom_m=None),
        _base_row(timestamp_utc=90.0, on_ground=False, alt=900.0, alt_geom_m=None),
    ]
    frame = _with_fresh_metadata(pd.DataFrame(rows))

    result = classify_s2_rows(frame)

    assert result["altitude_availability"].tolist() == [
        "GROUND_ALT_NOT_APPLICABLE",
        "BARO_ALT_DROPOUT",
        "ALL_ALTITUDE_UNAVAILABLE",
        "AVAILABLE",
    ]
    assert result["message_gap"].tolist() == [False, False, True, False]
    assert result.loc[2, "message_gap_s"] == 70.0
    assert result.loc[2, "message_gap_interval_start"] == 10.0
    assert result.loc[2, "message_gap_interval_end"] == 80.0
    assert result.loc[2, "message_gap_reason_code"] == "MESSAGE_GAP"
    assert result.loc[2, "altitude_availability"] == "ALL_ALTITUDE_UNAVAILABLE"


def test_message_gap_state_resets_per_entity_and_marks_nonmonotonic_time():
    rows = [
        _base_row(flight_id="a", timestamp_utc=0.0),
        _base_row(flight_id="b", timestamp_utc=1000.0),
        _base_row(flight_id="a", timestamp_utc=70.0),
        _base_row(flight_id="a", timestamp_utc=60.0),
    ]
    frame = _with_fresh_metadata(pd.DataFrame(rows))

    result = classify_s2_rows(frame, entity_col="flight_id")

    assert result["message_interval_status"].tolist() == [
        "FIRST_OBSERVATION",
        "FIRST_OBSERVATION",
        "MESSAGE_GAP",
        "NON_MONOTONIC_TIMESTAMP",
    ]


def test_reason_rows_can_be_eventized_without_anomaly_or_false_positive_labels():
    rows = [
        _base_row(timestamp_utc=0.0, emergency="general"),
        _base_row(timestamp_utc=30.0, emergency="general"),
        _base_row(timestamp_utc=100.0, emergency="general"),
    ]
    frame = _with_fresh_metadata(pd.DataFrame(rows))
    classified = classify_s2_rows(frame)
    long = explode_s2_reasons(classified)
    emergency = long.loc[long["reason_code"] == "DECLARED_EMERGENCY_GENERAL"]

    episodes = reason_episodes(emergency, entity_col="flight_id", merge_gap_s=60.0)

    assert len(emergency) == 3
    assert episodes[["episode_start", "episode_end", "n_rows"]].to_dict("records") == [
        {"episode_start": 0.0, "episode_end": 100.0, "n_rows": 3},
    ]
    assert not {"anomaly", "false_positive", "label"}.intersection(long.columns)


def test_reason_episodes_split_on_inactive_row_not_sparse_cadence():
    rows = [
        _base_row(timestamp_utc=0.0, emergency="general"),
        _base_row(timestamp_utc=10.0, emergency=None),
        _base_row(timestamp_utc=20.0, emergency="general"),
    ]
    classified = classify_s2_rows(_with_fresh_metadata(pd.DataFrame(rows)))
    emergency = explode_s2_reasons(classified).loc[
        lambda frame: frame["reason_code"] == "DECLARED_EMERGENCY_GENERAL"
    ]

    episodes = reason_episodes(emergency, entity_col="flight_id", merge_gap_s=10_000.0)

    assert episodes[["episode_start", "episode_end", "n_rows"]].to_dict("records") == [
        {"episode_start": 0.0, "episode_end": 0.0, "n_rows": 1},
        {"episode_start": 20.0, "episode_end": 20.0, "n_rows": 1},
    ]


def test_reason_episodes_treat_each_message_gap_row_as_a_point_event():
    rows = [
        _base_row(timestamp_utc=0.0),
        _base_row(timestamp_utc=100.0),
        _base_row(timestamp_utc=200.0),
    ]
    classified = classify_s2_rows(_with_fresh_metadata(pd.DataFrame(rows)))
    gaps = explode_s2_reasons(classified).loc[
        lambda frame: frame["reason_code"] == "MESSAGE_GAP"
    ]

    episodes = reason_episodes(gaps, entity_col="flight_id", merge_gap_s=10_000.0)

    assert episodes[["episode_start", "episode_end", "n_rows"]].to_dict("records") == [
        {"episode_start": 100.0, "episode_end": 100.0, "n_rows": 1},
        {"episode_start": 200.0, "episode_end": 200.0, "n_rows": 1},
    ]


def test_reason_episodes_require_original_row_positions():
    reason_rows = pd.DataFrame(
        {
            "flight_id": ["f1"],
            "timestamp_utc": [0.0],
            "channel": ["declared_status"],
            "reason_code": ["DECLARED_EMERGENCY_GENERAL"],
        }
    )

    with pytest.raises(KeyError, match="row_position"):
        reason_episodes(reason_rows, entity_col="flight_id")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"freshness_max_age_s": -1.0},
        {"message_gap_threshold_s": 0.0},
        {"faa_reference_scope_col": ""},
        {"faa_eligible_versions": ()},
    ],
)
def test_invalid_s2_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        S2Config(**kwargs)
