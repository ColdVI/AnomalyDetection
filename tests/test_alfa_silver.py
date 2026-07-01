import pandas as pd
import pytest

from src.common.minio_io import write_bronze
from src.common.provenance import add_provenance
from src.processing.alfa_silver import build_alfa_silver, build_scenario_table, normalize_fault_label


# Raw folder-suffix -> (fault_type, emergency_traj), taken verbatim from the 47 real ALFA
# processed/processed/ scenario folders enumerated on 2026-07-01 (see alfa_silver.py docstring).
@pytest.mark.parametrize(
    "raw_label,expected_type,expected_emr",
    [
        ("no_ground_truth", "no_ground_truth", False),
        ("1_no_failure", "no_failure", False),
        ("3_no_failure", "no_failure", False),
        ("1_engine_failure", "engine_failure", False),
        ("2_engine_failure", "engine_failure", False),
        ("engine_failure", "engine_failure", False),
        ("engine_failure_with_emr_traj", "engine_failure", True),
        ("3_engine_failure_with_emr_traj", "engine_failure", True),
        ("elevator_failure", "elevator_failure", False),
        ("1_elevator_failure", "elevator_failure", False),
        ("1_rudder_right_failure", "rudder_right_failure", False),
        ("2_rudder_right_failure", "rudder_right_failure", False),
        ("3_rudder_left_failure", "rudder_left_failure", False),
        ("1_right_aileron_failure", "right_aileron_failure", False),
        ("2_right_aileron_failure_with_emr_traj", "right_aileron_failure", True),
        ("2_left_aileron_failure", "left_aileron_failure", False),
        ("3_left_aileron_failure", "left_aileron_failure", False),
        ("2_both_ailerons_failure", "both_aileron_failure", False),
        ("left_aileron__right_aileron__failure", "both_aileron_failure", False),
        ("1_rudder_zero__left_aileron_failure", "rudder_aileron_failure", False),
    ],
)
def test_normalize_fault_label_matches_verified_alfa_folder_suffixes(raw_label, expected_type, expected_emr):
    fault_type, emergency_traj = normalize_fault_label(raw_label)
    assert fault_type == expected_type
    assert emergency_traj is expected_emr


def test_normalize_fault_label_heuristic_handles_unseen_rudder_right_variant(caplog):
    fault_type, emergency_traj = normalize_fault_label("rudder_right_stuck_hard")
    assert fault_type == "rudder_right_failure"
    assert emergency_traj is False
    assert "Unrecognised ALFA failure label" in caplog.text


def test_normalize_fault_label_returns_unknown_for_unrecognisable_suffix():
    fault_type, _ = normalize_fault_label("totally_new_sensor_glitch")
    assert fault_type == "unknown"


def test_build_scenario_table_merges_onto_backbone_and_drops_ros_boilerplate():
    rpy = pd.DataFrame({
        "%time": [1_000_000_000, 1_020_000_000, 1_040_000_000],
        "field.x": [0.1, 0.2, 0.3],
        "field.y": [1.1, 1.2, 1.3],
        "field.z": [2.1, 2.2, 2.3],
    })
    roll = pd.DataFrame({
        "%time": [1_000_000_000, 1_040_000_000],
        "field.header.seq": [0, 1],
        "field.header.stamp": [1_000_000_000, 1_040_000_000],
        "field.header.frame_id": ["NED", "NED"],
        "field.commanded": [10.0, 30.0],
        "field.measured": [9.5, 29.5],
    })

    table = build_scenario_table(
        "carbonZ_test_no_failure",
        {"mavctrl-rpy": rpy, "mavros-nav_info-roll": roll},
        "no_failure",
    )

    assert table is not None
    assert len(table) == len(rpy)
    assert "mavctrl-rpy__field.x" in table.columns
    assert "mavros-nav_info-roll__field.commanded" in table.columns
    assert "mavros-nav_info-roll__field.measured" in table.columns
    assert not any(c.endswith(("header.seq", "header.stamp", "header.frame_id")) for c in table.columns)
    assert (table["fault_type"] == "no_failure").all()
    assert not table["is_fault"].any()
    assert (table["sequence_id"] == "carbonZ_test_no_failure").all()


def test_build_scenario_table_marks_is_fault_from_failure_status_onset():
    rpy = pd.DataFrame({
        "%time": [0, 10_000_000, 20_000_000, 30_000_000],
        "field.x": [0.0, 0.0, 0.0, 0.0],
        "field.y": [0.0, 0.0, 0.0, 0.0],
        "field.z": [0.0, 0.0, 0.0, 0.0],
    })
    failure = pd.DataFrame({"%time": [15_000_000, 20_000_000, 30_000_000], "field.data": [1, 1, 1]})

    table = build_scenario_table(
        "carbonZ_test_1_engine_failure",
        {"mavctrl-rpy": rpy, "failure_status-engines": failure},
        "1_engine_failure",
    )

    assert table["fault_type"].iloc[0] == "engine_failure"
    assert table.sort_values("ts_ns")["is_fault"].tolist() == [False, False, True, True]


def test_build_scenario_table_no_failure_scenario_has_no_fault_rows_even_without_failure_topic():
    rpy = pd.DataFrame({"%time": [0, 1], "field.x": [0.0, 0.0], "field.y": [0.0, 0.0], "field.z": [0.0, 0.0]})

    table = build_scenario_table("carbonZ_test_no_failure", {"mavctrl-rpy": rpy}, "no_failure")

    assert not table["is_fault"].any()


def _write_alfa_bronze_topic(df, *, scenario, topic, failure_label, client):
    tagged = add_provenance(df, source_type="alfa", source_file=f"{scenario}/{scenario}-{topic}.csv")
    tagged["_alfa_scenario"] = scenario
    tagged["_alfa_topic"] = topic
    tagged["_alfa_failure_label"] = failure_label
    write_bronze(tagged, "alfa", client=client)


def test_build_alfa_silver_end_to_end_via_fake_bronze(fake_minio_client):
    scenarios = [
        ("carbonZ_2020-01-01-00-00-00_no_failure", "no_failure"),
        ("carbonZ_2020-01-01-00-05-00_1_engine_failure", "1_engine_failure"),
    ]
    for scenario, label in scenarios:
        rpy = pd.DataFrame({
            "%time": [0, 10_000_000, 20_000_000],
            "field.x": [0.0, 0.1, 0.2],
            "field.y": [1.0, 1.1, 1.2],
            "field.z": [2.0, 2.1, 2.2],
        })
        _write_alfa_bronze_topic(rpy, scenario=scenario, topic="mavctrl-rpy", failure_label=label, client=fake_minio_client)
        if label != "no_failure":
            failure = pd.DataFrame({"%time": [10_000_000], "field.data": [1]})
            _write_alfa_bronze_topic(
                failure, scenario=scenario, topic="failure_status-engines", failure_label=label, client=fake_minio_client
            )

    silver = build_alfa_silver(fake_minio_client)

    assert set(silver["sequence_id"].unique()) == {s for s, _ in scenarios}
    assert set(silver["fault_type"].unique()) == {"no_failure", "engine_failure"}

    fault_rows = silver[silver["sequence_id"] == "carbonZ_2020-01-01-00-05-00_1_engine_failure"]
    assert fault_rows.sort_values("ts_ns")["is_fault"].tolist() == [False, True, True]


def test_build_alfa_silver_returns_empty_dataframe_when_bronze_is_empty(fake_minio_client):
    result = build_alfa_silver(fake_minio_client)

    assert isinstance(result, pd.DataFrame)
    assert result.empty
