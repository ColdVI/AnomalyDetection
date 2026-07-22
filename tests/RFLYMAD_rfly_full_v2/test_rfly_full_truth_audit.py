import pandas as pd

from gecmis_calismalar.rfly_full.truth_audit import _audit_flight_frame, _near_duplicate_clusters, _trajectory_fingerprint


def _flight_frame(*, t_rel_s, fault_active, system_fault=True, truth_source="rfly_ctrl_lxl", disagreement=False):
    return pd.DataFrame({
        "t_rel_s": t_rel_s,
        "canonical_case_id": ["rfly_x"] * len(t_rel_s),
        "object_name": ["pkg/case/Log/x.ulg"] * len(t_rel_s),
        "package": ["SIL_Motor_1"] * len(t_rel_s),
        "domain": ["SIL"] * len(t_rel_s),
        "fault_family": ["Motor"] * len(t_rel_s),
        "fault_subtype": ["Motor_1"] * len(t_rel_s),
        "system_fault": [system_fault] * len(t_rel_s),
        "environment_condition": ["None"] * len(t_rel_s),
        "fault_active": fault_active,
        "environment_active": [False] * len(t_rel_s),
        "truth_source": [truth_source] * len(t_rel_s),
        "truth_crosscheck_disagreement": [disagreement] * len(t_rel_s),
    })


def test_audit_flight_frame_extracts_active_interval():
    frame = _flight_frame(t_rel_s=[0.0, 1.0, 2.0, 3.0, 4.0], fault_active=[False, False, True, True, False])
    result = _audit_flight_frame(frame, missing_features=[])
    assert result["fault_start_s"] == 2.0
    assert result["fault_end_s"] == 3.0
    assert result["duration_s"] == 4.0
    assert result["missing_active_interval"] is False
    assert result["interval_violation"] == ""
    assert result["active_from_first_sample"] is False


def test_audit_flight_frame_flags_active_from_first_sample():
    frame = _flight_frame(t_rel_s=[0.0, 1.0, 2.0], fault_active=[True, True, True])
    result = _audit_flight_frame(frame, missing_features=[])
    assert result["active_from_first_sample"] is True


def test_audit_flight_frame_flags_missing_active_interval_for_system_fault():
    frame = _flight_frame(t_rel_s=[0.0, 1.0, 2.0], fault_active=[False, False, False], truth_source="missing")
    result = _audit_flight_frame(frame, missing_features=[])
    assert result["missing_active_interval"] is True
    assert result["fault_start_s"] != result["fault_start_s"]  # NaN


def test_audit_flight_frame_normal_reference_never_flagged_missing():
    frame = _flight_frame(
        t_rel_s=[0.0, 1.0], fault_active=[False, False],
        system_fault=False, truth_source="normal_no_fault",
    )
    result = _audit_flight_frame(frame, missing_features=[])
    assert result["missing_active_interval"] is False


def test_audit_flight_frame_records_schema_gaps():
    frame = _flight_frame(t_rel_s=[0.0, 1.0], fault_active=[False, True])
    result = _audit_flight_frame(frame, missing_features=["imu_accel_mag_rms"])
    assert result["missing_v2_features"] == "imu_accel_mag_rms"


def test_audit_flight_frame_reports_tolerated_crosscheck_v2_shift():
    frame = _flight_frame(
        t_rel_s=[0.0, 1.0, 2.0], fault_active=[False, True, True]
    )
    frame["truth_crosscheck_eligible_v2"] = True
    frame["truth_crosscheck_onset_delta_s"] = 15.0
    frame["truth_crosscheck_offset_delta_s"] = 20.0
    frame["truth_crosscheck_overlap_s"] = 5.0
    frame["truth_crosscheck_disagreement_v2"] = False
    frame["truth_crosscheck_schema_version"] = 2

    result = _audit_flight_frame(frame, missing_features=[])

    assert result["truth_crosscheck_eligible_v2"] is True
    assert result["truth_crosscheck_onset_delta_s"] == 15.0
    assert result["truth_crosscheck_disagreement_v2"] is False


def test_audit_flight_frame_reports_true_crosscheck_v2_disagreement():
    frame = _flight_frame(
        t_rel_s=[0.0, 1.0, 2.0], fault_active=[False, True, True]
    )
    frame["truth_crosscheck_eligible_v2"] = True
    frame["truth_crosscheck_onset_delta_s"] = 30.0
    frame["truth_crosscheck_offset_delta_s"] = 30.0
    frame["truth_crosscheck_overlap_s"] = 0.0
    frame["truth_crosscheck_disagreement_v2"] = True
    frame["truth_crosscheck_schema_version"] = 2

    result = _audit_flight_frame(frame, missing_features=[])

    assert result["truth_crosscheck_disagreement_v2"] is True
    assert result["truth_crosscheck_overlap_s"] == 0.0


def test_trajectory_fingerprint_differs_for_different_paths():
    same_a = pd.DataFrame({"local_x": [0.0, 1.0, 2.0], "local_y": [0.0, 0.0, 0.0], "local_z": [0.0, 0.0, 0.0]})
    same_b = pd.DataFrame({"local_x": [0.0, 1.0, 2.0], "local_y": [0.0, 0.0, 0.0], "local_z": [0.0, 0.0, 0.0]})
    different = pd.DataFrame({"local_x": [5.0, 6.0, 7.0], "local_y": [1.0, 1.0, 1.0], "local_z": [0.0, 0.0, 0.0]})
    assert _trajectory_fingerprint(same_a) == _trajectory_fingerprint(same_b)
    assert _trajectory_fingerprint(same_a) != _trajectory_fingerprint(different)


def test_trajectory_fingerprint_handles_missing_position_columns():
    assert _trajectory_fingerprint(pd.DataFrame({"t_rel_s": [0.0, 1.0]})) == "no_position_data"


def test_near_duplicate_clusters_trajectory_tier_flags_cross_split_matches():
    frame = pd.DataFrame({
        "domain": ["SIL", "SIL", "HIL"],
        "fault_family": ["Motor", "Motor", "Motor"],
        "fault_subtype": ["Motor_1", "Motor_1", "Motor_1"],
        "duration_s": [30.0, 30.0, 30.0],
        "row_count": [300, 300, 300],
        "canonical_case_id": ["a", "b", "c"],
        "split_group_id": ["group_1", "group_2", "group_3"],
        "split": ["development", "locked_test", "development"],
        "trajectory_fingerprint": ["same_path", "same_path", "same_path"],
    })
    clusters = _near_duplicate_clusters(frame)
    trajectory = clusters.loc[clusters["tier"].eq("trajectory_signature")]
    sil_cluster = trajectory.loc[trajectory["signature"].str.startswith("SIL")]
    assert len(sil_cluster) == 1
    assert bool(sil_cluster.iloc[0]["spans_multiple_split_groups"])
    assert bool(sil_cluster.iloc[0]["spans_locked_and_development"])


def test_near_duplicate_clusters_duration_tier_overclusters_but_trajectory_tier_does_not():
    """Same duration/row-count but different physical paths: expected for
    standardized SIL/HIL batch protocols (observed on the real dataset),
    should not be reported as a leakage risk by the trajectory tier."""
    frame = pd.DataFrame({
        "domain": ["SIL", "SIL"],
        "fault_family": ["Sensor", "Sensor"],
        "fault_subtype": ["Accelerometer", "Accelerometer"],
        "duration_s": [101.2, 101.2],
        "row_count": [1013, 1013],
        "canonical_case_id": ["a", "b"],
        "split_group_id": ["group_1", "group_2"],
        "split": ["locked_test", "development"],
        "trajectory_fingerprint": ["path_one", "path_two"],
    })
    clusters = _near_duplicate_clusters(frame)
    assert len(clusters.loc[clusters["tier"].eq("duration_signature")]) == 1
    assert clusters.loc[clusters["tier"].eq("trajectory_signature")].empty


def test_near_duplicate_clusters_empty_when_all_unique():
    frame = pd.DataFrame({
        "domain": ["SIL", "HIL"],
        "fault_family": ["Motor", "Sensor"],
        "fault_subtype": ["Motor_1", "GPS"],
        "duration_s": [30.0, 45.0],
        "row_count": [300, 450],
        "canonical_case_id": ["a", "b"],
        "split_group_id": ["group_1", "group_2"],
        "split": ["development", "development"],
        "trajectory_fingerprint": ["path_one", "path_two"],
    })
    clusters = _near_duplicate_clusters(frame)
    assert clusters.empty
