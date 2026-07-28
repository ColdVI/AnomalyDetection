import pandas as pd
import pytest

from scripts.build_four_dataset_probabilistic_v3_group_registry import (
    GroupRegistryContractError,
    _annotate_group_audit,
    alfa_group,
    uav_attack_group,
    uav_sead_group,
)


def test_alfa_group_keeps_normal_and_fault_parts_in_one_session():
    normal = alfa_group("carbonZ_2018-10-05-14-34-20_1_no_failure")
    fault = alfa_group(
        "carbonZ_2018-10-05-14-34-20_2_right_aileron_failure_with_emr_traj"
    )
    assert normal[0] == fault[0] == "alfa:carbonZ_2018-10-05-14-34-20"
    with pytest.raises(GroupRegistryContractError):
        alfa_group("unparseable-session")


def test_uav_attack_group_combines_archive_mode_platform_and_campaign_date():
    assert uav_attack_group(
        "log_2_2020-8-2-19-54-17", "simulated", "PX4-PLANE-SITL"
    )[0] == (
        "uav_attack:simulated:PX4-PLANE-SITL:2020-08-02"
    )
    assert uav_attack_group(
        "ace-jamming-log_1_2033-8-19-16-46-46",
        "live",
        "live_platform_unspecified",
    )[0] == (
        "uav_attack:live:live_platform_unspecified:2033-08-19"
    )
    assert uav_attack_group(
        "001-2021-01-27-12-34-48-014", "simulated", "PX4-QUAD-SITL"
    )[0] == (
        "uav_attack:simulated:PX4-QUAD-SITL:2021-01-27"
    )


def test_uav_sead_parent_proxy_does_not_merge_unrelated_root_sources():
    assert uav_sead_group("2023-08/sess039/log001")[0] == (
        "uav_sead:2023-08/sess039"
    )
    first = uav_sead_group("00_14_08")[0]
    second = uav_sead_group("00_16_12")[0]
    assert first != second
    assert first == "uav_sead:root_source:00_14_08"


def test_group_audit_marks_only_groups_spanning_v2_roles():
    frame = pd.DataFrame(
        {
            "dataset": ["alfa", "alfa", "alfa"],
            "group_id": ["g1", "g1", "g2"],
            "source_id": ["a", "b", "c"],
            "v2_role": ["train", "test", "train"],
        }
    )
    audited = _annotate_group_audit(frame).set_index("source_id")
    assert bool(audited.loc["a", "v2_cross_role_group"])
    assert audited.loc["a", "group_v2_roles"] == "train|test"
    assert audited.loc["a", "group_source_count"] == 2
    assert not bool(audited.loc["c", "v2_cross_role_group"])
