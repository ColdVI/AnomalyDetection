from pathlib import Path

import numpy as np
import pytest

from gecmis_calismalar.rfly_full.pipeline import _k_of_n, _package_priority, _safe_target, case_id, is_essential


def test_case_id_supports_simulation_and_real_paths():
    assert case_id("SIL-Motor/hover/TestCase_1_2/Log/a.ulg") == "SIL-Motor/hover/TestCase_1_2"
    assert case_id("Real-Motor/hover/1_1/log_2/log_2.ulg") == "Real-Motor/hover/1_1/log_2"


def test_only_deployment_log_and_testinfo_are_essential():
    assert is_essential("SIL-Motor/a/Log/x.ulg")
    assert is_essential("SIL-Motor/a/TestInfo.csv")
    assert is_essential("Real-Motor/a/TestInfo.xlsx")
    assert is_essential("Real-Motor/a/TestInfo_2023-06-02.xlsx")
    assert not is_essential("SIL-Motor/a/TrueData/UAVState.xlsx")
    assert not is_essential("SIL-Motor/a/TLog/x.tlog")


def test_safe_target_rejects_traversal():
    with pytest.raises(ValueError):
        _safe_target(Path("data"), "../escape.ulg")


def test_four_of_six_emits_one_alarm_onset():
    result = _k_of_n(np.array([0, 1, 1, 0, 1, 1, 1, 1], dtype=bool))
    assert np.flatnonzero(result).tolist() == [5]


def test_nofault_is_downloaded_before_fault_packages():
    names = ["HIL-Motor", "SIL-Sensors", "HIL-NoFault", "SIL-NoFault"]
    assert sorted(names, key=_package_priority)[:2] == ["SIL-NoFault", "HIL-NoFault"]
