import numpy as np

from gecmis_calismalar.rfly_full.contract import domain_of, taxonomy, window_phase


def test_wind_is_environment_not_system_fault():
    result = taxonomy("SIL-Wind", "SIL-Wind/hover/TestCase_1")
    assert result["fault_family"] == "Environment"
    assert result["environment_condition"] == "Wind"
    assert result["system_fault"] is False
    assert result["evaluation_role"] == "environment_robustness"


def test_fault_taxonomy_retains_subtype():
    result = taxonomy(
        "HIL-Sensors", "HIL-Sensors/hover-gyroscope/TestCase_1"
    )
    assert result["fault_family"] == "Sensor"
    assert result["fault_subtype"] == "Gyroscope"
    assert result["system_fault"] is True
    assert domain_of("HIL-Sensors") == "HIL"


def test_transition_window_is_not_forced_to_normal_or_fault():
    assert window_phase(np.array([False, False]), is_system_fault=True) == "pre_or_post_fault"
    assert window_phase(np.array([True, True]), is_system_fault=True) == "fault_active"
    assert window_phase(np.array([False, True]), is_system_fault=True) == "transition"
