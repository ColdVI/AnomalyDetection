"""RFLY-1 simulation-only isolation tests."""

from __future__ import annotations

import pytest

from scripts.run_rfly1_simulation_capability import (
    SIMULATION_SUBSETS,
    assert_simulation_only_cases,
    simulation_family_from_case,
)


def test_simulation_subsets_are_separate_from_real_rfly():
    assert SIMULATION_SUBSETS == ("HIL-Wind", "SIL-Wind")
    assert "Real-Motor" not in SIMULATION_SUBSETS
    assert "Real-Sensors" not in SIMULATION_SUBSETS


def test_simulation_family_from_hil_and_sil_layouts():
    assert simulation_family_from_case("HIL-Wind/acce-wind/TestCase_106_2414000105") == "acce-wind"
    assert simulation_family_from_case("SIL-Wind/SIL-Wind/waypoint-wind/TestCase_200_1") == "waypoint-wind"


def test_simulation_track_rejects_real_cases():
    assert_simulation_only_cases([
        "HIL-Wind/acce-wind/TestCase_106_2414000105",
        "SIL-Wind/SIL-Wind/hover-wind/TestCase_120_1",
    ])
    with pytest.raises(AssertionError, match="non-simulation"):
        assert_simulation_only_cases(["Real-Motor/hover/001_1/log_0"])
