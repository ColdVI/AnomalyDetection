from __future__ import annotations

from src.adsb_behavioral.hard_rules import add_hard_rule_score
from src.adsb_behavioral.physics_residuals import add_physics_residuals
from tests.adsb_behavioral.test_physics_residuals import _straight_flight


def test_ten_thousand_foot_jump_with_zero_reported_vrate_is_detected_immediately():
    flight = _straight_flight(30)
    flight["vertical_rate_ms"] = 0.0
    jump_index = 15
    flight.loc[jump_index:, "alt"] += 10_000.0 * 0.3048
    scored = add_hard_rule_score(add_physics_residuals(flight))

    assert bool(scored.loc[jump_index, "hard_rule_violation"]) is True
    assert scored.loc[jump_index, "hard_rule_reason"] == "altitude_vertical_rate_mismatch"
    assert scored.loc[jump_index, "hard_rule_score"] > 10.0


def test_consistent_straight_flight_does_not_trigger_hard_rule():
    scored = add_hard_rule_score(add_physics_residuals(_straight_flight(30)))
    assert not scored["hard_rule_violation"].fillna(False).any()
