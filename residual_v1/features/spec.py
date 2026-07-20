"""Frozen residual-channel definitions and AR-leakage guard."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

_DERIVED_HISTORY_TOKENS = {"lag", "prev", "previous", "history", "tminus", "derivative", "delta"}


def _tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def _contains_response_history(candidate: str, response: str) -> bool:
    candidate_normalised = re.sub(r"[^a-z0-9]+", "_", candidate.lower()).strip("_")
    response_normalised = re.sub(r"[^a-z0-9]+", "_", response.lower()).strip("_")
    if candidate_normalised == response_normalised:
        return True
    if response_normalised not in candidate_normalised:
        return False
    candidate_tokens = set(_tokens(candidate))
    response_tokens = _tokens(response)
    has_response_sequence = "_".join(response_tokens) in candidate_normalised
    has_history_marker = bool(candidate_tokens & _DERIVED_HISTORY_TOKENS) or bool(
        re.search(r"(?:^|_)lag\d*(?:_|$)", candidate_normalised)
    )
    return has_response_sequence and has_history_marker


@dataclass(frozen=True)
class ResidualChannelSpec:
    name: str
    command_inputs: tuple[str, ...]
    response: str
    context_inputs: tuple[str, ...] = ()
    horizon_s: float = 0.5
    lag_summary: str = "tri4"
    boundary_masks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.response.strip():
            raise ValueError("name and response must be non-empty")
        if len(set(self.command_inputs)) != len(self.command_inputs):
            raise ValueError("command_inputs must be distinct")
        if len(set(self.context_inputs)) != len(self.context_inputs):
            raise ValueError("context_inputs must be distinct")
        overlap = sorted(set(self.command_inputs) & set(self.context_inputs))
        if overlap:
            raise ValueError(f"inputs cannot be both command and context: {overlap}")
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be positive")
        if self.lag_summary != "tri4":
            raise ValueError("RESIDUAL-V1 supports only the frozen tri4 lag summary")
        unsupported_masks = sorted(set(self.boundary_masks) - {"waypoint"})
        if unsupported_masks:
            raise ValueError(f"unsupported boundary masks: {unsupported_masks}")
        leaking = [
            feature
            for feature in self.all_inputs
            if _contains_response_history(feature, self.response)
        ]
        if leaking:
            raise ValueError(
                f"response or response history cannot appear in model inputs: {leaking}"
            )

    @property
    def all_inputs(self) -> tuple[str, ...]:
        """All declared predictors, retaining their explicit semantic roles."""

        return (*self.command_inputs, *self.context_inputs)


ALFA_SPECS: tuple[ResidualChannelSpec, ...] = (
    ResidualChannelSpec(
        "R1_aileron_roll_rate",
        ("aileron_cmd",),
        "roll_rate",
        ("airspeed_context",),
    ),
    ResidualChannelSpec(
        "R2_elevator_pitch_rate",
        ("elevator_cmd",),
        "pitch_rate",
        ("airspeed_context",),
    ),
    ResidualChannelSpec(
        "R3_rudder_coordinated_yaw_rate",
        ("rudder_cmd",),
        "yaw_rate",
        ("roll_context", "airspeed_context", "coordinated_turn_term"),
    ),
    ResidualChannelSpec(
        "R4_throttle_airspeed_derivative",
        ("throttle_cmd",),
        "airspeed_derivative",
        ("airspeed_context", "pitch_context"),
    ),
    ResidualChannelSpec(
        "R5_pitch_throttle_climb_rate",
        ("throttle_cmd",),
        "climb_rate",
        ("pitch_context", "airspeed_context"),
    ),
    ResidualChannelSpec(
        "R6_xtrack_error",
        (),
        "xtrack_error",
        boundary_masks=("waypoint",),
    ),
)

RFLY_SPECS: tuple[ResidualChannelSpec, ...] = (
    ResidualChannelSpec(
        "Q1_attitude_setpoint_rate_response",
        ("roll_rate_sp", "pitch_rate_sp", "yaw_rate_sp", "thrust_sp"),
        "attitude_rate_vector_norm",
    ),
    ResidualChannelSpec(
        "Q2_motor_pwm_distribution",
        ("thrust_sp", "roll_sp", "pitch_sp", "yaw_sp"),
        "motor_pwm_asymmetry",
        ("battery_voltage",),
    ),
    ResidualChannelSpec(
        "Q3_total_pwm_vertical_acceleration",
        ("motor_pwm_total",),
        "vertical_acceleration",
        ("battery_voltage",),
    ),
    ResidualChannelSpec(
        "Q4_position_setpoint_velocity_response",
        ("position_sp_x", "position_sp_y", "position_sp_z"),
        "velocity_response_norm",
    ),
)


def descriptor_schema_payload() -> dict:
    def model_descriptor(spec: ResidualChannelSpec) -> dict:
        # Boundary masks change row eligibility, not the G1 model descriptor.
        # They are versioned separately with their frozen config in feature runs.
        payload = asdict(spec)
        payload.pop("boundary_masks")
        return payload

    return {
        "schema": "descriptor_schema_residual_v1",
        "alfa": [model_descriptor(spec) for spec in ALFA_SPECS],
        "rfly": [model_descriptor(spec) for spec in RFLY_SPECS],
    }


def descriptor_schema_sha256() -> str:
    encoded = json.dumps(
        descriptor_schema_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
