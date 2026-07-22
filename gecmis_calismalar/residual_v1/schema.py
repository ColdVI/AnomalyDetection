"""Frozen raw-channel schema for RESIDUAL-V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

ChannelRole: TypeAlias = Literal["response", "command", "context"]


@dataclass(frozen=True)
class ChannelSpec:
    """Physical contract for one natural-rate telemetry channel."""

    name: str
    topic: str
    unit: str
    valid_min: float
    valid_max: float
    nominal_hz: float
    is_angle: bool = False
    role: ChannelRole = "context"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.topic.strip() or not self.unit.strip():
            raise ValueError("name, topic, and unit must be non-empty")
        if not self.valid_min < self.valid_max:
            raise ValueError("valid_min must be smaller than valid_max")
        if self.nominal_hz <= 0:
            raise ValueError("nominal_hz must be positive")
        if self.role not in {"response", "command", "context"}:
            raise ValueError(f"unsupported channel role: {self.role}")

