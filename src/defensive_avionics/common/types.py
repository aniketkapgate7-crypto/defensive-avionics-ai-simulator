"""Typed contracts exchanged between modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Urgency = Literal["low", "approaching", "critical"]
Status = Literal["stable", "caution", "critical"]


@dataclass(frozen=True, slots=True)
class SignalPrediction:
    label: str
    confidence: float
    snr_db: float | None = None


@dataclass(frozen=True, slots=True)
class VisionPrediction:
    detected: bool
    confidence: float
    urgency: Urgency


@dataclass(frozen=True, slots=True)
class PolicyPrediction:
    action_name: str
    confidence: float


@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    frame_id: int
    status: Status
    signal_label: str
    signal_confidence: float
    vision_urgency: Urgency
    policy_action: str
