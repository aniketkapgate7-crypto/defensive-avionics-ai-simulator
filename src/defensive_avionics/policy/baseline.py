"""Transparent baseline for the abstract academic environment."""

from __future__ import annotations

from collections.abc import Sequence


def choose_action(
    observation: Sequence[float],
) -> int:
    if len(observation) != 7:
        raise ValueError("Expected seven normalized observations.")

    (
        intensity,
        uncertainty,
        _change_rate,
        _signal_confidence,
        visual_urgency,
        resource_a,
        resource_b,
    ) = map(float, observation)

    if max(intensity, visual_urgency) > 0.75:
        return 4

    if visual_urgency > 0.50 and resource_b >= 0.08:
        return 2

    if intensity > 0.50 and resource_a >= 0.08:
        return 1

    if uncertainty > 0.50:
        return 3

    return 0
