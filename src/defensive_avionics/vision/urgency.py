"""Relative approach-urgency estimate from recent box areas."""

from __future__ import annotations

from collections.abc import Sequence


def estimate_urgency(box_areas: Sequence[float]) -> str:
    """Return low/approaching/critical; this is not a distance estimate."""

    if len(box_areas) < 2:
        return "low"
    if any(area < 0 for area in box_areas):
        raise ValueError("Bounding-box areas cannot be negative.")

    first = max(float(box_areas[0]), 1.0)
    growth_ratio = float(box_areas[-1]) / first
    if growth_ratio >= 2.0:
        return "critical"
    if growth_ratio >= 1.25:
        return "approaching"
    return "low"
