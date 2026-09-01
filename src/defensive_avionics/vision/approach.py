"""Estimate generic object-approach trends from bounding-box expansion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

VisualTrend = Literal[
    "insufficient_data",
    "receding",
    "stable",
    "growing",
    "rapid_growth",
]


@dataclass(frozen=True, slots=True)
class ApproachEstimate:
    """A normalized visual trend, not a physical distance estimate."""

    trend: VisualTrend
    relative_growth: float
    sample_count: int


class ExpansionTracker:
    """Track changes in normalized bounding-box area over time with smoothing and hysteresis."""

    def __init__(
        self,
        window_size: int = 16,
        minimum_samples: int = 4,
        min_duration_sec: float = 0.0,
        stable_tolerance: float = 0.03,
        rapid_growth_threshold: float = 0.12,
        hysteresis_count: int = 1,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")

        if not 2 <= minimum_samples <= window_size:
            raise ValueError("minimum_samples must be between 2 and window_size")

        if stable_tolerance < 0:
            raise ValueError("stable_tolerance cannot be negative")

        if rapid_growth_threshold <= stable_tolerance:
            raise ValueError("rapid_growth_threshold must exceed stable_tolerance")

        self.minimum_samples = minimum_samples
        self.min_duration_sec = min_duration_sec
        self.stable_tolerance = stable_tolerance
        self.rapid_growth_threshold = rapid_growth_threshold
        self.hysteresis_count = max(1, hysteresis_count)

        self._samples: deque[tuple[float, float | None]] = deque(maxlen=window_size)
        self._current_trend: VisualTrend = "insufficient_data"
        self._candidate_trend: VisualTrend = "insufficient_data"
        self._candidate_hits: int = 0

    def reset(self) -> None:
        """Clear the stored observation history and reset hysteresis."""
        self._samples.clear()
        self._current_trend = "insufficient_data"
        self._candidate_trend = "insufficient_data"
        self._candidate_hits = 0

    def update(
        self,
        area_ratio: float | None,
        timestamp: float | None = None,
    ) -> ApproachEstimate:
        """Add one normalized area measurement and return its smoothed trend."""
        if area_ratio is None:
            self.reset()
            return ApproachEstimate(
                trend="insufficient_data",
                relative_growth=0.0,
                sample_count=0,
            )

        if not 0.0 < area_ratio <= 1.0:
            raise ValueError("area_ratio must be between 0 and 1")

        self._samples.append((float(area_ratio), timestamp))
        sample_count = len(self._samples)

        if sample_count < self.minimum_samples:
            self._current_trend = "insufficient_data"
            self._candidate_trend = "insufficient_data"
            self._candidate_hits = 0
            return ApproachEstimate(
                trend="insufficient_data",
                relative_growth=0.0,
                sample_count=sample_count,
            )

        first_area, first_time = self._samples[0]
        last_area, last_time = self._samples[-1]

        # Check duration if timestamps are tracked and required
        if (
            self.min_duration_sec > 0.0
            and first_time is not None
            and last_time is not None
            and (last_time - first_time) < self.min_duration_sec
        ):
            return ApproachEstimate(
                trend="insufficient_data",
                relative_growth=0.0,
                sample_count=sample_count,
            )

        # Smooth boundary values using 3-sample average to reduce single-frame box jitter
        raw_areas = [s[0] for s in self._samples]
        smoothed_first = sum(raw_areas[: min(3, sample_count)]) / min(3, sample_count)
        smoothed_last = sum(raw_areas[-min(3, sample_count) :]) / min(3, sample_count)

        if first_time is not None and last_time is not None and (last_time - first_time) > 0.05:
            elapsed = last_time - first_time
            relative_growth = (smoothed_last - smoothed_first) / max(smoothed_first, 1e-9) / elapsed
        else:
            step_count = sample_count - 1
            relative_growth = (
                (smoothed_last - smoothed_first) / max(smoothed_first, 1e-9) / max(step_count, 1)
            )

        # Determine instantaneous candidate trend
        if relative_growth < -self.stable_tolerance:
            candidate: VisualTrend = "receding"
        elif relative_growth <= self.stable_tolerance:
            candidate = "stable"
        elif relative_growth < self.rapid_growth_threshold:
            candidate = "growing"
        else:
            candidate = "rapid_growth"

        # Apply hysteresis
        if self.hysteresis_count <= 1:
            self._current_trend = candidate
        else:
            if candidate == self._candidate_trend:
                self._candidate_hits += 1
            else:
                self._candidate_trend = candidate
                self._candidate_hits = 1

            if self._candidate_hits >= self.hysteresis_count:
                self._current_trend = candidate

        return ApproachEstimate(
            trend=self._current_trend,
            relative_growth=relative_growth,
            sample_count=sample_count,
        )
