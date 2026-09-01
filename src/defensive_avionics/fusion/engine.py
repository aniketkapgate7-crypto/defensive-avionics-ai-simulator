"""Confidence-weighted sensor fusion engine with temporal freshness decay."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

UrgencyLevel = Literal["low", "approaching", "critical"]


@dataclass(frozen=True, slots=True)
class Observation:
    """One typed sensor observation from a local or collaborative source."""

    source_id: str
    source_type: str
    label: str
    confidence: float
    uncertainty: float = 0.0
    relative_urgency: UrgencyLevel = "low"
    timestamp: float = 0.0
    information_age: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(f"uncertainty must be in [0.0, 1.0], got {self.uncertainty}")
        if self.information_age < 0.0:
            raise ValueError(f"information_age cannot be negative, got {self.information_age}")
        if self.relative_urgency not in {"low", "approaching", "critical"}:
            raise ValueError(f"Invalid urgency level: {self.relative_urgency}")


@dataclass(frozen=True, slots=True)
class FusedState:
    """Fused consensus state across multi-modal sensor observations."""

    fused_label: str
    fused_confidence: float
    fused_uncertainty: float
    relative_urgency: UrgencyLevel
    contributing_sources_count: int
    information_freshness: float
    explanation: str
    evidence_breakdown: dict[str, float] = field(default_factory=dict)


class SensorFusionEngine:
    """Combines signal, vision, scenario, and collaborative sensor inputs.

    Uses an explainable confidence-weighted consensus with exponential
    temporal freshness decay.
    """

    def __init__(self, half_life_seconds: float = 5.0) -> None:
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive.")
        self.decay_rate = math.log(2.0) / half_life_seconds

    def fuse(self, observations: list[Observation]) -> FusedState:
        """Fuse a list of sensor observations into a single explainable state."""
        if not observations:
            return FusedState(
                fused_label="unknown",
                fused_confidence=0.0,
                fused_uncertainty=1.0,
                relative_urgency="low",
                contributing_sources_count=0,
                information_freshness=1.0,
                explanation="No active observations available.",
                evidence_breakdown={},
            )

        # Validate all inputs
        for obs in observations:
            if not isinstance(obs, Observation):
                raise TypeError(f"Expected Observation instance, got {type(obs)}")

        # Single observation fast-path
        if len(observations) == 1:
            obs = observations[0]
            freshness = math.exp(-self.decay_rate * obs.information_age)
            effective_conf = obs.confidence * freshness
            eff_uncertainty = max(0.0, min(1.0, 1.0 - effective_conf + (obs.uncertainty * 0.5)))
            return FusedState(
                fused_label=obs.label,
                fused_confidence=round(effective_conf, 4),
                fused_uncertainty=round(eff_uncertainty, 4),
                relative_urgency=obs.relative_urgency,
                contributing_sources_count=1,
                information_freshness=round(freshness, 4),
                explanation=(
                    f"Single source '{obs.source_id}' ({obs.source_type}) "
                    f"reporting '{obs.label}' with {obs.confidence:.0%} confidence."
                ),
                evidence_breakdown={obs.label: round(effective_conf, 4)},
            )

        # Calculate effective weight for each observation
        # weight = confidence * freshness * (1 - uncertainty)
        weights: list[float] = []
        freshnesses: list[float] = []
        label_weighted_scores: dict[str, float] = {}
        urgency_counts: dict[UrgencyLevel, float] = {
            "low": 0.0,
            "approaching": 0.0,
            "critical": 0.0,
        }

        for obs in observations:
            freshness = math.exp(-self.decay_rate * obs.information_age)
            freshnesses.append(freshness)
            # Effective weight combines confidence, certainty, and freshness
            certainty = max(0.01, 1.0 - obs.uncertainty)
            weight = max(0.001, obs.confidence * freshness * certainty)
            weights.append(weight)

            label_weighted_scores[obs.label] = label_weighted_scores.get(obs.label, 0.0) + weight
            urgency_counts[obs.relative_urgency] += weight

        total_weight = sum(weights)
        mean_freshness = sum(freshnesses) / len(freshnesses)

        # Determine dominant label and confidence
        best_label = max(label_weighted_scores, key=lambda k: label_weighted_scores[k])
        best_score = label_weighted_scores[best_label]
        label_proportion = best_score / total_weight

        # Fused confidence balances agreement and raw source confidences
        avg_confidence = (
            sum(obs.confidence * w for obs, w in zip(observations, weights, strict=True))
            / total_weight
        )
        agreement_factor = label_proportion
        fused_conf = max(0.0, min(1.0, avg_confidence * agreement_factor * mean_freshness))

        # Fused uncertainty increases when sources disagree or are uncertain/stale
        disagreement = 1.0 - label_proportion
        fused_unc = max(0.0, min(1.0, (1.0 - fused_conf) * 0.6 + disagreement * 0.4))

        # Highest weighted urgency
        dominant_urgency: UrgencyLevel = "low"
        if urgency_counts["critical"] > 0 and (urgency_counts["critical"] / total_weight) >= 0.25:
            dominant_urgency = "critical"
        elif (
            urgency_counts["approaching"] > 0
            and ((urgency_counts["approaching"] + urgency_counts["critical"]) / total_weight)
            >= 0.35
        ):
            dominant_urgency = "approaching"
        else:
            dominant_urgency = max(urgency_counts, key=lambda k: urgency_counts[k])

        # Evidence breakdown
        breakdown = {
            lbl: round(score / total_weight, 4)
            for lbl, score in sorted(label_weighted_scores.items(), key=lambda x: -x[1])
        }

        # Human-readable explanation
        top_source = max(observations, key=lambda o: o.confidence)
        explanation = (
            f"Consensus '{best_label}' across {len(observations)} sources "
            f"({label_proportion:.0%} agreement, primary source: {top_source.source_id})."
        )

        return FusedState(
            fused_label=best_label,
            fused_confidence=round(fused_conf, 4),
            fused_uncertainty=round(fused_unc, 4),
            relative_urgency=dominant_urgency,
            contributing_sources_count=len(observations),
            information_freshness=round(mean_freshness, 4),
            explanation=explanation,
            evidence_breakdown=breakdown,
        )
