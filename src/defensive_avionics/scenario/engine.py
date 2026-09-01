"""Deterministic, offline synthetic scenario engine with collaborative nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from defensive_avionics.fusion.engine import Observation, UrgencyLevel

ObjectCategory = Literal["friendly", "neutral", "unknown", "resource"]
DifficultyPreset = Literal["low", "medium", "high"]


@dataclass
class SyntheticObject:
    """One generic synthetic aerial object in normalized classroom coordinates."""

    id: str
    category: ObjectCategory
    x: float  # Normalized [0.0, 1.0], ownship centered at (0.5, 0.5)
    y: float  # Normalized [0.0, 1.0]
    heading_deg: float  # [0.0, 360.0)
    speed: float  # Normalized velocity per step
    confidence: float
    urgency: UrgencyLevel
    source: str
    timestamp: float = 0.0

    @property
    def distance_from_ownship(self) -> float:
        """Euclidean distance in normalized coordinate space from (0.5, 0.5)."""
        dx = self.x - 0.5
        dy = self.y - 0.5
        return math.hypot(dx, dy)

    @property
    def bearing_deg(self) -> float:
        """Bearing from ownship in degrees [0, 360)."""
        dx = self.x - 0.5
        dy = self.y - 0.5
        angle = math.degrees(math.atan2(dx, -dy))  # 0 deg is North (-y)
        return (angle + 360.0) % 360.0


@dataclass
class CollaborativeNode:
    """Simulated remote peer node participating in distributed awareness."""

    node_id: str
    label: str
    status: Literal["connected", "degraded", "offline"]
    link_quality: float  # [0.0, 1.0]
    latency_ms: float
    packets_exchanged: int = 0
    observations_shared: int = 0


@dataclass
class ScenarioState:
    """Snapshot of the active scenario at a discrete time step."""

    step: int
    sim_time_sec: float
    difficulty: DifficultyPreset
    seed: int
    objects: list[SyntheticObject] = field(default_factory=list)
    nodes: list[CollaborativeNode] = field(default_factory=list)
    active_threat_level: UrgencyLevel = "low"
    is_paused: bool = False


class ScenarioEngine:
    """Deterministic, configurable generator of academic scenario objects and nodes."""

    DIFFICULTY_PARAMS = {
        "low": {"object_count": 3, "speed_scale": 0.008, "spawn_rate": 0.15},
        "medium": {"object_count": 5, "speed_scale": 0.015, "spawn_rate": 0.30},
        "high": {"object_count": 8, "speed_scale": 0.025, "spawn_rate": 0.50},
    }

    def __init__(
        self,
        seed: int = 42,
        difficulty: DifficultyPreset = "medium",
    ) -> None:
        self.seed = seed
        self.difficulty = difficulty
        self.step_count = 0
        self.sim_time = 0.0
        self.is_paused = False
        self.objects: list[SyntheticObject] = []
        self.nodes: list[CollaborativeNode] = []
        self._rng_state = seed

        self.reset(seed=seed, difficulty=difficulty)

    def _next_random(self) -> float:
        """Deterministic Linear Congruential Generator for offline reproducibility."""
        self._rng_state = (self._rng_state * 1664525 + 1013904223) % (2**32)
        return self._rng_state / (2**32)

    def _random_range(self, low: float, high: float) -> float:
        return low + (high - low) * self._next_random()

    def _random_choice(self, choices: list) -> any:
        idx = int(self._next_random() * len(choices)) % len(choices)
        return choices[idx]

    def reset(
        self,
        seed: int | None = None,
        difficulty: DifficultyPreset | None = None,
    ) -> ScenarioState:
        """Reset the scenario to initial conditions."""
        if seed is not None:
            self.seed = seed
        if difficulty is not None:
            self.difficulty = difficulty

        self._rng_state = self.seed
        self.step_count = 0
        self.sim_time = 0.0
        self.is_paused = False
        self.objects.clear()

        # Initialize collaborative nodes
        self.nodes = [
            CollaborativeNode(
                node_id="node_alpha",
                label="Node-Alpha (Forward)",
                status="connected",
                link_quality=0.96,
                latency_ms=12.4,
            ),
            CollaborativeNode(
                node_id="node_bravo",
                label="Node-Bravo (Flank)",
                status="connected",
                link_quality=0.89,
                latency_ms=21.8,
            ),
            CollaborativeNode(
                node_id="node_charlie",
                label="Node-Charlie (Rear Guard)",
                status="connected",
                link_quality=0.92,
                latency_ms=18.1,
            ),
        ]

        # Populate initial objects based on difficulty
        target_count = self.DIFFICULTY_PARAMS[self.difficulty]["object_count"]
        for i in range(target_count):
            self._spawn_object(f"OBJ-{i + 1:02d}")

        return self.get_state()

    def _spawn_object(self, obj_id: str) -> None:
        categories: list[ObjectCategory] = ["friendly", "neutral", "unknown", "resource"]
        category = self._random_choice(categories)

        # Place around perimeter or random quadrant
        angle_rad = self._random_range(0.0, 2.0 * math.pi)
        radius = self._random_range(0.25, 0.45)
        x = max(0.05, min(0.95, 0.5 + radius * math.cos(angle_rad)))
        y = max(0.05, min(0.95, 0.5 + radius * math.sin(angle_rad)))

        # Heading pointing roughly towards or across center
        target_heading = math.degrees(math.atan2(0.5 - y, 0.5 - x))
        heading_deg = (target_heading + self._random_range(-45.0, 45.0) + 360.0) % 360.0

        base_speed = self.DIFFICULTY_PARAMS[self.difficulty]["speed_scale"]
        speed = self._random_range(base_speed * 0.7, base_speed * 1.3)
        confidence = round(self._random_range(0.75, 0.98), 2)

        source_node = self._random_choice(
            ["synthetic_radar_primary", "node_alpha", "node_bravo", "node_charlie"]
        )

        # Urgency depends on proximity and category
        dist = math.hypot(x - 0.5, y - 0.5)
        if category in {"unknown"} and dist < 0.20:
            urgency: UrgencyLevel = "critical"
        elif category in {"unknown", "neutral"} and dist < 0.35:
            urgency = "approaching"
        else:
            urgency = "low"

        self.objects.append(
            SyntheticObject(
                id=obj_id,
                category=category,
                x=x,
                y=y,
                heading_deg=heading_deg,
                speed=speed,
                confidence=confidence,
                urgency=urgency,
                source=source_node,
                timestamp=self.sim_time,
            )
        )

    def step(self, dt: float = 1.0) -> ScenarioState:
        """Advance the simulation by one discrete time step."""
        if self.is_paused:
            return self.get_state()

        self.step_count += 1
        self.sim_time += dt

        updated_objects: list[SyntheticObject] = []
        for obj in self.objects:
            rad = math.radians(obj.heading_deg)
            # update position
            new_x = obj.x + math.cos(rad) * obj.speed * dt
            new_y = obj.y + math.sin(rad) * obj.speed * dt

            # Wrap or bounce within normalized [0.05, 0.95] boundary
            if new_x < 0.05 or new_x > 0.95:
                obj.heading_deg = (180.0 - obj.heading_deg) % 360.0
                new_x = max(0.05, min(0.95, new_x))
            if new_y < 0.05 or new_y > 0.95:
                obj.heading_deg = (-obj.heading_deg) % 360.0
                new_y = max(0.05, min(0.95, new_y))

            obj.x = new_x
            obj.y = new_y
            obj.timestamp = self.sim_time

            # Update urgency based on proximity to ownship
            dist = obj.distance_from_ownship
            if obj.category == "unknown" and dist < 0.22:
                obj.urgency = "critical"
            elif obj.category in {"unknown", "neutral"} and dist < 0.35:
                obj.urgency = "approaching"
            else:
                obj.urgency = "low"

            updated_objects.append(obj)

        self.objects = updated_objects

        # Update node telemetry
        for node in self.nodes:
            jitter = self._random_range(-0.02, 0.02)
            node.link_quality = max(0.60, min(1.0, node.link_quality + jitter))
            node.latency_ms = max(8.0, min(45.0, node.latency_ms + self._random_range(-1.5, 1.5)))
            node.packets_exchanged += int(self._random_range(5, 15))
            node.observations_shared += int(self._random_range(1, 4))
            node.status = "connected" if node.link_quality > 0.75 else "degraded"

        return self.get_state()

    def get_state(self) -> ScenarioState:
        """Return the current scenario snapshot."""
        # Active threat level is max across objects
        has_critical = any(o.urgency == "critical" for o in self.objects)
        has_approaching = any(o.urgency == "approaching" for o in self.objects)
        threat_level: UrgencyLevel = (
            "critical" if has_critical else ("approaching" if has_approaching else "low")
        )

        return ScenarioState(
            step=self.step_count,
            sim_time_sec=round(self.sim_time, 2),
            difficulty=self.difficulty,
            seed=self.seed,
            objects=list(self.objects),
            nodes=list(self.nodes),
            active_threat_level=threat_level,
            is_paused=self.is_paused,
        )

    def to_observations(self) -> list[Observation]:
        """Convert current scenario objects and nodes into Observation objects for Fusion."""
        observations: list[Observation] = []
        for obj in self.objects:
            age = max(0.0, self.sim_time - obj.timestamp)
            observations.append(
                Observation(
                    source_id=f"RADAR_{obj.id}",
                    source_type="synthetic_scenario_radar",
                    label=obj.category.upper(),
                    confidence=obj.confidence,
                    uncertainty=round(1.0 - obj.confidence, 3),
                    relative_urgency=obj.urgency,
                    timestamp=obj.timestamp,
                    information_age=age,
                )
            )

        for node in self.nodes:
            if node.status == "connected":
                observations.append(
                    Observation(
                        source_id=node.node_id,
                        source_type="collaborative_node",
                        label="RESOURCE" if node.link_quality > 0.85 else "NEUTRAL",
                        confidence=round(node.link_quality, 2),
                        uncertainty=round(1.0 - node.link_quality, 3),
                        relative_urgency="low",
                        timestamp=self.sim_time,
                        information_age=node.latency_ms / 1000.0,
                    )
                )

        return observations
