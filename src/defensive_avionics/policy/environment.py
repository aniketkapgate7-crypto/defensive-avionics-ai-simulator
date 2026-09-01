"""Dimensionless academic environment for decision-policy learning."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

OBSERVATION_NAMES = (
    "scenario_intensity",
    "uncertainty",
    "change_rate",
    "signal_confidence",
    "visual_urgency",
    "resource_a",
    "resource_b",
)

ACTION_NAMES = (
    "observe",
    "virtual_resource_a",
    "virtual_resource_b",
    "virtual_signal_response",
    "abstract_reposition",
)

ACTION_COSTS = (0.01, 0.06, 0.06, 0.05, 0.08)


class AbstractScenarioEnv(gym.Env):
    """A fictional environment using only normalized values."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 300) -> None:
        super().__init__()

        self.max_steps = max_steps
        self.steps = 0

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        self.state = np.zeros(
            len(OBSERVATION_NAMES),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.steps = 0

        self.state = np.array(
            [
                self.np_random.uniform(0.25, 0.55),
                self.np_random.uniform(0.30, 0.80),
                self.np_random.uniform(0.20, 0.60),
                self.np_random.uniform(0.40, 0.90),
                self.np_random.uniform(0.10, 0.50),
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        information = {
            "scope": "abstract_academic_simulation",
            "observation_names": OBSERVATION_NAMES,
            "action_names": ACTION_NAMES,
        }

        return self.state.copy(), information

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict,
    ]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        action = int(action)

        (
            intensity,
            uncertainty,
            change_rate,
            signal_confidence,
            visual_urgency,
            resource_a,
            resource_b,
        ) = map(float, self.state)

        previous_intensity = intensity
        previous_uncertainty = uncertainty
        previous_urgency = visual_urgency

        random_drift = float(self.np_random.normal(0.0, 0.015))

        intensity += 0.012 + 0.025 * change_rate + random_drift
        visual_urgency += 0.015 * intensity + float(self.np_random.normal(0.0, 0.01))
        uncertainty += 0.008 - 0.012 * signal_confidence

        change_rate += float(self.np_random.normal(0.0, 0.025))
        signal_confidence += float(self.np_random.normal(0.0, 0.012))

        resource_a += 0.003
        resource_b += 0.003
        action_effective = False

        if action == 0:
            uncertainty -= 0.08
            signal_confidence += 0.05
            action_effective = previous_uncertainty > 0.40

        elif action == 1 and resource_a >= 0.08:
            reduction = 0.16 * (0.50 + 0.50 * signal_confidence)
            intensity -= reduction
            resource_a -= 0.08
            action_effective = previous_intensity > 0.45

        elif action == 2 and resource_b >= 0.08:
            reduction = 0.16 * (1.0 - 0.35 * uncertainty)
            visual_urgency -= reduction
            resource_b -= 0.08
            action_effective = previous_urgency > 0.45

        elif action == 3:
            uncertainty -= 0.12
            intensity -= 0.05 * signal_confidence
            action_effective = previous_uncertainty > 0.50

        elif action == 4:
            intensity -= 0.10
            visual_urgency -= 0.10
            uncertainty += 0.02
            action_effective = (
                max(
                    previous_intensity,
                    previous_urgency,
                )
                > 0.65
            )

        next_state = np.array(
            [
                intensity,
                uncertainty,
                change_rate,
                signal_confidence,
                visual_urgency,
                resource_a,
                resource_b,
            ],
            dtype=np.float32,
        )
        self.state = np.clip(
            next_state,
            0.0,
            1.0,
        ).astype(np.float32)

        (
            intensity,
            uncertainty,
            _change_rate,
            _signal_confidence,
            visual_urgency,
            resource_a,
            resource_b,
        ) = map(float, self.state)

        stability_score = 1.0 - (0.45 * intensity + 0.35 * visual_urgency + 0.20 * uncertainty)
        resource_score = (resource_a + resource_b) / 2.0

        reward = 1.40 * stability_score + 0.15 * resource_score - ACTION_COSTS[action]
        reward += 0.20 if action_effective else -0.05

        terminated = intensity >= 0.98 and visual_urgency >= 0.95
        truncated = self.steps >= self.max_steps

        if terminated:
            reward -= 4.0

        information = {
            "step": self.steps,
            "action_name": ACTION_NAMES[action],
            "action_effective": action_effective,
            "stability_score": stability_score,
            "resource_score": resource_score,
        }

        return (
            self.state.copy(),
            float(reward),
            bool(terminated),
            bool(truncated),
            information,
        )


def build_environment(
    max_steps: int = 300,
) -> AbstractScenarioEnv:
    return AbstractScenarioEnv(max_steps=max_steps)
