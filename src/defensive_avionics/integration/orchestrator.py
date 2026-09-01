"""Integrated orchestrator combining Signal, Vision, Scenario, Fusion, and Policy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from defensive_avionics.common.types import (
    PolicyPrediction,
    SignalPrediction,
    Status,
    SystemSnapshot,
    Urgency,
    VisionPrediction,
)
from defensive_avionics.fusion.engine import FusedState, Observation, SensorFusionEngine
from defensive_avionics.policy.baseline import choose_action as choose_baseline_action
from defensive_avionics.policy.environment import ACTION_NAMES
from defensive_avionics.scenario.engine import DifficultyPreset, ScenarioEngine, ScenarioState
from defensive_avionics.signal.pipeline import SignalPipeline
from defensive_avionics.vision.pipeline import VisionPipeline


@dataclass(frozen=True, slots=True)
class ComprehensiveSnapshot:
    """Rich multi-modal system snapshot for dashboard telemetry and logs."""

    frame_id: int
    sim_time_sec: float
    status: Status
    signal: SignalPrediction
    vision: VisionPrediction
    fused: FusedState
    scenario: ScenarioState
    policy_action: str
    policy_confidence: float
    policy_observation: tuple[float, ...]
    signal_source_mode: str
    vision_source_mode: str
    policy_source_mode: str
    recent_actions: list[str] = field(default_factory=list)

    def to_legacy_snapshot(self) -> SystemSnapshot:
        """Convert to legacy SystemSnapshot for backward compatibility."""
        return SystemSnapshot(
            frame_id=self.frame_id,
            status=self.status,
            signal_label=self.signal.label,
            signal_confidence=self.signal.confidence,
            vision_urgency=self.vision.urgency,
            policy_action=self.policy_action,
        )


def combine_predictions(
    frame_id: int,
    signal: SignalPrediction,
    vision: VisionPrediction,
    policy: PolicyPrediction,
) -> SystemSnapshot:
    """Legacy helper function preserved for backwards compatibility."""
    if vision.urgency == "critical":
        status: Status = "critical"
    elif vision.urgency == "approaching" or signal.confidence < 0.60:
        status = "caution"
    else:
        status = "stable"

    return SystemSnapshot(
        frame_id=frame_id,
        status=status,
        signal_label=signal.label,
        signal_confidence=signal.confidence,
        vision_urgency=vision.urgency,
        policy_action=policy.action_name,
    )


class DemoOrchestrator:
    """Deterministic sequence used before trained models are available."""

    def synthetic_sequence(self) -> list[SystemSnapshot]:
        urgencies: tuple[Urgency, ...] = ("low", "low", "approaching", "critical")
        snapshots: list[SystemSnapshot] = []
        for frame_id, urgency in enumerate(urgencies):
            snapshots.append(
                combine_predictions(
                    frame_id,
                    SignalPrediction("QPSK", 0.86, 4.0),
                    VisionPrediction(urgency != "low", 0.80, urgency),
                    PolicyPrediction("observe", 0.70),
                )
            )
        return snapshots


class IntegratedOrchestrator:
    """Full production-grade simulation orchestrator.

    Integrates Signal Pipeline, Vision Pipeline, Scenario Engine,
    Sensor Fusion, and PPO Policy into a cohesive pipeline.
    """

    def __init__(
        self,
        seed: int = 42,
        difficulty: DifficultyPreset = "medium",
        signal_checkpoint: str | Path | None = None,
        vision_checkpoint: str | Path | None = None,
        policy_checkpoint: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.seed = seed
        self.difficulty = difficulty
        self.device = device
        self.step_count = 0
        self.sim_time = 0.0

        # Subsystems
        self.signal_pipeline = SignalPipeline(
            checkpoint_path=signal_checkpoint,
            device=device,
        )
        self.vision_pipeline = VisionPipeline(
            model_path=vision_checkpoint,
            device=device,
        )
        self.scenario_engine = ScenarioEngine(
            seed=seed,
            difficulty=difficulty,
        )
        self.fusion_engine = SensorFusionEngine(half_life_seconds=4.0)

        # Policy model
        self.ppo_model: Any = None
        self.policy_loaded = False
        self.policy_checkpoint = (
            Path(policy_checkpoint) if policy_checkpoint else Path("models/policy/best_model.zip")
        )
        self._init_policy()

        # Virtual system resources for policy state tracking
        self.resource_a = 1.0
        self.resource_b = 1.0
        self.action_history: list[str] = []

    def _init_policy(self) -> None:
        """Attempt to load trained PPO policy model."""
        target_path: Path | None = None
        if self.policy_checkpoint.is_file():
            target_path = self.policy_checkpoint
        elif Path("models/policy/ppo_policy_final.zip").is_file():
            target_path = Path("models/policy/ppo_policy_final.zip")

        if target_path:
            try:
                from stable_baselines3 import PPO

                self.ppo_model = PPO.load(str(target_path), device="cpu")
                self.policy_loaded = True
            except Exception as exc:
                print(f"Warning: Failed to load PPO policy ({exc}). Using rule-based heuristic.")
                self.ppo_model = None
                self.policy_loaded = False

    def reset(
        self,
        seed: int | None = None,
        difficulty: DifficultyPreset | None = None,
    ) -> ComprehensiveSnapshot:
        """Reset the simulation state."""
        if seed is not None:
            self.seed = seed
        if difficulty is not None:
            self.difficulty = difficulty

        self.step_count = 0
        self.sim_time = 0.0
        self.resource_a = 1.0
        self.resource_b = 1.0
        self.action_history.clear()
        self.scenario_engine.reset(seed=self.seed, difficulty=self.difficulty)
        self.vision_pipeline.tracker.reset()

        return self.step()

    def step(
        self,
        signal_sample: np.ndarray | None = None,
        vision_frame: np.ndarray | None = None,
        live_observation: Observation | None = None,
        dt: float = 1.0,
    ) -> ComprehensiveSnapshot:
        """Execute one complete multi-modal simulation step."""
        self.step_count += 1
        self.sim_time += dt

        # 1. Step Scenario Engine
        scenario_state = self.scenario_engine.step(dt=dt)
        scenario_observations = self.scenario_engine.to_observations()

        # 2. Step Signal Module
        modulations = ["QPSK", "BPSK", "8PSK", "PAM4", "QAM16"]
        target_mod = modulations[self.step_count % len(modulations)]
        if signal_sample is None:
            signal_sample = self.signal_pipeline.generate_synthetic_iq(
                modulation=target_mod,
                snr_db=10.0 - (self.step_count % 4) * 2.0,
                seed=self.seed + self.step_count,
            )
        signal_pred = self.signal_pipeline.predict(signal_sample, snr_db=8.0)

        # 3. Step Vision Module
        if live_observation is not None:
            vision_pred = VisionPrediction(
                detected=live_observation.confidence > 0.0,
                confidence=live_observation.confidence,
                urgency=live_observation.relative_urgency,
            )
        else:
            if vision_frame is None:
                shapes = ["triangle", "diamond", "circle"]
                target_shape = shapes[self.step_count % len(shapes)]
                scale = 0.8 + 0.15 * (self.step_count % 5)
                vision_frame, _ = VisionPipeline.generate_synthetic_sky_frame(
                    shape_type=target_shape,
                    scale=scale,
                    seed=self.seed + self.step_count,
                )
            vision_pred = self.vision_pipeline.predict(vision_frame)

        # 4. Assemble Observations for Sensor Fusion
        all_observations: list[Observation] = []

        # Signal observation
        all_observations.append(
            Observation(
                source_id="RF_RECEIVER_01",
                source_type="signal_classifier",
                label=signal_pred.label,
                confidence=signal_pred.confidence,
                uncertainty=round(1.0 - signal_pred.confidence, 3),
                relative_urgency="low",
                timestamp=self.sim_time,
                information_age=0.0,
            )
        )

        # Vision observation
        if live_observation is not None:
            all_observations.append(live_observation)
        elif vision_pred.detected:
            all_observations.append(
                Observation(
                    source_id="OPTICAL_SENSOR_01",
                    source_type="vision_detector",
                    label="AERIAL_OBJECT",
                    confidence=vision_pred.confidence,
                    uncertainty=round(1.0 - vision_pred.confidence, 3),
                    relative_urgency=vision_pred.urgency,
                    timestamp=self.sim_time,
                    information_age=0.0,
                )
            )

        # Scenario observations (radar & peer nodes)
        all_observations.extend(scenario_observations)

        # 5. Fuse Observations
        fused_state = self.fusion_engine.fuse(all_observations)

        # 6. Formulate 7-dimensional Abstract State for PPO Policy
        # OBSERVATION_NAMES:
        # 0: scenario_intensity, 1: uncertainty, 2: change_rate,
        # 3: signal_confidence, 4: visual_urgency, 5: resource_a, 6: resource_b
        threat_count = sum(
            1 for o in scenario_state.objects if o.urgency in {"approaching", "critical"}
        )
        intensity = min(1.0, max(0.1, threat_count / max(1, len(scenario_state.objects)) + 0.2))
        uncertainty = fused_state.fused_uncertainty
        change_rate = 0.35 + 0.1 * math.sin(self.step_count * 0.5)
        sig_conf = signal_pred.confidence
        vis_urgency_val = (
            0.9
            if fused_state.relative_urgency == "critical"
            else (0.5 if fused_state.relative_urgency == "approaching" else 0.15)
        )

        obs_vector = np.array(
            [
                intensity,
                uncertainty,
                change_rate,
                sig_conf,
                vis_urgency_val,
                self.resource_a,
                self.resource_b,
            ],
            dtype=np.float32,
        )

        # 7. Select Policy Action
        policy_source = "RULE-BASED"
        if self.policy_loaded and self.ppo_model is not None:
            try:
                action_idx, _ = self.ppo_model.predict(obs_vector, deterministic=True)
                act = int(action_idx)
                policy_source = "TRAINED PPO"
            except Exception:
                act = choose_baseline_action(obs_vector)
        else:
            act = choose_baseline_action(obs_vector)

        action_name = ACTION_NAMES[act]
        self.action_history.append(action_name)
        if len(self.action_history) > 10:
            self.action_history.pop(0)

        # Apply resource dynamics
        if act == 1:
            self.resource_a = max(0.0, self.resource_a - 0.08)
        elif act == 2:
            self.resource_b = max(0.0, self.resource_b - 0.08)
        self.resource_a = min(1.0, self.resource_a + 0.01)
        self.resource_b = min(1.0, self.resource_b + 0.01)

        # Determine overall system status
        if fused_state.relative_urgency == "critical":
            status: Status = "critical"
        elif fused_state.relative_urgency == "approaching" or fused_state.fused_uncertainty > 0.60:
            status = "caution"
        else:
            status = "stable"

        sig_mode = (
            "[TRAINED 1D-CNN]" if self.signal_pipeline.is_trained_model else "[SYNTHETIC DEMO]"
        )
        if live_observation is not None:
            vis_mode = f"[LIVE CAMERA: {live_observation.label}]"
        elif self.vision_pipeline.is_trained_model:
            vis_mode = "[TRAINED YOLO]"
        else:
            vis_mode = "[SYNTHETIC DEMO]"
        pol_mode = f"[{policy_source}]"

        return ComprehensiveSnapshot(
            frame_id=self.step_count,
            sim_time_sec=round(self.sim_time, 2),
            status=status,
            signal=signal_pred,
            vision=vision_pred,
            fused=fused_state,
            scenario=scenario_state,
            policy_action=action_name,
            policy_confidence=0.88,
            policy_observation=tuple(round(float(v), 3) for v in obs_vector),
            signal_source_mode=sig_mode,
            vision_source_mode=vis_mode,
            policy_source_mode=pol_mode,
            recent_actions=list(self.action_history),
        )
