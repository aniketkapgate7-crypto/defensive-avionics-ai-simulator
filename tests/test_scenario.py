"""Unit tests for the synthetic scenario engine."""

from __future__ import annotations

import unittest

from defensive_avionics.scenario.engine import ScenarioEngine


class ScenarioEngineTests(unittest.TestCase):
    def test_deterministic_initialization(self) -> None:
        engine1 = ScenarioEngine(seed=42, difficulty="medium")
        engine2 = ScenarioEngine(seed=42, difficulty="medium")

        state1 = engine1.get_state()
        state2 = engine2.get_state()

        self.assertEqual(len(state1.objects), len(state2.objects))
        for obj1, obj2 in zip(state1.objects, state2.objects, strict=True):
            self.assertEqual(obj1.id, obj2.id)
            self.assertEqual(obj1.category, obj2.category)
            self.assertAlmostEqual(obj1.x, obj2.x, places=5)
            self.assertAlmostEqual(obj1.y, obj2.y, places=5)

    def test_difficulty_presets(self) -> None:
        low_eng = ScenarioEngine(difficulty="low")
        med_eng = ScenarioEngine(difficulty="medium")
        high_eng = ScenarioEngine(difficulty="high")

        self.assertEqual(len(low_eng.get_state().objects), 3)
        self.assertEqual(len(med_eng.get_state().objects), 5)
        self.assertEqual(len(high_eng.get_state().objects), 8)

    def test_step_advances_positions_within_bounds(self) -> None:
        engine = ScenarioEngine(seed=123, difficulty="medium")
        initial_time = engine.sim_time

        state = engine.step(dt=1.0)
        self.assertEqual(state.step, 1)
        self.assertAlmostEqual(state.sim_time_sec, initial_time + 1.0)

        for obj in state.objects:
            self.assertGreaterEqual(obj.x, 0.05)
            self.assertLessEqual(obj.x, 0.95)
            self.assertGreaterEqual(obj.y, 0.05)
            self.assertLessEqual(obj.y, 0.95)
            self.assertIn(obj.category, {"friendly", "neutral", "unknown", "resource"})
            self.assertIn(obj.urgency, {"low", "approaching", "critical"})

    def test_to_observations(self) -> None:
        engine = ScenarioEngine(seed=42, difficulty="medium")
        observations = engine.to_observations()

        self.assertGreaterEqual(len(observations), len(engine.objects))
        for obs in observations:
            self.assertGreaterEqual(obs.confidence, 0.0)
            self.assertLessEqual(obs.confidence, 1.0)
            self.assertGreaterEqual(obs.information_age, 0.0)

    def test_collaborative_nodes_active(self) -> None:
        engine = ScenarioEngine(seed=42)
        state = engine.get_state()
        self.assertEqual(len(state.nodes), 3)
        node_ids = {n.node_id for n in state.nodes}
        self.assertIn("node_alpha", node_ids)
        self.assertIn("node_bravo", node_ids)
        self.assertIn("node_charlie", node_ids)

        for node in state.nodes:
            self.assertGreater(node.link_quality, 0.0)
            self.assertGreater(node.latency_ms, 0.0)

    def test_reset_preserves_determinism(self) -> None:
        engine = ScenarioEngine(seed=99)
        engine.step(1.0)
        engine.step(1.0)
        state_after_steps = engine.get_state()

        engine.reset(seed=99)
        engine.step(1.0)
        engine.step(1.0)
        state_after_reset = engine.get_state()

        self.assertEqual(state_after_steps.step, state_after_reset.step)
        for obj1, obj2 in zip(state_after_steps.objects, state_after_reset.objects, strict=True):
            self.assertAlmostEqual(obj1.x, obj2.x, places=5)
            self.assertAlmostEqual(obj1.y, obj2.y, places=5)


if __name__ == "__main__":
    unittest.main()
