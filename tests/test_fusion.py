"""Unit tests for the Sensor Fusion engine."""

from __future__ import annotations

import unittest

from defensive_avionics.fusion.engine import (
    FusedState,
    Observation,
    SensorFusionEngine,
)


class SensorFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SensorFusionEngine(half_life_seconds=5.0)

    def test_empty_input(self) -> None:
        result = self.engine.fuse([])
        self.assertIsInstance(result, FusedState)
        self.assertEqual(result.fused_label, "unknown")
        self.assertEqual(result.fused_confidence, 0.0)
        self.assertEqual(result.fused_uncertainty, 1.0)
        self.assertEqual(result.contributing_sources_count, 0)
        self.assertEqual(result.relative_urgency, "low")

    def test_single_observation(self) -> None:
        obs = Observation(
            source_id="SIG_01",
            source_type="signal_classifier",
            label="QPSK",
            confidence=0.90,
            uncertainty=0.10,
            relative_urgency="low",
            information_age=0.0,
        )
        result = self.engine.fuse([obs])
        self.assertEqual(result.fused_label, "QPSK")
        self.assertAlmostEqual(result.fused_confidence, 0.90, places=2)
        self.assertEqual(result.contributing_sources_count, 1)
        self.assertEqual(result.information_freshness, 1.0)

    def test_agreeing_observations_reinforce_confidence(self) -> None:
        obs1 = Observation(
            source_id="RADAR_01",
            source_type="scenario_radar",
            label="UNKNOWN",
            confidence=0.85,
            uncertainty=0.15,
            relative_urgency="approaching",
            information_age=0.0,
        )
        obs2 = Observation(
            source_id="OPTICAL_01",
            source_type="vision_detector",
            label="UNKNOWN",
            confidence=0.88,
            uncertainty=0.12,
            relative_urgency="approaching",
            information_age=0.0,
        )
        result = self.engine.fuse([obs1, obs2])
        self.assertEqual(result.fused_label, "UNKNOWN")
        self.assertGreaterEqual(result.fused_confidence, 0.80)
        self.assertLess(result.fused_uncertainty, 0.30)
        self.assertEqual(result.relative_urgency, "approaching")
        self.assertEqual(result.contributing_sources_count, 2)

    def test_conflicting_observations(self) -> None:
        obs1 = Observation(
            source_id="SIG_01",
            source_type="signal_classifier",
            label="FRIENDLY",
            confidence=0.85,
            uncertainty=0.15,
            relative_urgency="low",
            information_age=0.0,
        )
        obs2 = Observation(
            source_id="RADAR_01",
            source_type="scenario_radar",
            label="UNKNOWN",
            confidence=0.85,
            uncertainty=0.15,
            relative_urgency="critical",
            information_age=0.0,
        )
        result = self.engine.fuse([obs1, obs2])
        # Disagreement should increase uncertainty and dampen fused confidence
        self.assertGreater(result.fused_uncertainty, 0.25)
        self.assertEqual(result.contributing_sources_count, 2)
        self.assertIn("FRIENDLY", result.evidence_breakdown)
        self.assertIn("UNKNOWN", result.evidence_breakdown)

    def test_stale_observation_decay(self) -> None:
        fresh_obs = Observation(
            source_id="NODE_A",
            source_type="collaborative_node",
            label="NEUTRAL",
            confidence=0.85,
            information_age=0.0,
        )
        stale_obs = Observation(
            source_id="NODE_B",
            source_type="collaborative_node",
            label="FRIENDLY",
            confidence=0.85,
            information_age=15.0,  # 3 half-lives
        )
        result = self.engine.fuse([fresh_obs, stale_obs])
        # Fresh observation should dominate the consensus
        self.assertEqual(result.fused_label, "NEUTRAL")
        self.assertGreater(
            result.evidence_breakdown.get("NEUTRAL", 0.0),
            result.evidence_breakdown.get("FRIENDLY", 0.0),
        )

    def test_invalid_confidence_values(self) -> None:
        with self.assertRaises(ValueError):
            Observation(
                source_id="ERR",
                source_type="test",
                label="TEST",
                confidence=1.5,  # > 1.0
            )
        with self.assertRaises(ValueError):
            Observation(
                source_id="ERR",
                source_type="test",
                label="TEST",
                confidence=-0.1,  # < 0.0
            )

    def test_invalid_urgency(self) -> None:
        with self.assertRaises(ValueError):
            Observation(
                source_id="ERR",
                source_type="test",
                label="TEST",
                confidence=0.5,
                relative_urgency="extreme_danger",  # Invalid  # type: ignore
            )

    def test_deterministic_results(self) -> None:
        obs_list = [
            Observation("S1", "type1", "QPSK", 0.88, 0.12, "low", 1.0, 0.5),
            Observation("S2", "type2", "QPSK", 0.92, 0.08, "low", 1.0, 0.2),
        ]
        res1 = self.engine.fuse(obs_list)
        res2 = self.engine.fuse(obs_list)
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    unittest.main()
