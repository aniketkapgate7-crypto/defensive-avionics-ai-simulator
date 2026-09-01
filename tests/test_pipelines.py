"""Unit tests for SignalPipeline, VisionPipeline, and IntegratedOrchestrator."""

from __future__ import annotations

import unittest

from defensive_avionics.integration.orchestrator import IntegratedOrchestrator
from defensive_avionics.signal.pipeline import SignalPipeline
from defensive_avionics.vision.approach import ExpansionTracker
from defensive_avionics.vision.pipeline import VisionPipeline


class PipelineTests(unittest.TestCase):
    def test_signal_pipeline_synthetic_generation_and_prediction(self) -> None:
        pipeline = SignalPipeline(checkpoint_path="nonexistent.pt")
        # Generates deterministic synthetic waveform
        iq = pipeline.generate_synthetic_iq(modulation="QPSK", snr_db=10.0, seed=42)
        self.assertEqual(iq.shape, (2, 128))

        pred = pipeline.predict(iq, snr_db=10.0)
        self.assertIsNotNone(pred.label)
        self.assertGreaterEqual(pred.confidence, 0.0)
        self.assertLessEqual(pred.confidence, 1.0)
        self.assertEqual(pred.snr_db, 10.0)

    def test_signal_spectrogram_computation(self) -> None:
        pipeline = SignalPipeline()
        iq = pipeline.generate_synthetic_iq(modulation="BPSK", seed=42)
        freqs, times, spec = pipeline.compute_spectrogram(iq, nperseg=32)
        self.assertEqual(spec.ndim, 2)
        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(times), 0)

    def test_vision_expansion_tracker_trends(self) -> None:
        tracker = ExpansionTracker(window_size=6, minimum_samples=3)

        # Initial state has insufficient data
        est1 = tracker.update(0.01)
        self.assertEqual(est1.trend, "insufficient_data")

        # Stable sequence
        tracker.update(0.01)
        tracker.update(0.0101)
        est_stable = tracker.update(0.01)
        self.assertEqual(est_stable.trend, "stable")

        # Rapid growth sequence
        tracker.reset()
        tracker.update(0.01)
        tracker.update(0.02)
        tracker.update(0.05)
        est_growth = tracker.update(0.12)
        self.assertIn(est_growth.trend, {"growing", "rapid_growth"})

        # Receding sequence
        tracker.reset()
        tracker.update(0.10)
        tracker.update(0.07)
        tracker.update(0.04)
        est_receding = tracker.update(0.02)
        self.assertEqual(est_receding.trend, "receding")

    def test_vision_pipeline_synthetic_prediction(self) -> None:
        pipeline = VisionPipeline(model_path="nonexistent.pt")
        frame, _ = VisionPipeline.generate_synthetic_sky_frame(shape_type="circle", seed=42)
        self.assertEqual(frame.shape, (320, 320, 3))

        pred = pipeline.predict(frame)
        self.assertIsInstance(pred.detected, bool)
        self.assertIn(pred.urgency, {"low", "approaching", "critical"})

    def test_integrated_orchestrator_step(self) -> None:
        orchestrator = IntegratedOrchestrator(seed=42, difficulty="medium")
        snapshot = orchestrator.step()

        self.assertEqual(snapshot.frame_id, 1)
        self.assertIn(snapshot.status, {"stable", "caution", "critical"})
        self.assertIsNotNone(snapshot.signal)
        self.assertIsNotNone(snapshot.vision)
        self.assertIsNotNone(snapshot.fused)
        self.assertIsNotNone(snapshot.scenario)
        self.assertIsNotNone(snapshot.policy_action)
        self.assertEqual(len(snapshot.policy_observation), 7)

        # Step again to verify continuity
        snapshot2 = orchestrator.step()
        self.assertEqual(snapshot2.frame_id, 2)
        self.assertGreater(snapshot2.sim_time_sec, snapshot.sim_time_sec)


if __name__ == "__main__":
    unittest.main()
