from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defensive_avionics.integration.orchestrator import DemoOrchestrator  # noqa: E402


class OrchestratorTests(unittest.TestCase):
    def test_demo_sequence_is_deterministic(self) -> None:
        first = DemoOrchestrator().synthetic_sequence()
        second = DemoOrchestrator().synthetic_sequence()
        self.assertEqual(first, second)
        self.assertEqual(
            [item.status for item in first],
            ["stable", "stable", "caution", "critical"],
        )


if __name__ == "__main__":
    unittest.main()
