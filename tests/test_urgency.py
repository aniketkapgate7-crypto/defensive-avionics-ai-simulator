from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defensive_avionics.vision.urgency import estimate_urgency  # noqa: E402


class UrgencyTests(unittest.TestCase):
    def test_low_urgency(self) -> None:
        self.assertEqual(estimate_urgency([100, 110]), "low")

    def test_approaching(self) -> None:
        self.assertEqual(estimate_urgency([100, 150]), "approaching")

    def test_critical(self) -> None:
        self.assertEqual(estimate_urgency([100, 210]), "critical")

    def test_negative_area_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_urgency([100, -1])


if __name__ == "__main__":
    unittest.main()
