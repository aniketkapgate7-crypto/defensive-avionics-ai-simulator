from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defensive_avionics.config import CONFIG_DIR, resolve_project_path  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_config_directory_is_inside_project(self) -> None:
        self.assertEqual(CONFIG_DIR, ROOT / "configs")

    def test_relative_path_resolution(self) -> None:
        self.assertEqual(resolve_project_path("data/raw"), (ROOT / "data/raw").resolve())


if __name__ == "__main__":
    unittest.main()
