"""Validate repository structure and report optional dependency availability."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "pyproject.toml",
    "configs/default.yaml",
    "configs/signal.yaml",
    "configs/policy.yaml",
    "configs/vision.yaml",
    "src/defensive_avionics/__init__.py",
    "tests/test_config.py",
)
OPTIONAL_IMPORTS = ("numpy", "yaml", "torch", "gymnasium", "cv2", "pygame")


def main() -> int:
    missing_files = [relative for relative in REQUIRED if not (ROOT / relative).exists()]
    if missing_files:
        print("Missing required project files:")
        for relative in missing_files:
            print(f"  - {relative}")
        return 1

    print("Project structure: OK")
    print("Optional dependencies:")
    for package in OPTIONAL_IMPORTS:
        state = "installed" if importlib.util.find_spec(package) else "not installed yet"
        print(f"  - {package}: {state}")
    print("Safe runtime scope: offline academic simulation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
