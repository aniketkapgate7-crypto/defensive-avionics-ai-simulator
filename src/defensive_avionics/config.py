"""Configuration helpers with no import-time third-party dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def resolve_project_path(relative_path: str | Path) -> Path:
    """Resolve a repository-relative path without requiring it to exist."""

    return (PROJECT_ROOT / Path(relative_path)).resolve()


def load_yaml(name: str) -> dict[str, Any]:
    """Load one YAML file from configs/ and reject paths outside that folder."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local setup
        raise RuntimeError("Install PyYAML before loading configuration files.") from exc

    candidate = (CONFIG_DIR / name).resolve()
    if candidate.parent != CONFIG_DIR.resolve() or candidate.suffix not in {".yaml", ".yml"}:
        raise ValueError("Configuration must be a YAML file directly inside configs/.")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    with candidate.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {candidate.name}.")
    return loaded
