"""Preprocessing functions for fixed-length I/Q examples."""

from __future__ import annotations

from typing import Any


def normalize_iq(iq: Any) -> Any:
    """Scale a two-channel I/Q array using its maximum absolute magnitude."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for I/Q preprocessing.") from exc

    array = np.asarray(iq, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError("Expected I/Q data with shape (2, sequence_length).")
    scale = float(np.max(np.abs(array)))
    return array if scale == 0.0 else array / scale
