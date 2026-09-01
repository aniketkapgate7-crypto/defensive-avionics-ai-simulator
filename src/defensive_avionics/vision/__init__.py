"""Synthetic, licensed, and real-time live visual detection module."""

from defensive_avionics.vision.approach import ApproachEstimate, ExpansionTracker, VisualTrend
from defensive_avionics.vision.detector import Detection, GenericSkyDetector
from defensive_avionics.vision.live_camera import (
    CLASSROOM_SAFE_LABELS,
    LiveCameraProcessor,
    LiveCameraState,
)
from defensive_avionics.vision.urgency import estimate_urgency

__all__ = [
    "ApproachEstimate",
    "CLASSROOM_SAFE_LABELS",
    "Detection",
    "ExpansionTracker",
    "GenericSkyDetector",
    "LiveCameraProcessor",
    "LiveCameraState",
    "VisualTrend",
    "estimate_urgency",
]
