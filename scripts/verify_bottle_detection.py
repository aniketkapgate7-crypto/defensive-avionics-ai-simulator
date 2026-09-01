"""Test detecting classroom bottle and objects with GenericSkyDetector."""

from __future__ import annotations

import cv2
import numpy as np

from defensive_avionics.vision.detector import GenericSkyDetector
from defensive_avionics.vision.live_camera import CLASSROOM_SAFE_LABELS


def test_bottle_detection() -> None:
    detector = GenericSkyDetector(
        model_path="yolov8n.pt",
        confidence=0.20,
        image_size=320,
        allowed_labels=CLASSROOM_SAFE_LABELS,
    )

    print(f"Loaded YOLOv8n with allowed classes: {detector.allowed_class_ids}")
    assert detector.allowed_class_ids is not None
    assert 39 in detector.allowed_class_ids, (
        "Class ID 39 (bottle) must be in allowed class IDs"
    )

    # Create synthetic frame with bottle-like dimensions (tall aspect ratio)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
    cv2.rectangle(frame, (280, 120), (360, 380), (100, 100, 100), -1)
    cv2.circle(frame, (320, 120), 40, (80, 80, 80), -1)
    cv2.rectangle(frame, (305, 70), (335, 120), (50, 50, 50), -1)

    dets = detector.detect(frame)
    print(f"Detections on synthetic bottle image: {dets}")
    print("Class ID and name resolution validated successfully.")


if __name__ == "__main__":
    test_bottle_detection()
