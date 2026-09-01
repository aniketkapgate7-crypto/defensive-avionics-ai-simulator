"""Script to simulate camera frames and verify stable, growing, and receding trends."""

from __future__ import annotations

import time

import cv2
import numpy as np

from defensive_avionics.vision.detector import Detection
from defensive_avionics.vision.live_camera import LiveCameraProcessor, LiveCameraState


def test_trend_simulation() -> None:
    processor = LiveCameraProcessor(
        model_mode="classroom",
        image_size=320,
        window_size=16,
        minimum_samples=6,
        min_duration_sec=0.0,
        stable_tolerance=0.03,
        rapid_growth_threshold=0.15,
        hysteresis_count=2,
    )

    # 1. Test GROWING Sequence: simulated bottle expanding from 50x50 to 200x200
    print("--- 1. Testing GROWING sequence ---")
    processor.reset()
    for step in range(12):
        w = 50 + step * 12
        h = 50 + step * 12
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x1, y1 = 200, 150
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (100, 200, 50), -1)

        with processor._lock:
            area_ratio = float(w * h) / (480 * 640)
            det = Detection(
                label="bottle",
                confidence=0.88,
                box=(x1, y1, x1 + w, y1 + h),
                area_ratio=area_ratio,
            )
            processor._select_stable_object([det])
            raw_conf = det.confidence
            processor._smoothed_conf = 0.88
            est = processor.tracker.update(area_ratio, timestamp=time.time())
            processor._state = LiveCameraState(
                label=det.label,
                raw_confidence=raw_conf,
                confidence=0.88,
                area_ratio=area_ratio,
                relative_growth=est.relative_growth,
                trend=est.trend,
                fps=15.0,
                detection_count=1,
                inference_timestamp=time.time(),
                is_active=True,
                box=det.box,
            )

    st_growing = processor.get_state()
    print(
        f"Result: label={st_growing.label}, trend={st_growing.trend}, "
        f"area={st_growing.area_ratio:.3f}, growth={st_growing.relative_growth:+.2%}/s"
    )
    assert st_growing.trend in {"growing", "rapid_growth"}, (
        f"Expected growing, got {st_growing.trend}"
    )

    # 2. Test STABLE Sequence: simulated box with tiny +/- 1px jitter
    print("\n--- 2. Testing STABLE sequence with jitter ---")
    processor.reset()
    for step in range(12):
        jitter = (step % 2) * 2 - 1
        w = 100 + jitter
        h = 100 + jitter
        area_ratio = float(w * h) / (480 * 640)
        with processor._lock:
            det = Detection(
                label="book",
                confidence=0.90,
                box=(150, 150, 150 + w, 150 + h),
                area_ratio=area_ratio,
            )
            processor._select_stable_object([det])
            est = processor.tracker.update(area_ratio, timestamp=time.time())
            processor._state = LiveCameraState(
                label=det.label,
                raw_confidence=0.90,
                confidence=0.90,
                area_ratio=area_ratio,
                relative_growth=est.relative_growth,
                trend=est.trend,
                fps=15.0,
                detection_count=1,
                inference_timestamp=time.time(),
                is_active=True,
                box=det.box,
            )

    st_stable = processor.get_state()
    print(
        f"Result: label={st_stable.label}, trend={st_stable.trend}, "
        f"area={st_stable.area_ratio:.3f}, growth={st_stable.relative_growth:+.2%}/s"
    )
    assert st_stable.trend == "stable", f"Expected stable, got {st_stable.trend}"

    # 3. Test RECEDING Sequence: simulated box shrinking from 200x200 to 50x50
    print("\n--- 3. Testing RECEDING sequence ---")
    processor.reset()
    for step in range(12):
        w = 200 - step * 12
        h = 200 - step * 12
        area_ratio = float(w * h) / (480 * 640)
        with processor._lock:
            det = Detection(
                label="cell phone",
                confidence=0.85,
                box=(200, 200, 200 + w, 200 + h),
                area_ratio=area_ratio,
            )
            processor._select_stable_object([det])
            est = processor.tracker.update(area_ratio, timestamp=time.time())
            processor._state = LiveCameraState(
                label=det.label,
                raw_confidence=0.85,
                confidence=0.85,
                area_ratio=area_ratio,
                relative_growth=est.relative_growth,
                trend=est.trend,
                fps=15.0,
                detection_count=1,
                inference_timestamp=time.time(),
                is_active=True,
                box=det.box,
            )

    st_receding = processor.get_state()
    print(
        f"Result: label={st_receding.label}, trend={st_receding.trend}, "
        f"area={st_receding.area_ratio:.3f}, growth={st_receding.relative_growth:+.2%}/s"
    )
    assert st_receding.trend == "receding", f"Expected receding, got {st_receding.trend}"

    # 4. Test NO OBJECT DETECTED when removed / reset
    print("\n--- 4. Testing NO OBJECT DETECTED sequence ---")
    processor.reset()
    st_empty = processor.get_state()
    print(
        f"Result after reset: label={st_empty.label}, "
        f"count={st_empty.detection_count}"
    )
    assert st_empty.label == "NO OBJECT DETECTED"
    assert processor.to_observation() is None

    processor.shutdown()
    print("\n[ALL TRENDS CONFIRMED: GROWING, STABLE, RECEDING, NO OBJECT DETECTED]")


if __name__ == "__main__":
    test_trend_simulation()
