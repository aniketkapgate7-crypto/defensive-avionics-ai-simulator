"""Unit tests for the live camera processing module."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from defensive_avionics.fusion.engine import Observation
from defensive_avionics.vision.approach import ExpansionTracker
from defensive_avionics.vision.detector import Detection, GenericSkyDetector
from defensive_avionics.vision.live_camera import (
    CLASSROOM_SAFE_LABELS,
    LiveCameraProcessor,
    LiveCameraState,
    compute_box_iou,
)


@pytest.fixture
def dummy_bgr_frame() -> np.ndarray:
    """Create a 640x480 3-channel synthetic BGR frame with a drawn rectangle."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (30, 20, 10)
    frame[180:300, 260:380] = (200, 180, 50)
    return frame


def test_live_camera_initial_state() -> None:
    processor = LiveCameraProcessor(model_mode="classroom", image_size=320)
    state = processor.get_state()

    assert isinstance(state, LiveCameraState)
    assert state.label == "NO OBJECT DETECTED"
    assert state.raw_confidence == 0.0
    assert state.confidence == 0.0
    assert state.area_ratio == 0.0
    assert state.trend == "insufficient_data"
    assert not state.is_active
    assert processor.to_observation() is None
    processor.shutdown()


def test_live_camera_process_frame_dimensions_and_pixel_changes(
    dummy_bgr_frame: np.ndarray,
) -> None:
    processor = LiveCameraProcessor(model_mode="classroom", image_size=320)
    original_copy = dummy_bgr_frame.copy()
    annotated = processor.process_frame(dummy_bgr_frame)

    assert annotated is not None
    assert annotated.shape == dummy_bgr_frame.shape
    assert annotated.dtype == dummy_bgr_frame.dtype

    # HUD overlay should modify pixels on the annotated frame
    assert not np.array_equal(annotated, original_copy)

    state = processor.get_state()
    assert state.is_active
    assert state.fps > 0.0
    processor.shutdown()


def test_compute_box_iou() -> None:
    box1 = (10, 10, 50, 50)
    box2 = (10, 10, 50, 50)
    box3 = (30, 30, 70, 70)
    box4 = (100, 100, 150, 150)

    assert compute_box_iou(box1, box2) == pytest.approx(1.0)
    assert 0.0 < compute_box_iou(box1, box3) < 1.0
    assert compute_box_iou(box1, box4) == 0.0


def test_live_camera_invalid_frame_rejection() -> None:
    processor = LiveCameraProcessor(model_mode="classroom", image_size=320)

    gray_frame = np.zeros((480, 640), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel BGR"):
        processor.process_frame(gray_frame)

    rgba_frame = np.zeros((480, 640, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel BGR"):
        processor.process_frame(rgba_frame)

    processor.shutdown()


def test_live_camera_empty_or_none_frame() -> None:
    processor = LiveCameraProcessor(model_mode="classroom", image_size=320)
    empty_frame = np.array([], dtype=np.uint8)

    result = processor.process_frame(empty_frame)
    assert result.size == 0
    processor.shutdown()


def test_allowed_class_id_resolution() -> None:
    """Verify that GenericSkyDetector resolves allowed class names to valid IDs."""
    detector = GenericSkyDetector(
        model_path="yolov8n.pt",
        confidence=0.20,
        allowed_labels=CLASSROOM_SAFE_LABELS,
    )
    assert detector.allowed_class_ids is not None
    assert len(detector.allowed_class_ids) == len(CLASSROOM_SAFE_LABELS)

    # 39=bottle, 67=cell phone, 73=book, 41=cup, 24=backpack
    assert 39 in detector.allowed_class_ids
    assert 73 in detector.allowed_class_ids


def test_confidence_threshold_behavior() -> None:
    """Verify setting confidence threshold dynamically."""
    processor = LiveCameraProcessor(model_mode="classroom", confidence=0.25)
    assert processor.confidence == 0.25

    processor.set_confidence(0.40)
    assert processor.confidence == 0.40
    if processor.detector is not None:
        assert processor.detector.confidence == 0.40
    processor.shutdown()


def test_latest_frame_queue_drops_stale_frames(dummy_bgr_frame: np.ndarray) -> None:
    """Verify that queue maxsize=1 ensures newest frame replaces older queued frames."""
    processor = LiveCameraProcessor(model_mode="classroom")

    # Quickly process several frames
    for _ in range(5):
        processor.process_frame(dummy_bgr_frame)

    # Queue must not grow beyond 1 item
    assert processor._input_queue.qsize() <= 1
    processor.shutdown()


def test_result_age_and_stale_handling() -> None:
    """Verify that results older than 500 ms are marked stale and rejected from fusion."""
    processor = LiveCameraProcessor(model_mode="classroom")

    now = time.time()
    # Inject stale state (> 500 ms ago)
    with processor._lock:
        processor._state = LiveCameraState(
            label="bottle",
            confidence=0.85,
            inference_timestamp=now - 0.70,  # 700 ms old
            is_active=True,
            detection_count=1,
        )

    st = processor.get_state()
    assert st.is_stale
    assert st.result_age_ms >= 500.0

    # Stale observation must be rejected (returns None)
    obs = processor.to_observation()
    assert obs is None
    processor.shutdown()


def test_expansion_tracker_growing() -> None:
    tracker = ExpansionTracker(
        window_size=10,
        minimum_samples=4,
        stable_tolerance=0.03,
        rapid_growth_threshold=0.12,
        hysteresis_count=1,
    )

    areas = [0.02, 0.03, 0.045, 0.065, 0.09, 0.12]
    estimates = [tracker.update(a) for a in areas]

    assert estimates[0].trend == "insufficient_data"
    assert estimates[-1].trend in {"growing", "rapid_growth"}
    assert estimates[-1].relative_growth > 0.03


def test_expansion_tracker_receding() -> None:
    tracker = ExpansionTracker(
        window_size=10,
        minimum_samples=4,
        stable_tolerance=0.03,
        rapid_growth_threshold=0.12,
        hysteresis_count=1,
    )

    areas = [0.12, 0.10, 0.08, 0.06, 0.04, 0.02]
    estimates = [tracker.update(a) for a in areas]

    assert estimates[-1].trend == "receding"
    assert estimates[-1].relative_growth < -0.03


def test_expansion_tracker_stable_jitter() -> None:
    tracker = ExpansionTracker(
        window_size=10,
        minimum_samples=4,
        stable_tolerance=0.03,
        rapid_growth_threshold=0.12,
        hysteresis_count=1,
    )

    areas = [0.050, 0.051, 0.049, 0.050, 0.051, 0.050, 0.049]
    estimates = [tracker.update(a) for a in areas]

    assert estimates[-1].trend == "stable"


def test_expansion_tracker_insufficient_data() -> None:
    tracker = ExpansionTracker(
        window_size=10,
        minimum_samples=6,
        min_duration_sec=0.5,
    )

    est = tracker.update(0.05, timestamp=1.0)
    assert est.trend == "insufficient_data"
    est = tracker.update(0.05, timestamp=1.1)
    assert est.trend == "insufficient_data"
    est = tracker.update(0.05, timestamp=1.2)
    assert est.trend == "insufficient_data"


def test_live_camera_missing_detection_retention_and_reset() -> None:
    processor = LiveCameraProcessor(model_mode="classroom", image_size=320)

    det = Detection(label="bottle", confidence=0.90, box=(100, 100, 200, 200), area_ratio=0.05)
    sel = processor._select_stable_object([det])
    assert sel is not None
    assert sel.label == "bottle"

    for missing_step in range(1, 6):
        coasted = processor._select_stable_object([])
        assert coasted is not None
        assert coasted.label == "bottle"
        assert processor._missing_count == missing_step

    for _ in range(6, 11):
        processor._select_stable_object([])

    assert processor._missing_count >= 10
    assert processor._tracked_box is None
    assert processor._tracked_label is None
    processor.shutdown()


def test_live_camera_object_switching_resets_history() -> None:
    processor = LiveCameraProcessor(model_mode="classroom", image_size=320)

    det_bottle = Detection(
        label="bottle", confidence=0.88, box=(50, 50, 150, 150), area_ratio=0.04
    )
    processor._select_stable_object([det_bottle])
    assert processor._tracked_label == "bottle"

    det_book = Detection(label="book", confidence=0.92, box=(300, 300, 450, 450), area_ratio=0.08)
    sel_book = processor._select_stable_object([det_book])
    assert sel_book is not None
    assert sel_book.label == "book"
    assert processor._tracked_label == "book"
    processor.shutdown()


def test_live_camera_observation_conversion(dummy_bgr_frame: np.ndarray) -> None:
    processor = LiveCameraProcessor(model_mode="classroom", image_size=320)
    processor.process_frame(dummy_bgr_frame)

    now = time.time()
    with processor._lock:
        processor._state = LiveCameraState(
            label="bottle",
            raw_confidence=0.90,
            confidence=0.88,
            area_ratio=0.06,
            relative_growth=0.05,
            trend="growing",
            fps=15.2,
            detection_count=1,
            inference_timestamp=now,
            is_active=True,
            is_stale=False,
            box=(100, 100, 200, 250),
        )

    obs = processor.to_observation()
    assert obs is not None
    assert isinstance(obs, Observation)
    assert obs.source_id == "live_camera"
    assert obs.source_type == "live_camera"
    assert obs.label == "bottle"
    assert obs.confidence == 0.88
    assert round(obs.uncertainty, 2) == 0.12
    assert obs.relative_urgency == "approaching"
    assert obs.information_age >= 0.0
    processor.shutdown()


def test_live_camera_urgency_trend_mapping() -> None:
    processor = LiveCameraProcessor(model_mode="classroom", image_size=320)

    trend_to_urgency = {
        "receding": "low",
        "stable": "low",
        "insufficient_data": "low",
        "growing": "approaching",
        "rapid_growth": "critical",
    }

    now = time.time()
    for trend, expected_urgency in trend_to_urgency.items():
        with processor._lock:
            processor._state = LiveCameraState(
                label="book",
                raw_confidence=0.90,
                confidence=0.90,
                area_ratio=0.05,
                trend=trend,  # type: ignore[arg-type]
                fps=15.0,
                detection_count=1,
                inference_timestamp=now,
                is_active=True,
                is_stale=False,
            )
        obs = processor.to_observation()
        assert obs is not None
        assert obs.relative_urgency == expected_urgency

    processor.shutdown()


def test_live_camera_no_disk_output_leak(
    dummy_bgr_frame: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure no video files, screenshots, or dumps are written to disk."""
    monkeypatch.chdir(tmp_path)
    processor = LiveCameraProcessor(model_mode="classroom", image_size=320)

    for _ in range(5):
        processor.process_frame(dummy_bgr_frame)

    created_files = list(tmp_path.iterdir())
    assert len(created_files) == 0
    processor.shutdown()


def test_live_camera_reset() -> None:
    processor = LiveCameraProcessor(model_mode="classroom", image_size=320)
    with processor._lock:
        processor._state = LiveCameraState(
            label="bottle",
            raw_confidence=0.85,
            confidence=0.85,
            is_active=True,
            detection_count=1,
        )

    processor.reset()
    state = processor.get_state()
    assert state.label == "NO OBJECT DETECTED"
    assert not state.is_active
    assert state.detection_count == 0
    processor.shutdown()


def test_live_camera_thread_safety(dummy_bgr_frame: np.ndarray) -> None:
    """Verify thread-safe reading of get_state while frame processing runs concurrently."""
    processor = LiveCameraProcessor(model_mode="classroom", image_size=320)
    stop_event = threading.Event()
    read_states: list[LiveCameraState] = []

    def reader() -> None:
        while not stop_event.is_set():
            st = processor.get_state()
            read_states.append(st)
            time.sleep(0.005)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    for _ in range(10):
        processor.process_frame(dummy_bgr_frame)
        time.sleep(0.01)

    stop_event.set()
    reader_thread.join(timeout=2.0)

    assert len(read_states) > 0
    assert all(isinstance(s, LiveCameraState) for s in read_states)
    processor.shutdown()


def test_classroom_safe_labels() -> None:
    """Verify default allowed classroom labels exclude persons and weapons."""
    assert "bottle" in CLASSROOM_SAFE_LABELS
    assert "cell phone" in CLASSROOM_SAFE_LABELS
    assert "book" in CLASSROOM_SAFE_LABELS
    assert "cup" in CLASSROOM_SAFE_LABELS
    assert "backpack" in CLASSROOM_SAFE_LABELS
    assert "person" not in CLASSROOM_SAFE_LABELS
    assert "face" not in CLASSROOM_SAFE_LABELS
