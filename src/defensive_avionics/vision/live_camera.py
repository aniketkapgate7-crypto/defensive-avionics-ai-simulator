"""Real-time live camera processing for classroom demonstration and sensor fusion.

Provides a thread-safe WebRTC callback processor with a dedicated background inference
worker, queue-drop frame management, IoU tracking, coordinate smoothing,
and exports safe abstract observations without recording or storing frames.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from defensive_avionics.fusion.engine import Observation, UrgencyLevel
from defensive_avionics.vision.approach import ApproachEstimate, ExpansionTracker, VisualTrend
from defensive_avionics.vision.detector import Detection, GenericSkyDetector

if TYPE_CHECKING:
    import av

CLASSROOM_SAFE_LABELS: tuple[str, ...] = (
    "bottle",
    "cell phone",
    "book",
    "cup",
    "backpack",
)

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "bottle": (255, 180, 0),       # Azure / Cyan-Blue (BGR)
    "cell phone": (0, 215, 255),   # Amber / Gold (BGR)
    "book": (200, 100, 255),       # Pink / Lavender (BGR)
    "cup": (50, 220, 100),         # Spring Green (BGR)
    "backpack": (0, 140, 255),     # Tangerine / Orange (BGR)
}
DEFAULT_CLASS_COLOR: tuple[int, int, int] = (0, 215, 255)


def get_class_color(label: str) -> tuple[int, int, int]:
    """Return a consistent BGR color for a given object class."""
    return CLASS_COLORS.get(label.lower(), DEFAULT_CLASS_COLOR)


CameraModelMode = Literal["classroom", "synthetic"]

BANNER_TEXT = "LOCAL PROCESSING - NO RECORDING - NO PHYSICAL RANGE ESTIMATE"


def compute_box_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(0, (xb2 - xb1) * (yb2 - yb1))
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@dataclass(frozen=True, slots=True)
class LiveCameraState:
    """Thread-safe snapshot of live camera detection state."""

    seq: int = 0
    label: str = "NO OBJECT DETECTED"
    raw_confidence: float = 0.0
    confidence: float = 0.0
    area_ratio: float = 0.0
    relative_growth: float = 0.0
    trend: VisualTrend = "insufficient_data"
    fps: float = 0.0
    inference_ms: float = 0.0
    result_age_ms: float = 0.0
    inference_timestamp: float = 0.0
    capture_timestamp: float = 0.0
    detection_count: int = 0
    is_active: bool = False
    is_stale: bool = False
    box: tuple[int, int, int, int] | None = None
    all_detections: tuple[Detection, ...] = ()


class LiveCameraProcessor:
    """Thread-safe frame processor with asynchronous background inference worker."""

    def __init__(
        self,
        model_mode: CameraModelMode = "classroom",
        confidence: float = 0.20,
        image_size: int = 320,
        device: str = "cpu",
        window_size: int = 16,
        minimum_samples: int = 6,
        min_duration_sec: float = 0.40,
        stable_tolerance: float = 0.04,
        rapid_growth_threshold: float = 0.18,
        hysteresis_count: int = 2,
        max_detections: int = 10,
    ) -> None:
        if not (1 <= max_detections <= 20):
            raise ValueError("max_detections must be between 1 and 20")
        self.model_mode = model_mode
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.max_detections = max_detections

        self.tracker = ExpansionTracker(
            window_size=window_size,
            minimum_samples=minimum_samples,
            min_duration_sec=min_duration_sec,
            stable_tolerance=stable_tolerance,
            rapid_growth_threshold=rapid_growth_threshold,
            hysteresis_count=hysteresis_count,
        )
        self.detector: GenericSkyDetector | None = None
        self._lock = threading.Lock()

        # Thread-safe state
        self._state = LiveCameraState()
        self._last_detections: list[Detection] = []
        self._last_estimate = ApproachEstimate(
            trend="insufficient_data", relative_growth=0.0, sample_count=0
        )

        # Stable tracking & coordinate smoothing state
        self._tracked_box: tuple[int, int, int, int] | None = None
        self._smoothed_box: list[float] | None = None
        self._tracked_label: str | None = None
        self._missing_count: int = 0
        self._smoothed_conf: float = 0.0

        # Performance & timing
        self._frame_count: int = 0
        self._last_perf_time: float = 0.0
        self._smoothed_fps: float = 0.0

        # Latest-frame background inference worker queue
        self._input_queue: queue.Queue[
            tuple[int, float, float, np.ndarray]
        ] = queue.Queue(maxsize=1)
        self._worker_thread: threading.Thread | None = None
        self._worker_running: bool = False

        self._init_detector()
        self._start_worker()

    def _init_detector(self) -> None:
        """Instantiate detector based on the selected model mode."""
        try:
            project_root = Path(__file__).resolve().parents[3]
            coco_path = project_root / "yolov8n.pt"
            synth_path = project_root / "models" / "vision" / "synthetic_yolo_best.pt"

            if self.model_mode == "classroom":
                model_file = str(coco_path) if coco_path.is_file() else "yolov8n.pt"
                allowed = CLASSROOM_SAFE_LABELS
            else:
                model_file = str(synth_path) if synth_path.is_file() else str(coco_path)
                allowed = ()

            self.detector = GenericSkyDetector(
                model_path=model_file,
                confidence=self.confidence,
                image_size=self.image_size,
                device=self.device,
                allowed_labels=allowed,
                max_detections=self.max_detections,
            )
        except Exception as exc:
            print(f"Warning: Failed to load camera detector: {exc}")
            self.detector = None

    def _start_worker(self) -> None:
        """Start background inference worker thread if not already running."""
        if not self._worker_running:
            self._worker_running = True
            self._worker_thread = threading.Thread(
                target=self._inference_worker_loop,
                daemon=True,
                name="LiveCameraInferenceWorker",
            )
            self._worker_thread.start()

    def _inference_worker_loop(self) -> None:
        """Continuously process the newest available video frame in background."""
        while self._worker_running:
            try:
                frame_seq, t_perf_queued, t_wall_queued, frame = self._input_queue.get(
                    timeout=0.1
                )
            except queue.Empty:
                continue

            if frame is None or self.detector is None:
                continue

            t_infer_start = time.perf_counter()
            try:
                detections = self.detector.detect(frame)
            except Exception as exc:
                print(f"Inference error in worker: {exc}")
                detections = []
            t_infer_end = time.perf_counter()
            inference_duration_ms = (t_infer_end - t_infer_start) * 1000.0
            t_completed_wall = time.time()

            height, width = frame.shape[:2]
            frame_area = max(height * width, 1)

            # Associate primary object and smooth coordinates
            selected = self._select_stable_object(detections)

            if selected is not None:
                raw_conf = selected.confidence
                self._smoothed_conf = (
                    0.30 * raw_conf + 0.70 * self._smoothed_conf
                    if self._smoothed_conf > 0.0
                    else raw_conf
                )

                # Exponential smoothing of bounding box coordinates (alpha=0.65)
                cur_box = selected.box
                if self._smoothed_box is None:
                    self._smoothed_box = [float(v) for v in cur_box]
                else:
                    alpha = 0.65
                    self._smoothed_box = [
                        alpha * float(c) + (1.0 - alpha) * p
                        for c, p in zip(cur_box, self._smoothed_box, strict=True)
                    ]

                clamped_box = (
                    max(0, min(int(round(self._smoothed_box[0])), width - 1)),
                    max(0, min(int(round(self._smoothed_box[1])), height - 1)),
                    max(1, min(int(round(self._smoothed_box[2])), width)),
                    max(1, min(int(round(self._smoothed_box[3])), height)),
                )

                calc_area_ratio = max(
                    1e-6,
                    float((clamped_box[2] - clamped_box[0]) * (clamped_box[3] - clamped_box[1]))
                    / float(frame_area),
                )
                estimate = self.tracker.update(calc_area_ratio, timestamp=t_completed_wall)

                new_state = LiveCameraState(
                    seq=frame_seq,
                    label=selected.label,
                    raw_confidence=raw_conf,
                    confidence=self._smoothed_conf,
                    area_ratio=calc_area_ratio,
                    relative_growth=estimate.relative_growth,
                    trend=estimate.trend,
                    fps=self._smoothed_fps,
                    inference_ms=inference_duration_ms,
                    result_age_ms=0.0,
                    inference_timestamp=t_completed_wall,
                    capture_timestamp=t_wall_queued,
                    detection_count=len(detections),
                    is_active=True,
                    is_stale=False,
                    box=clamped_box,
                    all_detections=tuple(detections),
                )
            else:
                estimate = self.tracker.update(None, timestamp=t_completed_wall)
                self._smoothed_box = None
                self._smoothed_conf = 0.0
                new_state = LiveCameraState(
                    seq=frame_seq,
                    label="NO OBJECT DETECTED",
                    raw_confidence=0.0,
                    confidence=0.0,
                    area_ratio=0.0,
                    relative_growth=0.0,
                    trend="insufficient_data",
                    fps=self._smoothed_fps,
                    inference_ms=inference_duration_ms,
                    result_age_ms=0.0,
                    inference_timestamp=t_completed_wall,
                    capture_timestamp=t_wall_queued,
                    detection_count=0,
                    is_active=True,
                    is_stale=False,
                    box=None,
                    all_detections=(),
                )

            with self._lock:
                self._state = new_state
                self._last_detections = detections
                self._last_estimate = estimate

    def set_model_mode(self, model_mode: CameraModelMode) -> None:
        """Switch between classroom objects and synthetic geometric models."""
        with self._lock:
            if self.model_mode != model_mode:
                self.model_mode = model_mode
                self._init_detector()
                self.reset()

    def set_confidence(self, confidence: float) -> None:
        """Update detection confidence threshold dynamically."""
        if 0.0 < confidence <= 1.0:
            self.confidence = confidence
            if self.detector is not None:
                self.detector.set_confidence(confidence)

    def set_max_detections(self, max_detections: int) -> None:
        """Update max detections dynamically."""
        if not (1 <= max_detections <= 20):
            raise ValueError("max_detections must be between 1 and 20")
        with self._lock:
            self.max_detections = max_detections
            if self.detector is not None:
                self.detector.set_max_detections(max_detections)

    def reset(self) -> None:
        """Reset internal tracking, associations, and state."""
        with self._lock:
            self.tracker.reset()
            self._last_detections.clear()
            self._last_estimate = ApproachEstimate(
                trend="insufficient_data", relative_growth=0.0, sample_count=0
            )
            self._tracked_box = None
            self._smoothed_box = None
            self._tracked_label = None
            self._missing_count = 0
            self._smoothed_conf = 0.0
            self._frame_count = 0
            self._last_perf_time = 0.0
            self._smoothed_fps = 0.0
            self._state = LiveCameraState()
            # Clear input queue
            try:
                while True:
                    self._input_queue.get_nowait()
            except queue.Empty:
                pass

    def shutdown(self) -> None:
        """Stop worker thread cleanly."""
        self._worker_running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)

    def get_state(self) -> LiveCameraState:
        """Return the latest thread-safe camera state with updated result age."""
        with self._lock:
            st = self._state
            active = st.is_active or (self._frame_count > 0)
            if st.inference_timestamp > 0.0:
                age_ms = (time.time() - st.inference_timestamp) * 1000.0
                is_stale = age_ms > 500.0
                return LiveCameraState(
                    seq=st.seq,
                    label=st.label,
                    raw_confidence=st.raw_confidence,
                    confidence=st.confidence,
                    area_ratio=st.area_ratio,
                    relative_growth=st.relative_growth,
                    trend=st.trend,
                    fps=self._smoothed_fps,
                    inference_ms=st.inference_ms,
                    result_age_ms=age_ms,
                    inference_timestamp=st.inference_timestamp,
                    capture_timestamp=st.capture_timestamp,
                    detection_count=st.detection_count,
                    is_active=active,
                    is_stale=is_stale,
                    box=st.box,
                    all_detections=st.all_detections,
                )
            if active and not st.is_active:
                return LiveCameraState(
                    seq=st.seq,
                    label=st.label,
                    raw_confidence=st.raw_confidence,
                    confidence=st.confidence,
                    area_ratio=st.area_ratio,
                    relative_growth=st.relative_growth,
                    trend=st.trend,
                    fps=self._smoothed_fps,
                    inference_ms=st.inference_ms,
                    result_age_ms=st.result_age_ms,
                    inference_timestamp=st.inference_timestamp,
                    capture_timestamp=st.capture_timestamp,
                    detection_count=st.detection_count,
                    is_active=True,
                    is_stale=st.is_stale,
                    box=st.box,
                    all_detections=st.all_detections,
                )
            return st

    def to_observation(self) -> Observation | None:
        """Convert the latest fresh camera state into a typed Sensor Fusion observation."""
        with self._lock:
            state = self._state
            if (
                not state.is_active
                or state.detection_count == 0
                or state.confidence <= 0.0
                or state.is_stale
            ):
                return None

            now_wall = time.time()
            age_ms = (
                (now_wall - state.inference_timestamp) * 1000.0
                if state.inference_timestamp > 0
                else 0.0
            )
            if age_ms > 500.0:
                return None

            urgency_map: dict[VisualTrend, UrgencyLevel] = {
                "insufficient_data": "low",
                "receding": "low",
                "stable": "low",
                "growing": "approaching",
                "rapid_growth": "critical",
            }
            urgency = urgency_map.get(state.trend, "low")
            age_sec = max(0.0, age_ms / 1000.0)

            return Observation(
                source_id="live_camera",
                source_type="live_camera",
                label=state.label,
                confidence=float(min(1.0, max(0.0, state.confidence))),
                uncertainty=float(min(1.0, max(0.0, 1.0 - state.confidence))),
                relative_urgency=urgency,
                timestamp=state.inference_timestamp,
                information_age=age_sec,
            )

    def _select_stable_object(self, detections: list[Detection]) -> Detection | None:
        """Select and associate a stable primary object across successive frames using IoU."""
        if not detections:
            self._missing_count += 1
            if self._missing_count >= 10:
                self._tracked_box = None
                self._smoothed_box = None
                self._tracked_label = None
                self.tracker.reset()
                self._smoothed_conf = 0.0
                return None
            if self._missing_count <= 5 and self._tracked_box is not None and self._tracked_label:
                x1, y1, x2, y2 = self._tracked_box
                return Detection(
                    label=self._tracked_label,
                    confidence=self._smoothed_conf,
                    box=self._tracked_box,
                    area_ratio=(x2 - x1) * (y2 - y1),
                )
            return None

        # Determine target anchor box from smoothed coordinates or previous tracked box
        target_box: tuple[int, int, int, int] | None = None
        if self._smoothed_box is not None:
            target_box = (
                int(round(self._smoothed_box[0])),
                int(round(self._smoothed_box[1])),
                int(round(self._smoothed_box[2])),
                int(round(self._smoothed_box[3])),
            )
        elif self._tracked_box is not None:
            target_box = self._tracked_box

        # When detections exist, match with existing track if possible using IoU
        if target_box is not None and self._missing_count < 10:
            best_det: Detection | None = None
            best_iou = 0.0
            for det in detections:
                iou = compute_box_iou(det.box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det is not None and best_iou >= 0.20:
                if best_det.label != self._tracked_label:
                    self.tracker.reset()
                self._tracked_box = best_det.box
                self._tracked_label = best_det.label
                self._missing_count = 0
                return best_det

            if self._missing_count <= 5:
                largest = max(detections, key=lambda d: d.area_ratio)
                if largest.area_ratio >= 0.02:
                    self.tracker.reset()
                    self._tracked_box = largest.box
                    self._tracked_label = largest.label
                    self._missing_count = 0
                    return largest

        # Initial selection or genuine object switch: select largest detection
        largest = max(detections, key=lambda d: d.area_ratio)
        if self._tracked_label != largest.label:
            self.tracker.reset()
        self._tracked_box = largest.box
        self._tracked_label = largest.label
        self._missing_count = 0
        return largest

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one WebRTC video frame without blocking: queue newest frame & render HUD."""
        if frame is None or frame.size == 0:
            return frame

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must be a 3-channel BGR image")

        now_perf = time.perf_counter()
        now_wall = time.time()

        if self._last_perf_time > 0.0:
            dt = now_perf - self._last_perf_time
            instant_fps = (1.0 / dt) if dt > 0.0001 else 15.0
            self._smoothed_fps = (
                0.85 * self._smoothed_fps + 0.15 * instant_fps
                if self._smoothed_fps > 0.0
                else instant_fps
            )
        else:
            self._smoothed_fps = 15.0
        self._last_perf_time = now_perf

        self._frame_count += 1
        height, width = frame.shape[:2]

        # Enqueue newest frame for background inference (queue size 1 drop-oldest)
        try:
            if self._input_queue.full():
                self._input_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._input_queue.put_nowait(
                (self._frame_count, now_perf, now_wall, frame.copy())
            )
        except queue.Full:
            pass

        # Grab latest detection state
        with self._lock:
            state = self._state
            all_dets = state.all_detections

        # Calculate result age
        result_age_ms = (
            (now_wall - state.inference_timestamp) * 1000.0
            if state.inference_timestamp > 0.0
            else 0.0
        )
        is_stale = (result_age_ms > 500.0) and (state.inference_timestamp > 0.0)

        # Draw HUD Annotations onto a copy of the frame
        annotated = frame.copy()

        # Trend colour palette (BGR)
        trend_colors: dict[VisualTrend, tuple[int, int, int]] = {
            "insufficient_data": (0, 215, 255),  # Cyan
            "receding": (0, 230, 100),  # Green
            "stable": (240, 245, 255),  # White/Cyan
            "growing": (0, 180, 255),  # Amber
            "rapid_growth": (0, 110, 255),  # Orange
        }

        active_trend_color = (128, 128, 128) if is_stale else trend_colors.get(
            state.trend, (0, 215, 255)
        )

        # Identify primary object in all_dets using IoU with smoothed primary box
        primary_det_idx: int | None = None
        if state.box is not None and all_dets and not is_stale:
            best_iou = 0.0
            best_idx: int | None = None
            for idx, det in enumerate(all_dets):
                iou = compute_box_iou(det.box, state.box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou > 0.05:
                primary_det_idx = best_idx

        # 1. Draw every detection with 2-pixel box, class badge, and raw confidence
        for idx, det in enumerate(all_dets):
            # Primary object gets rendered with smoothed coordinates & primary styling below
            if idx == primary_det_idx and state.box is not None and not is_stale:
                continue

            dx1, dy1, dx2, dy2 = det.box
            dx1 = max(0, min(dx1, width - 1))
            dy1 = max(0, min(dy1, height - 1))
            dx2 = max(dx1 + 1, min(dx2, width))
            dy2 = max(dy1 + 1, min(dy2, height))

            color = (128, 128, 128) if is_stale else get_class_color(det.label)

            # 2-pixel bounding box
            cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), color, 2)

            # Class name and raw confidence above box (e.g. BOOK 82.4%)
            conf_str = f"{det.confidence * 100:.1f}%"
            badge_text = f"{det.label.upper()} {conf_str}"
            badge_w = max(80, len(badge_text) * 8 + 10)
            cv2.rectangle(
                annotated,
                (dx1, max(0, dy1 - 18)),
                (min(width, dx1 + badge_w), dy1),
                (7, 18, 34),
                -1,
            )
            cv2.putText(
                annotated,
                badge_text,
                (dx1 + 3, max(13, dy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

        # 2. Draw Primary Tracked Bounding Box (Thicker trend-colored border & PRIMARY badge)
        if state.box is not None and not is_stale and state.confidence > 0.0:
            bx1, by1, bx2, by2 = state.box
            bx1 = max(0, min(bx1, width - 1))
            by1 = max(0, min(by1, height - 1))
            bx2 = max(bx1 + 1, min(bx2, width))
            by2 = max(by1 + 1, min(by2, height))

            # Thicker trend-coloured border (3-pixel)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), active_trend_color, 3)

            # Primary label badge with PRIMARY tag and raw confidence
            raw_conf_str = f"{state.raw_confidence * 100:.1f}%"
            badge_text = f"PRIMARY \u25b6 {state.label.upper()} {raw_conf_str}"
            badge_w = max(130, len(badge_text) * 8 + 14)
            cv2.rectangle(
                annotated,
                (bx1, max(0, by1 - 22)),
                (min(width, bx1 + badge_w), by1),
                (7, 18, 34),
                -1,
            )
            cv2.line(
                annotated,
                (bx1, max(0, by1 - 22)),
                (min(width, bx1 + badge_w), max(0, by1 - 22)),
                active_trend_color,
                1,
            )
            cv2.putText(
                annotated,
                badge_text,
                (bx1 + 4, max(16, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                active_trend_color,
                1,
                cv2.LINE_AA,
            )
        elif is_stale and state.box is not None:
            # Stale primary box in grey
            bx1, by1, bx2, by2 = state.box
            bx1 = max(0, min(bx1, width - 1))
            by1 = max(0, min(by1, height - 1))
            bx2 = max(bx1 + 1, min(bx2, width))
            by2 = max(by1 + 1, min(by2, height))
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (128, 128, 128), 2)

        # 3. Top-Left HUD Telemetry Card
        trend_disp = "STALE RESULT" if is_stale else state.trend.replace("_", " ").upper()
        det_count_disp = (
            f"{state.detection_count} OBJECTS"
            if state.detection_count > 0
            else "NO OBJECTS DETECTED"
        )
        hud_lines = [
            "LIVE CAMERA // MULTI-OBJECT DEFENSE",
            f"PRIMARY OBJECT: {state.label.upper()}",
            f"DETECTED OBJECTS: {det_count_disp}",
            (
                f"CONFIDENCE: {state.confidence * 100:.1f}%"
                if state.confidence > 0
                else "CONFIDENCE: --"
            ),
            f"FPS: {self._smoothed_fps:04.1f}",
            f"INFERENCE: {state.inference_ms:02.0f} ms",
            f"RESULT AGE: {result_age_ms:02.0f} ms",
            f"TREND: {trend_disp}",
        ]

        card_h = 16 + len(hud_lines) * 16
        card_w = 275
        overlay_bg = annotated.copy()
        cv2.rectangle(overlay_bg, (10, 10), (10 + card_w, 10 + card_h), (5, 11, 20), -1)
        cv2.addWeighted(overlay_bg, 0.78, annotated, 0.22, 0, annotated)
        cv2.rectangle(annotated, (10, 10), (10 + card_w, 10 + card_h), (0, 215, 255), 1)

        for idx, line in enumerate(hud_lines):
            line_color = (0, 240, 255) if idx == 0 else (224, 242, 254)
            if "TREND:" in line:
                line_color = active_trend_color
            cv2.putText(
                annotated,
                line,
                (18, 26 + idx * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                line_color,
                1,
                cv2.LINE_AA,
            )

        # 4. Bottom Safe Disclaimer Banner
        footer_bg = annotated.copy()
        cv2.rectangle(footer_bg, (0, height - 24), (width, height), (5, 11, 20), -1)
        cv2.addWeighted(footer_bg, 0.85, annotated, 0.15, 0, annotated)
        cv2.putText(
            annotated,
            BANNER_TEXT,
            (12, height - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (11, 190, 245),
            1,
            cv2.LINE_AA,
        )

        return annotated

    def video_frame_callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Streamlit WebRTC callback: process frame and return annotated result."""
        import av

        img: np.ndarray = frame.to_ndarray(format="bgr24")
        annotated = self.process_frame(img)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
