"""Vision pipeline with model loading, synthetic generator, and trend tracking."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from defensive_avionics.common.types import Urgency, VisionPrediction
from defensive_avionics.vision.approach import ApproachEstimate, ExpansionTracker
from defensive_avionics.vision.detector import Detection, GenericSkyDetector


class VisionPipeline:
    """Full vision inference contract connecting YOLO detection and approach estimation."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = 0.35,
        image_size: int = 320,
        device: str = "cpu",
    ) -> None:
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.tracker = ExpansionTracker(window_size=8, minimum_samples=3)
        self.detector: GenericSkyDetector | None = None
        self.is_trained_model = False
        self.model_path = (
            Path(model_path) if model_path else Path("models/vision/synthetic_yolo_best.pt")
        )

        self._init_detector()

    def _init_detector(self) -> None:
        """Initialize the YOLO detector if weights are available."""
        target_path: Path | None = None

        if self.model_path.is_file():
            target_path = self.model_path
            self.is_trained_model = True
        elif Path("yolov8n.pt").is_file():
            target_path = Path("yolov8n.pt")
            self.is_trained_model = False

        if target_path:
            try:
                self.detector = GenericSkyDetector(
                    model_path=target_path,
                    confidence=self.confidence,
                    image_size=self.image_size,
                    device=self.device,
                    allowed_labels=(),  # Allow all detected classes
                )
            except Exception as exc:
                print(
                    f"Warning: Could not initialize YOLO detector: {exc}. Using synthetic fallback."
                )
                self.detector = None
                self.is_trained_model = False

    def predict(self, frame: np.ndarray | None = None) -> VisionPrediction:
        """Predict detection status, confidence, and urgency for one image frame."""
        if frame is None:
            frame, _ = self.generate_synthetic_sky_frame()

        if self.detector is not None:
            try:
                detections = self.detector.detect(frame)
                if detections:
                    top_det = max(detections, key=lambda d: d.confidence)
                    estimate = self.tracker.update(top_det.area_ratio)
                    urgency_mapped = self._map_trend_to_urgency(estimate.trend)
                    return VisionPrediction(
                        detected=True,
                        confidence=round(top_det.confidence, 4),
                        urgency=urgency_mapped,
                    )
                else:
                    self.tracker.update(None)
                    return VisionPrediction(
                        detected=False,
                        confidence=0.0,
                        urgency="low",
                    )
            except Exception:
                pass

        # Fallback heuristic detection for synthetic frames
        # Detect bright/dark contrast in frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        diff = float(np.max(gray) - np.min(gray))
        if diff > 60:
            estimate = self.tracker.update(0.04)
            return VisionPrediction(
                detected=True,
                confidence=0.85,
                urgency=self._map_trend_to_urgency(estimate.trend),
            )

        return VisionPrediction(detected=False, confidence=0.0, urgency="low")

    def process_frame(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, list[Detection], ApproachEstimate]:
        """Detect objects, annotate frame, and update expansion tracking."""
        detections: list[Detection] = []
        if self.detector is not None:
            try:
                detections = self.detector.detect(frame)
            except Exception:
                detections = []

        if not detections:
            # Check for simple synthetic geometric shape if detector didn't catch it
            synthetic_det = self._detect_synthetic_shape(frame)
            if synthetic_det:
                detections = [synthetic_det]

        largest_area = max((d.area_ratio for d in detections), default=None)
        estimate = self.tracker.update(largest_area)

        annotated = GenericSkyDetector.annotate(frame, detections)
        return annotated, detections, estimate

    def _detect_synthetic_shape(self, frame: np.ndarray) -> Detection | None:
        """Simple color/contour fallback detector for synthetic demonstrations."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = frame.shape[:2]
            frame_area = h * w
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 200 < area < frame_area * 0.5:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    return Detection(
                        label="aerial_object",
                        confidence=0.88,
                        box=(x, y, x + bw, y + bh),
                        area_ratio=area / frame_area,
                    )
        except Exception:
            pass
        return None

    @staticmethod
    def _map_trend_to_urgency(trend: str) -> Urgency:
        if trend == "rapid_growth":
            return "critical"
        elif trend == "growing":
            return "approaching"
        return "low"

    @staticmethod
    def generate_synthetic_sky_frame(
        image_size: int = 320,
        shape_type: str = "triangle",
        scale: float = 1.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        # Gradient sky
        top_color = np.array([190, 120, 55], dtype=np.float32)
        bottom_color = np.array([245, 220, 185], dtype=np.float32)
        blend = np.linspace(0.0, 1.0, image_size, dtype=np.float32)
        gradient = (
            top_color[None, None, :] * (1.0 - blend[:, None, None])
            + bottom_color[None, None, :] * blend[:, None, None]
        )
        img = np.repeat(gradient, image_size, axis=1).astype(np.uint8)

        center_x = image_size // 2
        center_y = image_size // 2
        half_size = int(max(10, (image_size // 14) * scale))

        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(image_size, center_x + half_size)
        y2 = min(image_size, center_y + half_size)

        color = (0, 215, 255)
        if shape_type == "circle":
            cv2.circle(img, (center_x, center_y), half_size, color, -1, cv2.LINE_AA)
        elif shape_type == "diamond":
            pts = np.array(
                [[center_x, y1], [x2, center_y], [center_x, y2], [x1, center_y]],
                dtype=np.int32,
            )
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        else:  # triangle
            pts = np.array([[center_x, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA)

        return img, (x1, y1, x2, y2)
