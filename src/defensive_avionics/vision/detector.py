"""Generic object detection for synthetic or prerecorded sky footage and classroom objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_SKY_LABELS = ("airplane", "bird", "kite")


@dataclass(frozen=True, slots=True)
class Detection:
    """One normalized object-detection result."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]
    area_ratio: float


class GenericSkyDetector:
    """Run a lightweight YOLO detector on individual video frames."""

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        confidence: float = 0.20,
        image_size: int = 320,
        device: str = "cpu",
        allowed_labels: tuple[str, ...] = DEFAULT_SKY_LABELS,
    ) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.allowed_labels = set(allowed_labels)
        self.allowed_class_ids: list[int] | None = None

        self._resolve_class_ids()

    def _resolve_class_ids(self) -> None:
        """Dynamically resolve allowed class names to model class IDs."""
        if not self.allowed_labels:
            self.allowed_class_ids = None
            return

        name_to_id = {name: cls_id for cls_id, name in self.model.names.items()}
        self.allowed_class_ids = [
            name_to_id[lbl] for lbl in self.allowed_labels if lbl in name_to_id
        ]
        if not self.allowed_class_ids:
            print(
                f"Warning: None of allowed labels {self.allowed_labels} found in model names: "
                f"{list(self.model.names.values())[:10]}"
            )

    def set_confidence(self, confidence: float) -> None:
        """Update confidence threshold dynamically."""
        if 0.0 < confidence <= 1.0:
            self.confidence = confidence

    def set_allowed_labels(self, allowed_labels: tuple[str, ...]) -> None:
        """Update allowed labels and re-resolve class IDs."""
        self.allowed_labels = set(allowed_labels)
        self._resolve_class_ids()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return filtered detections for one BGR image."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a three-channel BGR image")

        height, width = frame.shape[:2]
        frame_area = max(height * width, 1)

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            classes=self.allowed_class_ids,
            max_det=10,
            augment=False,
            verbose=False,
        )

        detections: list[Detection] = []
        boxes = results[0].boxes

        if boxes is None:
            return detections

        for result_box in boxes:
            class_id = int(result_box.cls[0].item())
            label = str(self.model.names[class_id])

            if self.allowed_labels and label not in self.allowed_labels:
                continue

            coordinates = result_box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = (int(round(value)) for value in coordinates)

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            box_area = (x2 - x1) * (y2 - y1)

            detections.append(
                Detection(
                    label=label,
                    confidence=float(result_box.conf[0].item()),
                    box=(x1, y1, x2, y2),
                    area_ratio=box_area / frame_area,
                )
            )

        return detections

    @staticmethod
    def annotate(
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Draw detection boxes on a copy of the frame."""
        annotated = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            label = f"{detection.label.upper()} {detection.confidence:.0%}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 215, 255), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated
