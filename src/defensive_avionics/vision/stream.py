"""Run the generic visual approach-warning classroom demo."""

from __future__ import annotations

import argparse
import time

import cv2

from defensive_avionics.vision.approach import ExpansionTracker
from defensive_avionics.vision.detector import (
    DEFAULT_SKY_LABELS,
    GenericSkyDetector,
)


def build_parser() -> argparse.ArgumentParser:
    """Create command-line options."""

    parser = argparse.ArgumentParser(description="Run a generic object-approach demonstration.")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera number such as 0, or a prerecorded video path.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics model path.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence.",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_SKY_LABELS),
        help="Comma-separated labels to display.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, normally cpu for this project.",
    )
    return parser


def parse_source(source_text: str) -> int | str:
    """Convert camera numbers to integers while preserving file paths."""

    return int(source_text) if source_text.isdigit() else source_text


def draw_status(
    frame,
    trend: str,
    frames_per_second: float,
) -> None:
    """Draw the normalized trend and processing speed."""

    readable_trend = trend.replace("_", " ").upper()
    status_text = f"TREND: {readable_trend}"
    fps_text = f"FPS: {frames_per_second:.1f}"

    color = (0, 0, 255) if trend == "rapid_growth" else (0, 215, 255)

    cv2.putText(
        frame,
        status_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        fps_text,
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if trend == "rapid_growth":
        height, width = frame.shape[:2]
        banner_top = max(height - 70, 0)

        cv2.rectangle(
            frame,
            (0, banner_top),
            (width, height),
            (0, 0, 150),
            -1,
        )
        cv2.putText(
            frame,
            "VISUAL APPROACH ALERT - CLASSROOM DEMO",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    """Open the selected video source and run the demo."""

    arguments = build_parser().parse_args()
    labels = tuple(label.strip() for label in arguments.labels.split(",") if label.strip())

    detector = GenericSkyDetector(
        model_path=arguments.model,
        confidence=arguments.confidence,
        device=arguments.device,
        allowed_labels=labels,
    )
    tracker = ExpansionTracker()

    capture = cv2.VideoCapture(parse_source(arguments.source))

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {arguments.source}")

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            detections = detector.detect(frame)
            largest_area = max(item.area_ratio for item in detections) if detections else None
            estimate = tracker.update(largest_area)

            annotated = detector.annotate(frame, detections)

            current_time = time.perf_counter()
            elapsed = max(current_time - previous_time, 1e-9)
            instantaneous_fps = 1.0 / elapsed
            previous_time = current_time

            smoothed_fps = (
                instantaneous_fps
                if smoothed_fps == 0.0
                else (0.9 * smoothed_fps) + (0.1 * instantaneous_fps)
            )

            draw_status(
                annotated,
                estimate.trend,
                smoothed_fps,
            )

            cv2.imshow("Visual Approach Demo", annotated)
            pressed_key = cv2.waitKey(1) & 0xFF

            if pressed_key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
