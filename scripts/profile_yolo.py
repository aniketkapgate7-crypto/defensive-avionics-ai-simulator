"""Benchmark script to profile frame conversion, preprocessing, and YOLO inference."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def profile_yolo_pipeline() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "yolov8n.pt"

    print(f"Loading model from {model_path}...")
    t0 = time.perf_counter()
    model = YOLO(str(model_path))
    t_load = time.perf_counter() - t0
    print(f"Model loaded in {t_load * 1000:.2f} ms")

    # Resolve allowed classes
    allowed_labels = ("bottle", "cell phone", "book", "cup", "backpack")
    name_to_id = {name: cls_id for cls_id, name in model.names.items()}
    allowed_ids = [name_to_id[lbl] for lbl in allowed_labels if lbl in name_to_id]
    print(f"Resolved allowed classes {allowed_labels} -> IDs: {allowed_ids}")

    # Create synthetic 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (35, 25, 15)

    # Warmup
    for _ in range(3):
        model.predict(
            source=frame,
            conf=0.20,
            imgsz=320,
            device="cpu",
            classes=allowed_ids,
            max_det=10,
            augment=False,
            verbose=False,
        )

    times_infer = []
    for _ in range(15):
        t_start = time.perf_counter()
        model.predict(
            source=frame,
            conf=0.20,
            imgsz=320,
            device="cpu",
            classes=allowed_ids,
            max_det=10,
            augment=False,
            verbose=False,
        )
        t_end = time.perf_counter()
        times_infer.append((t_end - t_start) * 1000)

    print(f"\nInference Benchmark on CPU (320x320, classes={allowed_ids}):")
    print(f"  Min latency:    {min(times_infer):.2f} ms")
    print(f"  Median latency: {np.median(times_infer):.2f} ms")
    print(f"  Mean latency:   {np.mean(times_infer):.2f} ms")
    print(f"  Max latency:    {max(times_infer):.2f} ms")
    print(f"  Theoretical Max Inference FPS: {1000.0 / np.median(times_infer):.1f} FPS")


if __name__ == "__main__":
    profile_yolo_pipeline()
