"""Train YOLO on the safe synthetic aerial-object dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    """Create training command-line options."""

    parser = argparse.ArgumentParser(description="Train a synthetic classroom object detector.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/vision_synthetic/dataset.yaml"),
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/vision"),
    )
    parser.add_argument(
        "--run-name",
        default="synthetic_yolo",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/vision/synthetic_yolo_best.pt"),
    )
    return parser


def main() -> int:
    """Train YOLO and preserve the best checkpoint."""

    arguments = build_parser().parse_args()

    if not arguments.data.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {arguments.data}")

    if arguments.epochs <= 0:
        raise ValueError("epochs must be positive")

    if arguments.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    if arguments.image_size <= 0:
        raise ValueError("image-size must be positive")

    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    print("Starting synthetic YOLO training")
    print("Dataset:", arguments.data.resolve())
    print("Device:", arguments.device)
    print("Epochs:", arguments.epochs)

    model = YOLO(arguments.model)
    model.train(
        data=str(arguments.data.resolve()),
        epochs=arguments.epochs,
        imgsz=arguments.image_size,
        batch=arguments.batch_size,
        device=arguments.device,
        workers=arguments.workers,
        project=str(output_directory),
        name=arguments.run_name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
    )

    run_directory = output_directory / arguments.run_name
    best_checkpoint = run_directory / "weights" / "best.pt"

    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint was not created: {best_checkpoint}")

    arguments.model_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(best_checkpoint, arguments.model_output)

    metadata = {
        "dataset": str(arguments.data.resolve()),
        "base_model": arguments.model,
        "epochs": arguments.epochs,
        "image_size": arguments.image_size,
        "batch_size": arguments.batch_size,
        "device": arguments.device,
        "best_model": str(arguments.model_output.resolve()),
        "run_directory": str(run_directory),
        "scope": "synthetic classroom demonstration",
    }

    report_path = Path("outputs/reports/vision_training_metadata.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nVision training completed")
    print("Best model:", arguments.model_output.resolve())
    print("Training results:", run_directory)
    print("Metadata:", report_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
