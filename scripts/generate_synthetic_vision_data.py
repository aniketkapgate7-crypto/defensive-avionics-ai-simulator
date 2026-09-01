"""Generate a safe synthetic YOLO dataset of geometric aerial objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]


def create_sky_background(
    image_size: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Create a randomized sky-style background."""

    top_color = np.array([190, 120, 55], dtype=np.float32)
    bottom_color = np.array([245, 220, 185], dtype=np.float32)

    blend = np.linspace(0.0, 1.0, image_size, dtype=np.float32)
    gradient = (
        top_color[None, None, :] * (1.0 - blend[:, None, None])
        + bottom_color[None, None, :] * blend[:, None, None]
    )
    image = np.repeat(gradient, image_size, axis=1)

    noise = generator.normal(
        0.0,
        5.0,
        size=(image_size, image_size, 1),
    )
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    cloud_layer = image.copy()
    cloud_count = int(generator.integers(0, 5))

    for _ in range(cloud_count):
        center = (
            int(generator.integers(0, image_size)),
            int(generator.integers(0, image_size)),
        )
        axes = (
            int(generator.integers(15, 55)),
            int(generator.integers(6, 22)),
        )
        cv2.ellipse(
            cloud_layer,
            center,
            axes,
            0,
            0,
            360,
            (245, 245, 245),
            -1,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(cloud_layer, 0.25, image, 0.75, 0)


def draw_geometric_object(
    image: np.ndarray,
    generator: np.random.Generator,
) -> Box:
    """Draw one safe geometric object and return its bounding box."""

    image_size = image.shape[0]
    minimum_half_size = max(7, image_size // 45)
    maximum_half_size = max(minimum_half_size + 1, image_size // 9)

    half_size = int(generator.integers(minimum_half_size, maximum_half_size))
    center_x = int(generator.integers(half_size + 2, image_size - half_size - 2))
    center_y = int(generator.integers(half_size + 2, image_size - half_size - 2))

    x1 = center_x - half_size
    y1 = center_y - half_size
    x2 = center_x + half_size
    y2 = center_y + half_size

    colors = (
        (0, 215, 255),
        (255, 255, 255),
        (65, 65, 65),
        (255, 170, 70),
    )
    color = colors[int(generator.integers(0, len(colors)))]
    shape_id = int(generator.integers(0, 3))

    if shape_id == 0:
        cv2.circle(
            image,
            (center_x, center_y),
            half_size,
            color,
            -1,
            cv2.LINE_AA,
        )
    elif shape_id == 1:
        points = np.array(
            [
                [center_x, y1],
                [x2, center_y],
                [center_x, y2],
                [x1, center_y],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [points], color, cv2.LINE_AA)
    else:
        points = np.array(
            [
                [center_x, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [points], color, cv2.LINE_AA)

    return x1, y1, x2, y2


def format_yolo_label(box: Box, image_size: int) -> str:
    """Convert pixel coordinates into normalized YOLO coordinates."""

    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2.0) / image_size
    y_center = ((y1 + y2) / 2.0) / image_size
    width = (x2 - x1) / image_size
    height = (y2 - y1) / image_size

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def generate_split(
    dataset_root: Path,
    split_name: str,
    image_count: int,
    image_size: int,
    generator: np.random.Generator,
) -> None:
    """Generate one YOLO image and label split."""

    image_directory = dataset_root / "images" / split_name
    label_directory = dataset_root / "labels" / split_name

    image_directory.mkdir(parents=True, exist_ok=True)
    label_directory.mkdir(parents=True, exist_ok=True)

    for index in range(image_count):
        image = create_sky_background(image_size, generator)
        labels: list[str] = []

        is_negative_example = generator.random() < 0.10
        object_count = 0 if is_negative_example else int(generator.integers(1, 4))

        for _ in range(object_count):
            box = draw_geometric_object(image, generator)
            labels.append(format_yolo_label(box, image_size))

        file_stem = f"{split_name}_{index:05d}"
        image_path = image_directory / f"{file_stem}.jpg"
        label_path = label_directory / f"{file_stem}.txt"

        written = cv2.imwrite(
            str(image_path),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        if not written:
            raise RuntimeError(f"Unable to write image: {image_path}")

        label_text = "\n".join(labels)
        if label_text:
            label_text += "\n"

        label_path.write_text(label_text, encoding="utf-8")

        if (index + 1) % 100 == 0 or index + 1 == image_count:
            print(f"{split_name}: {index + 1}/{image_count}")


def build_parser() -> argparse.ArgumentParser:
    """Create command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate a synthetic geometric-object dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/vision_synthetic"),
    )
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    """Generate the complete dataset and configuration."""

    arguments = build_parser().parse_args()

    if (
        min(
            arguments.train_count,
            arguments.val_count,
            arguments.test_count,
        )
        < 0
    ):
        raise ValueError("split counts cannot be negative")

    if arguments.image_size < 64:
        raise ValueError("image-size must be at least 64")

    dataset_root = arguments.output.resolve()
    generator = np.random.default_rng(arguments.seed)

    split_counts = {
        "train": arguments.train_count,
        "val": arguments.val_count,
        "test": arguments.test_count,
    }

    for split_name, image_count in split_counts.items():
        generate_split(
            dataset_root=dataset_root,
            split_name=split_name,
            image_count=image_count,
            image_size=arguments.image_size,
            generator=generator,
        )

    dataset_yaml = (
        f"path: '{dataset_root.as_posix()}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: aerial_object\n"
    )
    (dataset_root / "dataset.yaml").write_text(
        dataset_yaml,
        encoding="utf-8",
    )

    metadata = {
        "dataset_type": "synthetic classroom demonstration",
        "class_names": ["aerial_object"],
        "image_size": arguments.image_size,
        "seed": arguments.seed,
        "split_counts": split_counts,
    }
    (dataset_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSynthetic dataset generated successfully")
    print("Dataset:", dataset_root)
    print("Configuration:", dataset_root / "dataset.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
