from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader

from defensive_avionics.signal.dataset import create_dataloader
from defensive_avionics.signal.model import build_iq_cnn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "signal" / "best_iq_cnn.pt"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RadioML modulation classifier.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    true_labels: list[np.ndarray] = []
    predicted_labels: list[np.ndarray] = []
    snr_values: list[np.ndarray] = []

    model.eval()
    start_time = time.perf_counter()

    with torch.inference_mode():
        for signals, labels, snrs in loader:
            signals = signals.to(device)

            logits = model(signals)
            predictions = logits.argmax(dim=1)

            true_labels.append(labels.numpy())
            predicted_labels.append(predictions.cpu().numpy())
            snr_values.append(snrs.numpy())

    elapsed_seconds = time.perf_counter() - start_time

    return (
        np.concatenate(true_labels),
        np.concatenate(predicted_labels),
        np.concatenate(snr_values),
        elapsed_seconds,
    )


def save_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    classes: list[str],
) -> Path:
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        normalize="true",
    )

    output_path = FIGURE_DIR / "signal_confusion_matrix.png"

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("RadioML Normalized Confusion Matrix")
    plt.xlabel("Predicted modulation")
    plt.ylabel("True modulation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def save_snr_plot(
    accuracy_by_snr: dict[int, float],
) -> Path:
    output_path = FIGURE_DIR / "signal_accuracy_by_snr.png"

    snr_values = sorted(accuracy_by_snr)
    accuracies = [accuracy_by_snr[snr] for snr in snr_values]

    plt.figure(figsize=(10, 6))
    plt.plot(
        snr_values,
        accuracies,
        marker="o",
        color="#00A6C7",
        linewidth=2,
    )
    plt.axvline(
        0,
        color="#E3A008",
        linestyle="--",
        label="0 dB boundary",
    )
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.title("Modulation Classification Accuracy by SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def main() -> int:
    args = parse_args()
    device = select_device(args.device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_loader = create_dataloader(
        "test",
        batch_size=args.batch_size,
        num_workers=0,
    )
    classes = test_loader.dataset.classes

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = build_iq_cnn(
        num_classes=len(classes),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Evaluating complete test split...")
    print("Test samples:", len(test_loader.dataset))
    print("Device:", device)

    (
        true_labels,
        predicted_labels,
        snr_values,
        elapsed_seconds,
    ) = collect_predictions(
        model,
        test_loader,
        device,
    )

    overall_accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )
    overall_macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
    )

    above_zero_mask = snr_values > 0

    above_zero_accuracy = accuracy_score(
        true_labels[above_zero_mask],
        predicted_labels[above_zero_mask],
    )
    above_zero_macro_f1 = f1_score(
        true_labels[above_zero_mask],
        predicted_labels[above_zero_mask],
        average="macro",
    )

    accuracy_by_snr: dict[int, float] = {}

    for snr in sorted(np.unique(snr_values).astype(int)):
        mask = snr_values == snr

        accuracy_by_snr[int(snr)] = accuracy_score(
            true_labels[mask],
            predicted_labels[mask],
        )

    confusion_path = save_confusion_matrix(
        true_labels,
        predicted_labels,
        classes,
    )
    snr_plot_path = save_snr_plot(accuracy_by_snr)

    report_text = classification_report(
        true_labels,
        predicted_labels,
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    report_path = REPORT_DIR / "signal_classification_report.txt"
    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    metrics = {
        "task": "communication modulation classification",
        "test_samples": int(len(true_labels)),
        "overall_accuracy": float(overall_accuracy),
        "overall_macro_f1": float(overall_macro_f1),
        "accuracy_above_0_db": float(above_zero_accuracy),
        "macro_f1_above_0_db": float(above_zero_macro_f1),
        "accuracy_by_snr": {str(snr): float(accuracy) for snr, accuracy in accuracy_by_snr.items()},
        "inference_seconds": float(elapsed_seconds),
        "milliseconds_per_sample": float(elapsed_seconds * 1000 / len(true_labels)),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_accuracy": float(checkpoint["validation_accuracy"]),
    }

    metrics_path = REPORT_DIR / "signal_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nEvaluation completed")
    print("Overall accuracy:", f"{overall_accuracy:.2%}")
    print("Overall macro-F1:", f"{overall_macro_f1:.4f}")
    print(
        "Accuracy above 0 dB:",
        f"{above_zero_accuracy:.2%}",
    )
    print(
        "Macro-F1 above 0 dB:",
        f"{above_zero_macro_f1:.4f}",
    )
    print("Confusion matrix:", confusion_path)
    print("SNR graph:", snr_plot_path)
    print("Metrics:", metrics_path)
    print("Classification report:", report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
