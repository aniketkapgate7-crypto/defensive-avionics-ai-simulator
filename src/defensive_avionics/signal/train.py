from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from defensive_avionics.common.seed import set_seed
from defensive_avionics.signal.dataset import RadioMLDataset
from defensive_avionics.signal.model import build_iq_cnn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "signal"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RadioML communication-modulation classifier."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
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
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_loader(
    split: str,
    batch_size: int,
    limit: int | None,
    seed: int,
) -> tuple[DataLoader, list[str]]:
    dataset = RadioMLDataset(split)
    selected_dataset = dataset

    if limit is not None and limit < len(dataset):
        generator = torch.Generator().manual_seed(seed)
        selected_indices = torch.randperm(
            len(dataset),
            generator=generator,
        )[:limit].tolist()

        selected_dataset = Subset(
            dataset,
            selected_indices,
        )

    loader_generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )

    return loader, dataset.classes


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for signals, labels, _snrs in loader:
            signals = signals.to(device)
            labels = labels.to(device)

            logits = model(signals)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = select_device(args.device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, classes = create_loader(
        "train",
        args.batch_size,
        args.limit_train,
        args.seed,
    )
    validation_loader, _ = create_loader(
        "validation",
        args.batch_size,
        args.limit_validation,
        args.seed,
    )

    model = build_iq_cnn(
        num_classes=len(classes),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    history: list[dict[str, float | int]] = []
    best_validation_accuracy = 0.0
    checkpoint_path = MODEL_DIR / "best_iq_cnn.pt"

    print("Task: communication modulation classification")
    print("Device:", device)
    print("Classes:", len(classes))
    print("Training examples:", len(train_loader.dataset))
    print("Validation examples:", len(validation_loader.dataset))

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        model.train()

        training_loss = 0.0
        training_correct = 0
        training_examples = 0

        for signals, labels, _snrs in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(signals)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item() * labels.size(0)
            training_correct += (logits.argmax(dim=1) == labels).sum().item()
            training_examples += labels.size(0)

        training_loss /= training_examples
        training_accuracy = training_correct / training_examples

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        elapsed_seconds = time.perf_counter() - start_time

        record = {
            "epoch": epoch,
            "training_loss": training_loss,
            "training_accuracy": training_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "elapsed_seconds": elapsed_seconds,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {training_loss:.4f} | "
            f"train accuracy {training_accuracy:.2%} | "
            f"validation loss {validation_loss:.4f} | "
            f"validation accuracy {validation_accuracy:.2%} | "
            f"time {elapsed_seconds:.1f}s"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                    "configuration": vars(args),
                },
                checkpoint_path,
            )

    history_path = LOG_DIR / "signal_training_history.json"

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print("\nTraining completed")
    print("Best validation accuracy:", f"{best_validation_accuracy:.2%}")
    print("Checkpoint:", checkpoint_path)
    print("History:", history_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
