from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SplitName = Literal["train", "validation", "test"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "signal"
SPLIT_DIR = PROJECT_ROOT / "data" / "interim" / "signal"


class RadioMLDataset(Dataset):
    def __init__(self, split: SplitName) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown dataset split: {split}")

        self.signals = np.load(
            PROCESSED_DIR / "signals.npy",
            mmap_mode="r",
        )
        self.labels = np.load(
            PROCESSED_DIR / "labels.npy",
            mmap_mode="r",
        )
        self.snrs = np.load(
            PROCESSED_DIR / "snrs.npy",
            mmap_mode="r",
        )
        self.indices = np.load(
            SPLIT_DIR / f"{split}_indices.npy",
        )

        with (PROCESSED_DIR / "metadata.json").open(
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        self.classes = self.metadata["classes"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        item: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dataset_index = int(self.indices[item])

        signal = np.array(
            self.signals[dataset_index],
            dtype=np.float32,
            copy=True,
        )
        label = int(self.labels[dataset_index])
        snr = float(self.snrs[dataset_index])

        return (
            torch.from_numpy(signal),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(snr, dtype=torch.float32),
        )


def create_dataloader(
    split: SplitName,
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    dataset = RadioMLDataset(split)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )
