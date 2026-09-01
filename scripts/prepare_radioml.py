import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
RAW_PATH = Path("data/raw/radioml/RML2016.10a_dict.pkl")
PROCESSED_DIR = Path("data/processed/signal")
SPLIT_DIR = Path("data/interim/signal")


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trusted RadioML dataset...")

    with RAW_PATH.open("rb") as file:
        dataset = pickle.load(file, encoding="latin1")

    classes = sorted({key[0] for key in dataset})
    snr_values = sorted({int(key[1]) for key in dataset})
    class_to_id = {name: index for index, name in enumerate(classes)}
    snr_to_id = {value: index for index, value in enumerate(snr_values)}

    total_samples = sum(len(batch) for batch in dataset.values())
    first_batch = next(iter(dataset.values()))
    sample_shape = tuple(first_batch.shape[1:])

    signals = np.empty(
        (total_samples, *sample_shape),
        dtype=np.float32,
    )
    labels = np.empty(total_samples, dtype=np.int64)
    snrs = np.empty(total_samples, dtype=np.int16)
    strata = np.empty(total_samples, dtype=np.int16)

    offset = 0

    for modulation, snr in sorted(
        dataset,
        key=lambda key: (str(key[0]), int(key[1])),
    ):
        batch = np.asarray(
            dataset[(modulation, snr)],
            dtype=np.float32,
        )

        end = offset + len(batch)
        class_id = class_to_id[modulation]

        signals[offset:end] = batch
        labels[offset:end] = class_id
        snrs[offset:end] = int(snr)
        strata[offset:end] = class_id * len(snr_values) + snr_to_id[int(snr)]

        offset = end

    all_indices = np.arange(total_samples)

    train_indices, temporary_indices = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=SEED,
        stratify=strata,
    )

    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=SEED,
        stratify=strata[temporary_indices],
    )

    np.save(PROCESSED_DIR / "signals.npy", signals)
    np.save(PROCESSED_DIR / "labels.npy", labels)
    np.save(PROCESSED_DIR / "snrs.npy", snrs)

    np.save(SPLIT_DIR / "train_indices.npy", train_indices)
    np.save(SPLIT_DIR / "validation_indices.npy", validation_indices)
    np.save(SPLIT_DIR / "test_indices.npy", test_indices)

    metadata = {
        "dataset": "RadioML 2016.10A",
        "task": "communication modulation classification",
        "classes": [str(name) for name in classes],
        "snr_values": snr_values,
        "sample_shape": list(sample_shape),
        "total_samples": total_samples,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_samples": len(test_indices),
        "seed": SEED,
    }

    with (PROCESSED_DIR / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nDataset preparation completed")
    print("Classes:", len(classes))
    print("Total samples:", total_samples)
    print("Training samples:", len(train_indices))
    print("Validation samples:", len(validation_indices))
    print("Test samples:", len(test_indices))
    print("Signal shape:", signals.shape)


if __name__ == "__main__":
    main()
