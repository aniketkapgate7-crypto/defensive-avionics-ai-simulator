import pickle
from pathlib import Path

DATASET_PATH = Path("data/raw/radioml/RML2016.10a_dict.pkl")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

print("Loading trusted official RadioML dataset...")

with DATASET_PATH.open("rb") as file:
    dataset = pickle.load(file, encoding="latin1")

modulations = sorted({key[0] for key in dataset})
snr_values = sorted({int(key[1]) for key in dataset})
total_samples = sum(len(samples) for samples in dataset.values())

first_key = next(iter(dataset))
first_batch = dataset[first_key]

print("\nDataset loaded successfully")
print("Dictionary entries:", len(dataset))
print("Modulation classes:", modulations)
print("Number of classes:", len(modulations))
print("SNR values:", snr_values)
print("Total samples:", total_samples)
print("Example key:", first_key)
print("Example batch shape:", first_batch.shape)
print("Example data type:", first_batch.dtype)
