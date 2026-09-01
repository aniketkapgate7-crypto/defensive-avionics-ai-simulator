# Dataset Setup

Datasets are intentionally excluded from Git and from the starter archive.

## Signal dataset

Use a legally obtained copy of RadioML 2016.10A and place the trusted extracted
file here:

```text
data/raw/radioml/RML2016.10a_dict.pkl
```

Important: RadioML 2016.10A contains communication-modulation classes, not
labels for real radar or missile systems. The project must report the task as
**automatic modulation classification**.

Python pickle files can execute unsafe content while loading. Only load the
dataset file if it came from a trusted, verified source. Do not accept unknown
pickle files from strangers.

## Vision dataset

Use self-created synthetic frames or properly licensed images. Keep the raw
YOLO-format data in:

```text
data/raw/vision/images/
data/raw/vision/labels/
```

Create train, validation, and test splits under `data/processed/vision/`.
Record image licenses or generation details in a dataset card before training.

## Dataset card checklist

- Source and license
- Creation or collection date
- Class definitions
- Number of examples per class
- Duplicate-removal method
- Train/validation/test split method
- Known biases and limitations
- Prohibited uses

## Do not commit

- Raw archives or extracted datasets
- Personally identifying media
- API keys or account credentials
- Unlicensed images
- Model checkpoints larger than the repository host permits
