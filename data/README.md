# Local Data Directory

Large data files are ignored by Git. Preserve only `.gitkeep` placeholders and
dataset documentation in the repository.

- `raw/`: unchanged, trusted source files
- `interim/`: temporary transformations and split manifests
- `processed/`: model-ready arrays, labels, and dataset YAML files

Never overwrite raw data during preprocessing.
