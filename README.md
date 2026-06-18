# MedMAS

MedMAS contains local utilities for chest X-ray preprocessing, exploratory analysis, DenseNet121 fine-tuning code, and training visualization.

## Repository Layout

```text
src/preprocess_image/      Preprocess manifest, target labels, patient-wise split, and audit helpers
src/eda_image/             Image EDA, metadata summaries, and quality analysis helpers
src/train_densenet/        Simple DenseNet121 fine-tuning pipeline
src/visualize_densenet/    Basic figures from saved DenseNet training artifacts
scripts/                   CLI entrypoints
tests/                     Unit tests
notebook/                  Analysis and visualization notebooks
```

Large local data, generated artifacts, local plans, virtual environments, and runtime state are intentionally ignored by Git.

## Setup

The base project dependencies are managed with `uv`:

```bash
uv sync
```

DenseNet training requires PyTorch and Torchvision. They are not pinned in `pyproject.toml` because the correct install command depends on CPU/GPU/CUDA setup. Install the matching PyTorch build for your machine before running training.

## Tests

```bash
uv run python -m unittest
```

The DenseNet unit tests avoid running real training and can run without a trained checkpoint.

## Preprocess Artifacts

```bash
uv run python scripts/generate_preprocess_artifacts.py
```

Generated preprocess outputs are written under `artifacts/preprocess/`, which is ignored by Git.

## DenseNet Training

After installing PyTorch/Torchvision manually:

```bash
uv run python scripts/train_densenet121_cnn_head.py --dry-run-smoke
uv run python scripts/train_densenet121_cnn_head.py
```

For GPU training on the sample subset, keep the default lighter input first:

```bash
uv run python scripts/train_densenet121_cnn_head.py \
  --manifest-path artifacts/preprocess/split_manifest.csv \
  --target-labels-path artifacts/preprocess/target_labels.json \
  --output-dir artifacts/training/densenet121 \
  --seed 0 \
  --epochs 20 \
  --batch-size 32 \
  --num-workers 4 \
  --image-size 224 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --warmup-epochs 2 \
  --warmup-start-factor 0.1 \
  --min-lr 1e-6 \
  --threshold 0.5 \
  --device cuda
```

The training CLI defaults to `20` epochs with `2` epochs of linear warmup followed by cosine learning-rate decay. Training uses rare-label balanced sampling by default and light X-ray-safe augmentation. Final validation and test exports tune one threshold per label on the validation split by maximizing per-label F1, and reuse those tuned thresholds for test predictions.

Training outputs are written under:

```text
artifacts/training/densenet121/
```

## Visualization

After a training run has produced saved artifacts:

```bash
uv run python scripts/visualize_densenet121_cnn_head.py
```

Visualization reads `config.json`, `training_history.csv`, predictions, and metrics from the latest run directory, including the learning-rate curve and the saved per-label thresholds.
