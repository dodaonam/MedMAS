from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from train_densenet.artifacts import DISEASE_LABELS


SPLITS = ["train", "val", "test"]


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_split_row_counts(manifest: pd.DataFrame, figures_dir: Path) -> Path:
    counts = manifest["split"].value_counts().reindex(SPLITS).fillna(0)
    patients = manifest.groupby("split", observed=False)["Patient ID"].nunique().reindex(SPLITS).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(SPLITS))
    ax.bar(x - 0.18, counts.to_numpy(), width=0.36, label="rows")
    ax.bar(x + 0.18, patients.to_numpy(), width=0.36, label="patients")
    ax.set_xticks(x, SPLITS)
    ax.set_ylabel("count")
    ax.set_title("Split row and patient counts")
    ax.legend()
    return _save(fig, figures_dir / "00_data" / "split_row_counts.png")


def plot_target_positive_counts_by_split(manifest: pd.DataFrame, figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or DISEASE_LABELS
    counts = manifest.groupby("split", observed=False)[target_labels].sum().reindex(SPLITS).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(target_labels))
    width = 0.25
    for offset, split in zip([-width, 0, width], SPLITS, strict=True):
        ax.bar(x + offset, counts.loc[split].to_numpy(dtype=float), width=width, label=split)
    ax.set_xticks(x, target_labels, rotation=30, ha="right")
    ax.set_ylabel("positive images")
    ax.set_title("Disease positive counts by split")
    ax.legend()
    return _save(fig, figures_dir / "00_data" / "target_positive_counts_by_split.png")


def plot_target_prevalence_by_split(manifest: pd.DataFrame, figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or DISEASE_LABELS
    counts = manifest.groupby("split", observed=False)[target_labels].sum().reindex(SPLITS).fillna(0)
    rows = manifest["split"].value_counts().reindex(SPLITS).fillna(0)
    prevalence = counts.div(rows.replace(0, np.nan), axis=0) * 100.0
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(target_labels))
    width = 0.25
    for offset, split in zip([-width, 0, width], SPLITS, strict=True):
        ax.bar(x + offset, prevalence.loc[split].to_numpy(dtype=float), width=width, label=split)
    ax.set_xticks(x, target_labels, rotation=30, ha="right")
    ax.set_ylabel("prevalence (%)")
    ax.set_title("Disease prevalence by split")
    ax.legend()
    return _save(fig, figures_dir / "00_data" / "target_prevalence_by_split.png")


def plot_selected_label_cardinality(manifest: pd.DataFrame, figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or DISEASE_LABELS
    cardinality = manifest[target_labels].sum(axis=1).value_counts().sort_index()
    cardinality = cardinality.reindex(range(0, len(target_labels) + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cardinality.index.astype(str), cardinality.to_numpy())
    ax.set_xlabel("disease labels per image")
    ax.set_ylabel("rows")
    ax.set_title("Disease-label cardinality")
    return _save(fig, figures_dir / "00_data" / "selected_label_cardinality.png")


def plot_target_cooccurrence_heatmap(manifest: pd.DataFrame, figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or DISEASE_LABELS
    values = manifest[target_labels].to_numpy(dtype=int)
    cooccurrence = values.T @ values
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cooccurrence, cmap="Blues")
    ax.set_xticks(np.arange(len(target_labels)), target_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(target_labels)), target_labels)
    for row in range(len(target_labels)):
        for col in range(len(target_labels)):
            ax.text(col, row, str(int(cooccurrence[row, col])), ha="center", va="center", fontsize=8)
    ax.set_title("Disease co-occurrence counts")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, figures_dir / "00_data" / "target_cooccurrence_heatmap.png")


def plot_image_grid_from_paths(
    rows: pd.DataFrame,
    *,
    root: Path,
    output_path: Path,
    title_column: str = "Image Index",
    max_images: int = 12,
) -> Path:
    sample = rows.head(max_images)
    n_cols = 4
    n_rows = max(1, int(np.ceil(len(sample) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")
    for ax, (_, row) in zip(axes_array, sample.iterrows(), strict=False):
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = root / image_path
        if image_path.exists():
            with Image.open(image_path) as image:
                ax.imshow(image.convert("L"), cmap="gray")
        else:
            ax.text(0.5, 0.5, "missing image", ha="center", va="center")
        ax.set_title(str(row.get(title_column, "")), fontsize=8)
        ax.axis("off")
    return _save(fig, output_path)


def _display_tensor(image_tensor: Any) -> np.ndarray:
    from train_densenet.transforms import denormalize_image_tensor

    image = denormalize_image_tensor(image_tensor.detach().cpu())
    return image.permute(1, 2, 0).numpy()


def _label_text(target_values: Any, labels: list[str]) -> str:
    positives = [label for label, value in zip(labels, target_values, strict=True) if int(value) == 1]
    return ", ".join(positives) if positives else "0 disease"


def plot_train_batch_after_transform(
    manifest: pd.DataFrame,
    *,
    root: Path,
    figures_dir: Path,
    labels: list[str] | None = None,
    max_images: int = 12,
) -> Path:
    from train_densenet.dataset import ChestXrayMultiLabelDataset
    from train_densenet.transforms import build_train_transform

    target_labels = labels or DISEASE_LABELS
    dataset = ChestXrayMultiLabelDataset(
        manifest,
        root=root,
        split="train",
        target_labels=target_labels,
        transform=build_train_transform(),
    )
    count = min(max_images, len(dataset))
    n_cols = 4
    n_rows = max(1, int(np.ceil(count / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.2))
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")
    for sample_idx, ax in enumerate(axes_array[:count]):
        image_tensor, target_tensor, metadata = dataset[sample_idx]
        ax.imshow(_display_tensor(image_tensor))
        title = str(metadata.get("Image Index", ""))
        label_text = _label_text(target_tensor.detach().cpu().numpy(), target_labels)
        ax.set_title(f"{title}\n{label_text}", fontsize=8)
        ax.axis("off")
    return _save(fig, figures_dir / "00_data" / "train_batch_after_transform.png")


def plot_augmentation_preview(
    manifest: pd.DataFrame,
    *,
    root: Path,
    figures_dir: Path,
    labels: list[str] | None = None,
    image_count: int = 3,
    augmentations_per_image: int = 3,
) -> Path:
    from train_densenet.transforms import build_eval_transform, build_train_transform

    target_labels = labels or DISEASE_LABELS
    train_rows = manifest.loc[manifest["split"].astype(str) == "train"].head(image_count)
    eval_transform = build_eval_transform()
    train_transform = build_train_transform()
    n_rows = max(1, len(train_rows))
    n_cols = augmentations_per_image + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.1, n_rows * 3.1))
    axes_array = np.asarray(axes).reshape(n_rows, n_cols)
    for row_idx, (_, row) in enumerate(train_rows.iterrows()):
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = root / image_path
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            tensors = [eval_transform(image_rgb), *[train_transform(image_rgb) for _ in range(augmentations_per_image)]]
        labels_text = _label_text([row[label] for label in target_labels], target_labels)
        for col_idx, tensor in enumerate(tensors):
            ax = axes_array[row_idx, col_idx]
            ax.imshow(_display_tensor(tensor))
            title = "eval crop" if col_idx == 0 else f"train aug {col_idx}"
            if col_idx == 0:
                title = f"{row.get('Image Index', '')}\n{labels_text}\n{title}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    return _save(fig, figures_dir / "00_data" / "augmentation_preview.png")


def create_data_plots(
    *,
    manifest: pd.DataFrame,
    loss_config: dict[str, Any],
    root: Path,
    figures_dir: Path,
    labels: list[str] | None = None,
) -> list[Path]:
    _ = loss_config
    target_labels = labels or DISEASE_LABELS
    return [
        plot_split_row_counts(manifest, figures_dir),
        plot_target_positive_counts_by_split(manifest, figures_dir, target_labels),
        plot_target_prevalence_by_split(manifest, figures_dir, target_labels),
        plot_selected_label_cardinality(manifest, figures_dir, target_labels),
        plot_target_cooccurrence_heatmap(manifest, figures_dir, target_labels),
        plot_train_batch_after_transform(manifest, root=root, figures_dir=figures_dir, labels=target_labels),
        plot_augmentation_preview(manifest, root=root, figures_dir=figures_dir, labels=target_labels),
    ]
