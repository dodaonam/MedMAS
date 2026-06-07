from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from train_densenet.artifacts import TARGET_LABELS


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
    target_labels = labels or TARGET_LABELS
    counts = manifest.groupby("split", observed=False)[target_labels].sum().reindex(SPLITS).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(target_labels))
    width = 0.25
    for offset, split in zip([-width, 0, width], SPLITS, strict=True):
        ax.bar(x + offset, counts.loc[split].to_numpy(dtype=float), width=width, label=split)
    ax.set_xticks(x, target_labels, rotation=30, ha="right")
    ax.set_ylabel("positive images")
    ax.set_title("Target positive counts by split")
    ax.legend()
    return _save(fig, figures_dir / "00_data" / "target_positive_counts_by_split.png")


def plot_target_prevalence_by_split(manifest: pd.DataFrame, figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or TARGET_LABELS
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
    ax.set_title("Target prevalence by split")
    ax.legend()
    return _save(fig, figures_dir / "00_data" / "target_prevalence_by_split.png")


def plot_train_pos_weight_by_label(
    class_weights: dict[str, Any],
    figures_dir: Path,
    labels: list[str] | None = None,
) -> Path:
    target_labels = labels or TARGET_LABELS
    raw = class_weights.get("raw_train_only_pos_weight", {})
    used = class_weights.get("selected_clipped_pos_weight", {})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(target_labels))
    ax.bar(x - 0.18, [float(raw.get(label, np.nan)) for label in target_labels], width=0.36, label="raw")
    ax.bar(x + 0.18, [float(used.get(label, np.nan)) for label in target_labels], width=0.36, label="used")
    ax.set_xticks(x, target_labels, rotation=30, ha="right")
    ax.set_ylabel("pos_weight")
    ax.set_title("Train-only positive weights")
    ax.legend()
    return _save(fig, figures_dir / "00_data" / "train_pos_weight_by_label.png")


def plot_selected_label_cardinality(manifest: pd.DataFrame, figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or TARGET_LABELS
    cardinality = manifest[target_labels].sum(axis=1).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cardinality.index.astype(str), cardinality.to_numpy())
    ax.set_xlabel("selected labels per image")
    ax.set_ylabel("rows")
    ax.set_title("Selected-label cardinality")
    return _save(fig, figures_dir / "00_data" / "selected_label_cardinality.png")


def plot_target_cooccurrence_heatmap(manifest: pd.DataFrame, figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or TARGET_LABELS
    values = manifest[target_labels].to_numpy(dtype=int)
    cooccurrence = values.T @ values
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cooccurrence, cmap="Blues")
    ax.set_xticks(np.arange(len(target_labels)), target_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(target_labels)), target_labels)
    for row in range(len(target_labels)):
        for col in range(len(target_labels)):
            ax.text(col, row, str(int(cooccurrence[row, col])), ha="center", va="center", fontsize=8)
    ax.set_title("Target co-occurrence counts")
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


def create_data_plots(
    *,
    manifest: pd.DataFrame,
    class_weights: dict[str, Any],
    figures_dir: Path,
    labels: list[str] | None = None,
) -> list[Path]:
    target_labels = labels or TARGET_LABELS
    return [
        plot_split_row_counts(manifest, figures_dir),
        plot_target_positive_counts_by_split(manifest, figures_dir, target_labels),
        plot_target_prevalence_by_split(manifest, figures_dir, target_labels),
        plot_train_pos_weight_by_label(class_weights, figures_dir, target_labels),
        plot_selected_label_cardinality(manifest, figures_dir, target_labels),
        plot_target_cooccurrence_heatmap(manifest, figures_dir, target_labels),
    ]
