from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from train_densenet import label_slug


SPLITS = ["train", "val", "test"]


def create_all_plots(
    *,
    manifest: pd.DataFrame,
    history: pd.DataFrame,
    predictions_test: pd.DataFrame,
    metrics_test: dict[str, Any],
    figures_dir: Path,
    root: Path,
    labels: list[str],
) -> list[Path]:
    paths: list[Path] = []
    paths.extend(create_data_plots(manifest, figures_dir, labels))
    paths.extend(create_training_plots(history, figures_dir))
    paths.extend(create_test_plots(predictions_test, metrics_test, figures_dir, labels))
    paths.extend(create_error_grids(predictions_test, figures_dir, root, labels))
    return paths


def create_data_plots(manifest: pd.DataFrame, figures_dir: Path, labels: list[str]) -> list[Path]:
    return [
        plot_split_counts(manifest, figures_dir),
        plot_label_counts(manifest, figures_dir, labels),
        plot_label_prevalence(manifest, figures_dir, labels),
    ]


def create_training_plots(history: pd.DataFrame, figures_dir: Path) -> list[Path]:
    paths = [plot_loss_curve(history, figures_dir)]
    learning_rate_path = plot_learning_rate_curve(history, figures_dir)
    if learning_rate_path is not None:
        paths.append(learning_rate_path)
    metric_path = plot_validation_metrics(history, figures_dir)
    if metric_path is not None:
        paths.append(metric_path)
    return paths


def create_test_plots(
    predictions_test: pd.DataFrame,
    metrics_test: dict[str, Any],
    figures_dir: Path,
    labels: list[str],
) -> list[Path]:
    return [
        plot_per_label_metrics(metrics_test, figures_dir, labels),
        plot_confusion_matrices(metrics_test, figures_dir, labels),
        plot_probability_histograms(predictions_test, figures_dir, labels),
    ]


def plot_split_counts(manifest: pd.DataFrame, figures_dir: Path) -> Path:
    rows = manifest["split"].value_counts().reindex(SPLITS).fillna(0)
    patients = manifest.groupby("split", observed=False)["Patient ID"].nunique().reindex(SPLITS).fillna(0)
    x = np.arange(len(SPLITS))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, rows.to_numpy(dtype=float), width=0.36, label="images")
    ax.bar(x + 0.18, patients.to_numpy(dtype=float), width=0.36, label="patients")
    ax.set_xticks(x, SPLITS)
    ax.set_ylabel("count")
    ax.set_title("Split counts")
    ax.legend()
    return _save(fig, figures_dir / "00_data" / "split_counts.png")


def plot_label_counts(manifest: pd.DataFrame, figures_dir: Path, labels: list[str]) -> Path:
    counts = manifest.groupby("split", observed=False)[labels].sum().reindex(SPLITS).fillna(0)
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, split in zip([-width, 0, width], SPLITS, strict=True):
        ax.bar(x + offset, counts.loc[split].to_numpy(dtype=float), width=width, label=split)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("positive images")
    ax.set_title("Positive labels by split")
    ax.legend()
    return _save(fig, figures_dir / "00_data" / "label_counts.png")


def plot_label_prevalence(manifest: pd.DataFrame, figures_dir: Path, labels: list[str]) -> Path:
    counts = manifest.groupby("split", observed=False)[labels].sum().reindex(SPLITS).fillna(0)
    split_rows = manifest["split"].value_counts().reindex(SPLITS).fillna(0)
    prevalence = counts.div(split_rows.replace(0, np.nan), axis=0) * 100.0
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, split in zip([-width, 0, width], SPLITS, strict=True):
        ax.bar(x + offset, prevalence.loc[split].to_numpy(dtype=float), width=width, label=split)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("prevalence (%)")
    ax.set_title("Label prevalence by split")
    ax.legend()
    return _save(fig, figures_dir / "00_data" / "label_prevalence.png")


def plot_loss_curve(history: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="train")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.legend()
    return _save(fig, figures_dir / "01_training" / "loss_curve.png")


def plot_learning_rate_curve(history: pd.DataFrame, figures_dir: Path) -> Path | None:
    if "learning_rate" not in history.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["learning_rate"], marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    return _save(fig, figures_dir / "01_training" / "learning_rate_curve.png")


def plot_validation_metrics(history: pd.DataFrame, figures_dir: Path) -> Path | None:
    columns = [
        "val_macro_average_precision",
        "val_macro_auroc",
        "val_macro_f1",
    ]
    available = [column for column in columns if column in history.columns]
    if not available:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for column in available:
        ax.plot(history["epoch"], history[column], marker="o", label=column.removeprefix("val_"))
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.set_ylim(0, 1)
    ax.set_title("Validation metrics")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir / "01_training" / "validation_metrics.png")


def plot_per_label_metrics(metrics: dict[str, Any], figures_dir: Path, labels: list[str]) -> Path:
    per_label = metrics.get("per_label", {})
    metric_names = ["average_precision", "auroc", "f1"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, metric_name in zip([-width, 0, width], metric_names, strict=True):
        values = [_metric_value(per_label.get(label, {}).get(metric_name)) for label in labels]
        ax.bar(x + offset, values, width=width, label=metric_name)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Test per-label metrics")
    ax.legend()
    return _save(fig, figures_dir / "03_test" / "per_label_metrics.png")


def plot_confusion_matrices(metrics: dict[str, Any], figures_dir: Path, labels: list[str]) -> Path:
    per_label = metrics.get("per_label", {})
    n_cols = 3
    n_rows = int(np.ceil(len(labels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, max(3.2, n_rows * 3.2)))
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")
    for ax, label in zip(axes_array, labels, strict=False):
        item = per_label.get(label, {})
        matrix = np.array([[item.get("tn", 0), item.get("fp", 0)], [item.get("fn", 0), item.get("tp", 0)]])
        ax.axis("on")
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(label)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center")
    return _save(fig, figures_dir / "03_test" / "confusion_matrices.png")


def plot_probability_histograms(predictions: pd.DataFrame, figures_dir: Path, labels: list[str]) -> Path:
    n_cols = 3
    n_rows = int(np.ceil(len(labels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, max(3.2, n_rows * 3.2)), sharex=True, sharey=False)
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")
    for ax, label in zip(axes_array, labels, strict=False):
        slug = label_slug(label)
        true_col = f"true_{slug}"
        prob_col = f"prob_{slug}"
        threshold = prediction_threshold(predictions, label)
        ax.axis("on")
        ax.hist(predictions.loc[predictions[true_col] == 0, prob_col], bins=20, alpha=0.65, label="true 0")
        ax.hist(predictions.loc[predictions[true_col] == 1, prob_col], bins=20, alpha=0.65, label="true 1")
        ax.axvline(threshold, color="0.2", linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)
    return _save(fig, figures_dir / "03_test" / "probability_histograms.png")


def create_error_grids(
    predictions: pd.DataFrame,
    figures_dir: Path,
    root: Path,
    labels: list[str],
    max_images_per_label: int = 12,
) -> list[Path]:
    paths: list[Path] = []
    for label in labels:
        slug = label_slug(label)
        true_col = f"true_{slug}"
        pred_col = f"pred_{slug}"
        prob_col = f"prob_{slug}"
        false_positives = predictions.loc[(predictions[true_col] == 0) & (predictions[pred_col] == 1)].sort_values(
            prob_col,
            ascending=False,
        )
        false_negatives = predictions.loc[(predictions[true_col] == 1) & (predictions[pred_col] == 0)].sort_values(
            prob_col,
            ascending=True,
        )
        paths.append(
            plot_image_grid(
                false_positives,
                root=root,
                output_path=figures_dir / "05_errors" / f"false_positives_{slug}.png",
                title=f"{label} false positives",
                max_images=max_images_per_label,
            )
        )
        paths.append(
            plot_image_grid(
                false_negatives,
                root=root,
                output_path=figures_dir / "05_errors" / f"false_negatives_{slug}.png",
                title=f"{label} false negatives",
                max_images=max_images_per_label,
            )
        )
    return paths


def plot_image_grid(
    rows: pd.DataFrame,
    *,
    root: Path,
    output_path: Path,
    title: str,
    max_images: int = 12,
) -> Path:
    sample = rows.head(max_images)
    n_cols = 4
    n_rows = max(1, int(np.ceil(max(len(sample), 1) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")
    if sample.empty:
        axes_array[0].text(0.5, 0.5, "no rows", ha="center", va="center")
    for ax, (_, row) in zip(axes_array, sample.iterrows(), strict=False):
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = root / image_path
        if image_path.exists():
            with Image.open(image_path) as image:
                ax.imshow(image.convert("L"), cmap="gray")
        else:
            ax.text(0.5, 0.5, "missing image", ha="center", va="center")
        ax.set_title(str(row.get("Image Index", "")), fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    return _save(fig, output_path)


def _metric_value(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value)


def prediction_threshold(predictions: pd.DataFrame, label: str) -> float:
    slug = label_slug(label)
    column = f"threshold_{slug}"
    if column not in predictions.columns:
        return 0.5
    values = predictions[column].dropna().astype(float).unique().tolist()
    if not values:
        return 0.5
    if len(values) > 1:
        raise ValueError(f"Multiple threshold values found for {label!r}: {values}")
    return float(values[0])


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
