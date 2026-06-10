from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_densenet.artifacts import DISEASE_LABELS, label_slug


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_test_per_label_metrics(metrics_test: dict[str, Any], figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or DISEASE_LABELS
    per_label = metrics_test.get("per_label", {})
    metric_names = ["average_precision", "auroc", "precision", "recall", "specificity", "f1"]
    matrix = np.array(
        [
            [np.nan if per_label.get(label, {}).get(metric) is None else float(per_label[label][metric]) for metric in metric_names]
            for label in target_labels
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metric_names)), metric_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(target_labels)), target_labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "NA" if np.isnan(value) else f"{value:.2f}"
            ax.text(col, row, text, ha="center", va="center", color="white" if not np.isnan(value) and value < 0.5 else "black")
    ax.set_title("Test per-disease-label metrics")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, figures_dir / "03_test" / "test_per_label_metrics.png")


def plot_test_macro_micro_summary(metrics_test: dict[str, Any], figures_dir: Path) -> Path:
    keys = [
        "disease_macro_average_precision",
        "disease_macro_auroc",
        "disease_macro_f1",
        "disease_micro_precision",
        "disease_micro_recall",
        "disease_micro_f1",
    ]
    values = [np.nan if metrics_test.get(key) is None else float(metrics_test[key]) for key in keys]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(keys, values)
    ax.set_xlim(0, 1)
    ax.set_xlabel("metric value")
    ax.set_title("Test macro and micro summary")
    return _save(fig, figures_dir / "03_test" / "test_macro_micro_summary.png")


def plot_derived_no_finding_metrics(metrics_test: dict[str, Any], figures_dir: Path) -> Path:
    metrics = metrics_test.get("derived_no_finding_metrics", {})
    keys = ["precision", "recall", "specificity", "f1"]
    values = [np.nan if metrics.get(key) is None else float(metrics[key]) for key in keys]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(keys, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("metric value")
    ax.set_title("Derived No Finding metrics")
    return _save(fig, figures_dir / "03_test" / "derived_no_finding_metrics.png")


def plot_test_confusion_matrices(metrics_test: dict[str, Any], figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or DISEASE_LABELS
    per_label = metrics_test.get("per_label", {})
    include_derived = "derived_no_finding_metrics" in metrics_test
    panel_count = len(target_labels) + (1 if include_derived else 0)
    n_cols = 3
    n_rows = int(np.ceil(panel_count / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, max(3.2, n_rows * 3)))
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")
    panels: list[tuple[str, dict[str, Any]]] = [(label, per_label.get(label, {})) for label in target_labels]
    if include_derived:
        panels.append(("Derived No Finding", metrics_test.get("derived_no_finding_metrics", {})))
    for ax, (label, item) in zip(axes_array, panels, strict=False):
        matrix = np.array([[item.get("tn", 0), item.get("fp", 0)], [item.get("fn", 0), item.get("tp", 0)]], dtype=int)
        ax.axis("on")
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(label)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center")
    return _save(fig, figures_dir / "03_test" / "test_confusion_matrices.png")


def plot_test_label_cardinality_true_vs_predicted(
    predictions_test: pd.DataFrame,
    figures_dir: Path,
    labels: list[str] | None = None,
) -> Path:
    target_labels = labels or DISEASE_LABELS
    true_cols = [f"true_{label_slug(label)}" for label in target_labels]
    pred_cols = [f"pred_{label_slug(label)}" for label in target_labels]
    true_card = predictions_test[true_cols].sum(axis=1).value_counts().sort_index()
    pred_card = predictions_test[pred_cols].sum(axis=1).value_counts().sort_index()
    index = sorted(set(true_card.index.tolist()) | set(pred_card.index.tolist()))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(index))
    ax.bar(x - 0.18, true_card.reindex(index).fillna(0).to_numpy(), width=0.36, label="true")
    ax.bar(x + 0.18, pred_card.reindex(index).fillna(0).to_numpy(), width=0.36, label="pred")
    ax.set_xticks(x, [str(value) for value in index])
    ax.set_xlabel("positive disease labels per image")
    ax.set_ylabel("rows")
    ax.set_title("Test disease-label cardinality")
    ax.legend()
    return _save(fig, figures_dir / "03_test" / "test_label_cardinality_true_vs_predicted.png")


def create_test_plots(
    *,
    predictions_test: pd.DataFrame,
    metrics_test: dict[str, Any],
    figures_dir: Path,
    labels: list[str] | None = None,
) -> list[Path]:
    target_labels = labels or DISEASE_LABELS
    return [
        plot_test_per_label_metrics(metrics_test, figures_dir, target_labels),
        plot_test_macro_micro_summary(metrics_test, figures_dir),
        plot_derived_no_finding_metrics(metrics_test, figures_dir),
        plot_test_confusion_matrices(metrics_test, figures_dir, target_labels),
        plot_test_label_cardinality_true_vs_predicted(predictions_test, figures_dir, target_labels),
    ]
