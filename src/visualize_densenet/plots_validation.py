from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_densenet.artifacts import TARGET_LABELS, label_slug
from train_densenet.metrics import confusion_counts, metrics_from_counts
from train_densenet.thresholds import threshold_grid, thresholds_by_label


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _pr_curve(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_prob, kind="mergesort")
    sorted_true = y_true[order]
    positives = max(int(np.sum(sorted_true == 1)), 1)
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    return recall, precision


def _roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_prob, kind="mergesort")
    sorted_true = y_true[order]
    positives = max(int(np.sum(sorted_true == 1)), 1)
    negatives = max(int(np.sum(sorted_true == 0)), 1)
    tpr = np.cumsum(sorted_true == 1) / positives
    fpr = np.cumsum(sorted_true == 0) / negatives
    return fpr, tpr


def plot_validation_pr_curves(predictions_val: pd.DataFrame, figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or TARGET_LABELS
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in target_labels:
        slug = label_slug(label)
        recall, precision = _pr_curve(
            predictions_val[f"true_{slug}"].to_numpy(dtype=int),
            predictions_val[f"prob_{slug}"].to_numpy(dtype=float),
        )
        ax.plot(recall, precision, label=label)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Validation precision-recall curves")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir / "02_validation" / "validation_pr_curves.png")


def plot_validation_roc_curves(predictions_val: pd.DataFrame, figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or TARGET_LABELS
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in target_labels:
        slug = label_slug(label)
        fpr, tpr = _roc_curve(
            predictions_val[f"true_{slug}"].to_numpy(dtype=int),
            predictions_val[f"prob_{slug}"].to_numpy(dtype=float),
        )
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Validation ROC curves")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir / "02_validation" / "validation_roc_curves.png")


def plot_threshold_sweep_by_label(predictions_val: pd.DataFrame, figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or TARGET_LABELS
    fig, ax = plt.subplots(figsize=(9, 5))
    grid = threshold_grid()
    for label in target_labels:
        slug = label_slug(label)
        true = predictions_val[f"true_{slug}"].to_numpy(dtype=int)
        prob = predictions_val[f"prob_{slug}"].to_numpy(dtype=float)
        f1_values = []
        for threshold in grid:
            counts = confusion_counts(true, (prob >= threshold).astype(int))
            f1_values.append(metrics_from_counts(counts)["f1"])
        ax.plot(grid, f1_values, label=label)
    ax.set_xlabel("threshold")
    ax.set_ylabel("validation F1")
    ax.set_title("Validation threshold sweep")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir / "02_validation" / "threshold_sweep_by_label.png")


def plot_selected_thresholds(thresholds: dict[str, Any], figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or TARGET_LABELS
    selected = thresholds_by_label(thresholds, target_labels)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(target_labels, [selected[label] for label in target_labels])
    ax.set_ylim(0, 1)
    ax.set_ylabel("threshold")
    ax.set_title("Validation-selected thresholds")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, figures_dir / "02_validation" / "selected_thresholds.png")


def plot_threshold_diagnostics_by_label(thresholds: dict[str, Any], figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or TARGET_LABELS
    per_label = thresholds.get("per_label", {})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    f1 = [float(per_label.get(label, {}).get("f1", np.nan)) for label in target_labels]
    recall = [float(per_label.get(label, {}).get("recall", np.nan)) for label in target_labels]
    precision = [float(per_label.get(label, {}).get("precision", np.nan)) for label in target_labels]
    x = np.arange(len(target_labels))
    ax.bar(x - 0.25, precision, width=0.25, label="precision")
    ax.bar(x, recall, width=0.25, label="recall")
    ax.bar(x + 0.25, f1, width=0.25, label="F1")
    for idx, label in enumerate(target_labels):
        if per_label.get(label, {}).get("threshold_unstable"):
            ax.text(idx, 1.02, "unstable", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x, target_labels, rotation=30, ha="right")
    ax.set_title("Threshold diagnostics at selected thresholds")
    ax.legend()
    return _save(fig, figures_dir / "02_validation" / "threshold_diagnostics_by_label.png")


def create_validation_plots(
    *,
    predictions_val: pd.DataFrame,
    thresholds: dict[str, Any],
    figures_dir: Path,
    labels: list[str] | None = None,
) -> list[Path]:
    target_labels = labels or TARGET_LABELS
    return [
        plot_validation_pr_curves(predictions_val, figures_dir, target_labels),
        plot_validation_roc_curves(predictions_val, figures_dir, target_labels),
        plot_threshold_sweep_by_label(predictions_val, figures_dir, target_labels),
        plot_selected_thresholds(thresholds, figures_dir, target_labels),
        plot_threshold_diagnostics_by_label(thresholds, figures_dir, target_labels),
    ]
