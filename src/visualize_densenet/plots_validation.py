from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_densenet.artifacts import DISEASE_LABELS, label_slug
from train_densenet.metrics import confusion_counts, metrics_from_counts, safe_auroc, safe_average_precision
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


def _selected_threshold_point(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    counts = confusion_counts(y_true, (y_prob >= threshold).astype(int))
    metrics = metrics_from_counts(counts)
    fpr = 0.0 if counts["fp"] + counts["tn"] == 0 else counts["fp"] / (counts["fp"] + counts["tn"])
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "fpr": fpr,
        "tpr": metrics["recall"],
    }


def plot_validation_pr_curves(
    predictions_val: pd.DataFrame,
    thresholds: dict[str, Any],
    figures_dir: Path,
    labels: list[str] | None = None,
) -> Path:
    target_labels = labels or DISEASE_LABELS
    selected = thresholds_by_label(thresholds, target_labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in target_labels:
        slug = label_slug(label)
        y_true = predictions_val[f"true_{slug}"].to_numpy(dtype=int)
        y_prob = predictions_val[f"prob_{slug}"].to_numpy(dtype=float)
        recall, precision = _pr_curve(
            y_true,
            y_prob,
        )
        ap = safe_average_precision(y_true, y_prob)["value"]
        ap_text = "NA" if ap is None else f"{ap:.3f}"
        label_text = f"{label} AP={ap_text}"
        ax.plot(recall, precision, label=label_text)
        point = _selected_threshold_point(y_true, y_prob, selected[label])
        ax.scatter(point["recall"], point["precision"], s=24)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Validation precision-recall curves")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir / "02_validation" / "validation_pr_curves.png")


def plot_validation_roc_curves(
    predictions_val: pd.DataFrame,
    thresholds: dict[str, Any],
    figures_dir: Path,
    labels: list[str] | None = None,
) -> Path:
    target_labels = labels or DISEASE_LABELS
    selected = thresholds_by_label(thresholds, target_labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in target_labels:
        slug = label_slug(label)
        y_true = predictions_val[f"true_{slug}"].to_numpy(dtype=int)
        y_prob = predictions_val[f"prob_{slug}"].to_numpy(dtype=float)
        fpr, tpr = _roc_curve(
            y_true,
            y_prob,
        )
        auroc = safe_auroc(y_true, y_prob)["value"]
        auroc_text = "NA" if auroc is None else f"{auroc:.3f}"
        label_text = f"{label} AUROC={auroc_text}"
        ax.plot(fpr, tpr, label=label_text)
        point = _selected_threshold_point(y_true, y_prob, selected[label])
        ax.scatter(point["fpr"], point["tpr"], s=24)
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Validation ROC curves")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir / "02_validation" / "validation_roc_curves.png")


def plot_threshold_sweep_by_label(
    predictions_val: pd.DataFrame,
    thresholds: dict[str, Any],
    figures_dir: Path,
    labels: list[str] | None = None,
) -> Path:
    target_labels = labels or DISEASE_LABELS
    selected = thresholds_by_label(thresholds, target_labels)
    per_label = thresholds.get("per_label", {})
    n_cols = 2
    n_rows = int(np.ceil(len(target_labels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, max(4, n_rows * 3.4)), sharex=True, sharey=True)
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")
    grid = threshold_grid()
    for ax, label in zip(axes_array, target_labels, strict=False):
        ax.axis("on")
        slug = label_slug(label)
        true = predictions_val[f"true_{slug}"].to_numpy(dtype=int)
        prob = predictions_val[f"prob_{slug}"].to_numpy(dtype=float)
        precision_values = []
        recall_values = []
        f1_values = []
        for threshold in grid:
            counts = confusion_counts(true, (prob >= threshold).astype(int))
            metrics = metrics_from_counts(counts)
            precision_values.append(metrics["precision"])
            recall_values.append(metrics["recall"])
            f1_values.append(metrics["f1"])
        ax.plot(grid, precision_values, label="precision")
        ax.plot(grid, recall_values, label="recall")
        ax.plot(grid, f1_values, label="F1")
        ax.axvline(selected[label], color="0.2", linestyle="--", linewidth=1, label="selected")
        title = label
        if per_label.get(label, {}).get("threshold_unstable"):
            title = f"{label} (unstable)"
        ax.set_title(title)
        ax.set_xlabel("threshold")
        ax.set_ylabel("metric")
        ax.legend(fontsize=7)
    fig.suptitle("Validation threshold sweep by disease label")
    return _save(fig, figures_dir / "02_validation" / "threshold_sweep_by_label.png")


def plot_selected_thresholds(thresholds: dict[str, Any], figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or DISEASE_LABELS
    selected = thresholds_by_label(thresholds, target_labels)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(target_labels, [selected[label] for label in target_labels])
    ax.axhline(0.5, color="0.35", linestyle="--", linewidth=1, label="0.5 reference")
    ax.set_ylim(0, 1)
    ax.set_ylabel("threshold")
    ax.set_title("Validation-selected thresholds")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    return _save(fig, figures_dir / "02_validation" / "selected_thresholds.png")


def plot_threshold_diagnostics_by_label(thresholds: dict[str, Any], figures_dir: Path, labels: list[str] | None = None) -> Path:
    target_labels = labels or DISEASE_LABELS
    per_label = thresholds.get("per_label", {})
    columns = ["label", "val +", "threshold", "precision", "recall", "F1", "TP", "FP", "FN", "flag"]
    rows = []
    for label in target_labels:
        item = per_label.get(label, {})
        rows.append(
            [
                label,
                str(item.get("positive_count", "")),
                "" if item.get("threshold") is None else f"{float(item['threshold']):.2f}",
                "" if item.get("precision") is None else f"{float(item['precision']):.2f}",
                "" if item.get("recall") is None else f"{float(item['recall']):.2f}",
                "" if item.get("f1") is None else f"{float(item['f1']):.2f}",
                str(item.get("tp", "")),
                str(item.get("fp", "")),
                str(item.get("fn", "")),
                "unstable" if item.get("threshold_unstable") else "",
            ]
        )
    fig, ax = plt.subplots(figsize=(11, 0.7 * len(rows) + 1.6))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    ax.set_title("Threshold diagnostics at selected thresholds", pad=14)
    return _save(fig, figures_dir / "02_validation" / "threshold_diagnostics_by_label.png")


def create_validation_plots(
    *,
    predictions_val: pd.DataFrame,
    thresholds: dict[str, Any],
    figures_dir: Path,
    labels: list[str] | None = None,
) -> list[Path]:
    target_labels = labels or DISEASE_LABELS
    return [
        plot_validation_pr_curves(predictions_val, thresholds, figures_dir, target_labels),
        plot_validation_roc_curves(predictions_val, thresholds, figures_dir, target_labels),
        plot_threshold_sweep_by_label(predictions_val, thresholds, figures_dir, target_labels),
        plot_selected_thresholds(thresholds, figures_dir, target_labels),
        plot_threshold_diagnostics_by_label(thresholds, figures_dir, target_labels),
    ]
