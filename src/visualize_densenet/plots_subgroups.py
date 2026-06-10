from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_densenet.artifacts import DISEASE_LABELS, label_slug
from train_densenet.metrics import compute_subgroup_metrics_from_frame


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_subgroup_metric(
    predictions: pd.DataFrame,
    *,
    group_column: str,
    metric_name: str,
    output_path: Path,
    labels: list[str] | None = None,
    min_positives: int = 20,
) -> Path:
    target_labels = labels or DISEASE_LABELS
    subgroup_metrics = compute_subgroup_metrics_from_frame(
        predictions,
        group_column,
        target_labels,
        min_positives=min_positives,
    )
    groups = list(subgroup_metrics["groups"].keys())
    matrix = np.full((len(groups), len(target_labels)), np.nan)
    for group_idx, group in enumerate(groups):
        for label_idx, label in enumerate(target_labels):
            item = subgroup_metrics["groups"][group]["per_label"][label]
            if metric_name == "prevalence":
                matrix[group_idx, label_idx] = float(item["prevalence"])
            elif item.get("reported") and item.get(metric_name) is not None:
                matrix[group_idx, label_idx] = float(item[metric_name])
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(groups) * 0.7)))
    im = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(target_labels)), target_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(groups)), groups)
    title_metric = metric_name.replace("_", " ")
    ax.set_title(f"Test {title_metric} by {group_column}")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            text = "NA" if np.isnan(matrix[row, col]) else f"{matrix[row, col]:.2f}"
            ax.text(col, row, text, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, output_path)


def plot_subgroup_recall(
    predictions: pd.DataFrame,
    *,
    group_column: str,
    output_path: Path,
    labels: list[str] | None = None,
    min_positives: int = 20,
) -> Path:
    return plot_subgroup_metric(
        predictions,
        group_column=group_column,
        metric_name="recall",
        output_path=output_path,
        labels=labels,
        min_positives=min_positives,
    )


def out_of_scope_error_count_table(predictions: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    target_labels = labels or DISEASE_LABELS
    out_of_scope = predictions["has_out_of_scope_label"].astype(str).str.lower().isin({"1", "true", "yes"})
    rows: list[dict[str, int | str]] = []
    for label in target_labels:
        slug = label_slug(label)
        false_positive = (predictions[f"true_{slug}"] == 0) & (predictions[f"pred_{slug}"] == 1)
        false_negative = (predictions[f"true_{slug}"] == 1) & (predictions[f"pred_{slug}"] == 0)
        rows.append(
            {
                "label": label,
                "in_scope_fp": int(np.sum(false_positive & ~out_of_scope)),
                "in_scope_fn": int(np.sum(false_negative & ~out_of_scope)),
                "out_of_scope_fp": int(np.sum(false_positive & out_of_scope)),
                "out_of_scope_fn": int(np.sum(false_negative & out_of_scope)),
            }
        )
    return pd.DataFrame(rows)


def plot_out_of_scope_error_counts(
    predictions: pd.DataFrame,
    *,
    output_path: Path,
    labels: list[str] | None = None,
) -> Path:
    target_labels = labels or DISEASE_LABELS
    table = out_of_scope_error_count_table(predictions, target_labels)
    columns = ["in_scope_fp", "in_scope_fn", "out_of_scope_fp", "out_of_scope_fn"]
    matrix = table[columns].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(target_labels) * 0.7)))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(columns)), [column.replace("_", " ") for column in columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(target_labels)), target_labels)
    ax.set_title("Test false positives/negatives by out-of-scope co-label flag")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, output_path)


def create_subgroup_plots(
    *,
    predictions_test: pd.DataFrame,
    figures_dir: Path,
    labels: list[str] | None = None,
) -> list[Path]:
    target_labels = labels or DISEASE_LABELS
    paths: list[Path] = []
    if "View Position" in predictions_test.columns:
        paths.append(
            plot_subgroup_metric(
                predictions_test,
                group_column="View Position",
                metric_name="average_precision",
                output_path=figures_dir / "04_subgroups" / "test_average_precision_by_view_position.png",
                labels=target_labels,
            )
        )
        paths.append(
            plot_subgroup_metric(
                predictions_test,
                group_column="View Position",
                metric_name="auroc",
                output_path=figures_dir / "04_subgroups" / "test_auroc_by_view_position.png",
                labels=target_labels,
            )
        )
        paths.append(
            plot_subgroup_recall(
                predictions_test,
                group_column="View Position",
                output_path=figures_dir / "04_subgroups" / "test_metrics_by_view_position.png",
                labels=target_labels,
            )
        )
    if "Patient Gender" in predictions_test.columns:
        paths.append(
            plot_subgroup_metric(
                predictions_test,
                group_column="Patient Gender",
                metric_name="average_precision",
                output_path=figures_dir / "04_subgroups" / "test_average_precision_by_gender.png",
                labels=target_labels,
            )
        )
        paths.append(
            plot_subgroup_metric(
                predictions_test,
                group_column="Patient Gender",
                metric_name="auroc",
                output_path=figures_dir / "04_subgroups" / "test_auroc_by_gender.png",
                labels=target_labels,
            )
        )
        paths.append(
            plot_subgroup_recall(
                predictions_test,
                group_column="Patient Gender",
                output_path=figures_dir / "04_subgroups" / "test_metrics_by_gender.png",
                labels=target_labels,
            )
        )
    if "AgeBin" in predictions_test.columns:
        paths.append(
            plot_subgroup_metric(
                predictions_test,
                group_column="AgeBin",
                metric_name="prevalence",
                output_path=figures_dir / "04_subgroups" / "test_prevalence_by_age_bin.png",
                labels=target_labels,
            )
        )
        paths.append(
            plot_subgroup_recall(
                predictions_test,
                group_column="AgeBin",
                output_path=figures_dir / "04_subgroups" / "test_metrics_by_age_bin.png",
                labels=target_labels,
            )
        )
    if "has_out_of_scope_label" in predictions_test.columns:
        paths.append(
            plot_out_of_scope_error_counts(
                predictions_test,
                output_path=figures_dir / "04_subgroups" / "test_errors_by_out_of_scope_label.png",
                labels=target_labels,
            )
        )
    return paths
