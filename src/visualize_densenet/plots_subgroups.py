from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_densenet.artifacts import TARGET_LABELS
from train_densenet.metrics import compute_subgroup_metrics_from_frame


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_subgroup_recall(
    predictions: pd.DataFrame,
    *,
    group_column: str,
    output_path: Path,
    labels: list[str] | None = None,
    min_positives: int = 20,
) -> Path:
    target_labels = labels or TARGET_LABELS
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
            if item.get("reported"):
                matrix[group_idx, label_idx] = float(item["recall"])
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(groups) * 0.7)))
    im = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(target_labels)), target_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(groups)), groups)
    ax.set_title(f"Test recall by {group_column}")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            text = "NA" if np.isnan(matrix[row, col]) else f"{matrix[row, col]:.2f}"
            ax.text(col, row, text, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, output_path)


def create_subgroup_plots(
    *,
    predictions_test: pd.DataFrame,
    figures_dir: Path,
    labels: list[str] | None = None,
) -> list[Path]:
    target_labels = labels or TARGET_LABELS
    paths: list[Path] = []
    if "View Position" in predictions_test.columns:
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
            plot_subgroup_recall(
                predictions_test,
                group_column="Patient Gender",
                output_path=figures_dir / "04_subgroups" / "test_metrics_by_gender.png",
                labels=target_labels,
            )
        )
    if "AgeBin" in predictions_test.columns:
        paths.append(
            plot_subgroup_recall(
                predictions_test,
                group_column="AgeBin",
                output_path=figures_dir / "04_subgroups" / "test_metrics_by_age_bin.png",
                labels=target_labels,
            )
        )
    return paths
