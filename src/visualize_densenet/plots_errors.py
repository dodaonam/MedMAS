from __future__ import annotations

from pathlib import Path

import pandas as pd

from train_densenet.artifacts import TARGET_LABELS, label_slug
from .plots_data import plot_image_grid_from_paths


def false_positive_rows(predictions: pd.DataFrame, label: str) -> pd.DataFrame:
    slug = label_slug(label)
    return predictions.loc[(predictions[f"true_{slug}"] == 0) & (predictions[f"pred_{slug}"] == 1)].sort_values(
        f"prob_{slug}",
        ascending=False,
    )


def false_negative_rows(predictions: pd.DataFrame, label: str) -> pd.DataFrame:
    slug = label_slug(label)
    return predictions.loc[(predictions[f"true_{slug}"] == 1) & (predictions[f"pred_{slug}"] == 0)].sort_values(
        f"prob_{slug}",
        ascending=True,
    )


def create_error_grids(
    *,
    predictions_test: pd.DataFrame,
    figures_dir: Path,
    root: Path,
    labels: list[str] | None = None,
    max_images_per_label: int = 12,
) -> list[Path]:
    target_labels = labels or TARGET_LABELS
    paths: list[Path] = []
    for label in target_labels:
        slug = label_slug(label)
        fp = false_positive_rows(predictions_test, label)
        fn = false_negative_rows(predictions_test, label)
        paths.append(
            plot_image_grid_from_paths(
                fp,
                root=root,
                output_path=figures_dir / "05_errors" / f"false_positives_{slug}.png",
                max_images=max_images_per_label,
            )
        )
        paths.append(
            plot_image_grid_from_paths(
                fn,
                root=root,
                output_path=figures_dir / "05_errors" / f"false_negatives_{slug}.png",
                max_images=max_images_per_label,
            )
        )
    return paths
