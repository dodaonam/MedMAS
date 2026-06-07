from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .image_quality import load_gray


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.dpi": 120,
            "savefig.dpi": 120,
        }
    )


def plot_label_frequency(label_summary: pd.DataFrame):
    plot_df = label_summary.reset_index().rename(columns={"index": "label"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(plot_df["label"], plot_df["count"], color="#2563eb")
    axes[0].set_title("Label frequency")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=90)
    axes[1].bar(plot_df["label"], plot_df["count"], color="#1e40af")
    axes[1].set_yscale("log")
    axes[1].set_title("Label frequency on log scale")
    axes[1].set_ylabel("Count (log)")
    axes[1].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig, axes


def plot_horizontal_risk_ratio(df: pd.DataFrame, label_col: str = "label", value_col: str = "log2_rr", title: str = ""):
    plot_df = df.sort_values(value_col)
    colors = ["#dc2626" if value > 0 else "#2563eb" for value in plot_df[value_col]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df[label_col], plot_df[value_col], color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("log2 risk ratio")
    fig.tight_layout()
    return fig, ax


def plot_heatmap(matrix: pd.DataFrame, title: str, cmap: str = "Blues", figsize: tuple[int, int] = (10, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix.values, cmap=cmap)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=90)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_image_grid(
    rows: pd.DataFrame,
    image_dir: Path,
    title_cols: list[str],
    n_cols: int = 3,
    figsize_per_image: float = 3.6,
    suptitle: str | None = None,
):
    n_images = len(rows)
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_image * n_cols, figsize_per_image * n_rows))
    axes = np.array(axes, dtype=object).reshape(n_rows, n_cols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
        arr = load_gray(image_dir / row["Image Index"])
        ax.imshow(arr, cmap="gray")
        title_parts = [str(row[col]) for col in title_cols if col in row]
        ax.set_title("\n".join(title_parts), fontsize=8)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, y=1.01)
    fig.tight_layout()
    return fig, axes

