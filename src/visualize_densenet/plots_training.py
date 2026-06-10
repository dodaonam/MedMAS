from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _stage_transition_epoch(history: pd.DataFrame) -> int | None:
    if "stage" not in history.columns:
        return None
    stage2 = history.loc[history["stage"].astype(str) == "denseblock4_norm5_finetune", "epoch"]
    if stage2.empty:
        return None
    return int(stage2.min())


def _draw_stage_line(ax: Any, history: pd.DataFrame) -> None:
    transition = _stage_transition_epoch(history)
    if transition is not None:
        ax.axvline(transition, color="0.3", linestyle="--", linewidth=1, label="stage 2 start")


def plot_loss_curve(history: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="train loss")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="val loss")
    _draw_stage_line(ax, history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Train vs validation loss")
    ax.legend()
    return _save(fig, figures_dir / "01_training" / "loss_curve.png")


def plot_macro_metric_curves(history: pd.DataFrame, figures_dir: Path) -> Path:
    metric_columns = [
        "val_disease_macro_auroc",
        "val_disease_macro_average_precision",
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for column in metric_columns:
        if column in history.columns:
            ax.plot(history["epoch"], history[column], marker="o", label=column)
    _draw_stage_line(ax, history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.set_title("Validation macro metric curves")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir / "01_training" / "macro_metric_curves.png")


def plot_per_label_metric_curves(history: pd.DataFrame, figures_dir: Path) -> Path:
    columns = [column for column in history.columns if column.startswith("val_") and column.endswith("_average_precision")]
    per_label_columns = [
        column for column in columns if column != "val_disease_macro_average_precision"
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if per_label_columns:
        for column in per_label_columns:
            ax.plot(history["epoch"], history[column], marker="o", label=column.removeprefix("val_"))
    else:
        ax.text(0.5, 0.5, "per-disease-label metric columns not saved", ha="center", va="center", transform=ax.transAxes)
    _draw_stage_line(ax, history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("average precision")
    ax.set_title("Per-disease-label validation metric curves")
    if per_label_columns:
        ax.legend(fontsize=8)
    return _save(fig, figures_dir / "01_training" / "per_label_metric_curves.png")


def plot_learning_rate_curve(history: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "learning_rate_cnn_head" in history.columns:
        ax.plot(history["epoch"], history["learning_rate_cnn_head"], marker="o", label="cnn_head")
    if "learning_rate_denseblock4_norm5" in history.columns:
        valid = history["learning_rate_denseblock4_norm5"].notna()
        ax.plot(history.loc[valid, "epoch"], history.loc[valid, "learning_rate_denseblock4_norm5"], marker="o", label="denseblock4_norm5")
    _draw_stage_line(ax, history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning-rate schedule")
    ax.legend()
    return _save(fig, figures_dir / "01_training" / "learning_rate_curve.png")


def plot_epoch_duration(history: pd.DataFrame, figures_dir: Path) -> Path | None:
    if "epoch_duration" not in history.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["epoch_duration"], marker="o")
    _draw_stage_line(ax, history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds")
    ax.set_title("Epoch duration")
    return _save(fig, figures_dir / "01_training" / "epoch_duration.png")


def create_training_plots(history: pd.DataFrame, figures_dir: Path) -> list[Path]:
    paths = [
        plot_loss_curve(history, figures_dir),
        plot_macro_metric_curves(history, figures_dir),
        plot_per_label_metric_curves(history, figures_dir),
        plot_learning_rate_curve(history, figures_dir),
    ]
    optional = plot_epoch_duration(history, figures_dir)
    if optional is not None:
        paths.append(optional)
    return paths
