from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .split import patient_overlap_matrix, target_counts_by_split
from .targets import TARGET_LABELS


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def value_distribution(df: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split, group in df.groupby("split", observed=False):
        counts = group[column].fillna("Missing").astype(str).value_counts().sort_index()
        result[str(split)] = {str(key): int(value) for key, value in counts.items()}
    return result


def create_split_audit(
    manifest_all: pd.DataFrame,
    manifest_filtered: pd.DataFrame,
    split_manifest: pd.DataFrame,
    integrity_summary: dict[str, int],
    selected_seed: int,
    target_labels: list[str] | None = None,
) -> dict[str, Any]:
    targets = target_labels or TARGET_LABELS
    split_order = ["train", "val", "test"]
    row_counts = split_manifest["split"].value_counts().reindex(split_order).fillna(0).astype(int)
    patient_counts = (
        split_manifest.groupby("split", observed=False)["Patient ID"].nunique().reindex(split_order).fillna(0).astype(int)
    )
    target_counts = target_counts_by_split(split_manifest, targets)
    target_prevalence = target_counts.div(row_counts.replace(0, np.nan), axis=0).fillna(0)
    overlap = patient_overlap_matrix(split_manifest)

    dropped = manifest_all.loc[~manifest_all["keep_for_mvp"].astype(bool)]
    retained_with_out_of_scope = manifest_filtered.loc[manifest_filtered["has_out_of_scope_label"].astype(bool)]

    audit = {
        "selected_seed": selected_seed,
        "integrity_summary": integrity_summary,
        "rows_per_split": row_counts.to_dict(),
        "patients_per_split": patient_counts.to_dict(),
        "patient_overlap_matrix": overlap.to_dict(),
        "positive_count_per_target_per_split": target_counts.to_dict(),
        "prevalence_per_target_per_split": target_prevalence.round(6).to_dict(),
        "dropped_row_summary": {
            "all_rows": int(len(manifest_all)),
            "retained_rows": int(len(manifest_filtered)),
            "dropped_out_of_scope_only_rows": int(len(dropped)),
        },
        "target_plus_out_of_scope_summary": {
            "retained_rows_with_target_plus_out_of_scope": int(len(retained_with_out_of_scope)),
        },
        "view_distribution_per_split": value_distribution(split_manifest, "View Position"),
        "gender_distribution_per_split": value_distribution(split_manifest, "Patient Gender"),
        "age_bin_distribution_per_split": value_distribution(split_manifest, "AgeBin"),
        "num_labels_distribution_per_split": value_distribution(split_manifest, "NumLabels"),
        "num_images_per_patient_distribution_per_split": value_distribution(split_manifest, "NumImagesForPatient"),
    }
    return _json_safe(audit)


def write_split_audit(path: Path, audit: dict[str, Any]) -> None:
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def _apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.dpi": 120,
            "savefig.dpi": 120,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_split_label_counts(split_manifest: pd.DataFrame, figures_dir: Path, target_labels: list[str] | None = None) -> Path:
    _apply_plot_style()
    targets = target_labels or TARGET_LABELS
    counts = target_counts_by_split(split_manifest, targets)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.T.plot(kind="bar", ax=ax, color=["#2563eb", "#f97316", "#16a34a"])
    ax.set_title("Target positive counts by split")
    ax.set_xlabel("Target")
    ax.set_ylabel("Positive rows")
    ax.tick_params(axis="x", rotation=35)
    path = figures_dir / "split_label_counts.png"
    _save(fig, path)
    return path


def plot_split_label_prevalence(
    split_manifest: pd.DataFrame, figures_dir: Path, target_labels: list[str] | None = None
) -> Path:
    _apply_plot_style()
    targets = target_labels or TARGET_LABELS
    counts = target_counts_by_split(split_manifest, targets)
    row_counts = split_manifest["split"].value_counts().reindex(["train", "val", "test"]).fillna(0)
    prevalence = counts.div(row_counts.replace(0, np.nan), axis=0).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    prevalence.T.plot(kind="bar", ax=ax, color=["#2563eb", "#f97316", "#16a34a"])
    ax.set_title("Target prevalence by split")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prevalence (%)")
    ax.tick_params(axis="x", rotation=35)
    path = figures_dir / "split_label_prevalence.png"
    _save(fig, path)
    return path


def plot_distribution_by_split(
    split_manifest: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    path: Path,
) -> Path:
    _apply_plot_style()
    table = (
        split_manifest.groupby(["split", column], observed=False)
        .size()
        .unstack("split")
        .reindex(columns=["train", "val", "test"])
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    table.plot(kind="bar", ax=ax, color=["#2563eb", "#f97316", "#16a34a"])
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path)
    return path


def create_all_plots(split_manifest: pd.DataFrame, figures_dir: Path) -> list[Path]:
    return [
        plot_split_label_counts(split_manifest, figures_dir),
        plot_split_label_prevalence(split_manifest, figures_dir),
        plot_distribution_by_split(
            split_manifest,
            "View Position",
            "AP/PA view distribution by split",
            "Rows",
            figures_dir / "split_view_distribution.png",
        ),
        plot_distribution_by_split(
            split_manifest,
            "Patient Gender",
            "Gender distribution by split",
            "Rows",
            figures_dir / "split_gender_distribution.png",
        ),
        plot_distribution_by_split(
            split_manifest,
            "AgeBin",
            "Age-bin distribution by split",
            "Rows",
            figures_dir / "split_age_distribution.png",
        ),
    ]
