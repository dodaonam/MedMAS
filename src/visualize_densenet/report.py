from __future__ import annotations

from pathlib import Path

from .data import load_manifest_for_visualization, load_visualization_artifacts
from .plots import create_all_plots


def generate_visualization_report(
    *,
    run_dir: Path,
    manifest_path: Path,
    target_labels_path: Path,
    root: Path,
) -> list[Path]:
    artifacts = load_visualization_artifacts(run_dir, target_labels_path)
    manifest = load_manifest_for_visualization(manifest_path, artifacts.target_labels)
    return create_all_plots(
        manifest=manifest,
        history=artifacts.history,
        predictions_test=artifacts.predictions_test,
        metrics_test=artifacts.metrics_test,
        figures_dir=artifacts.figures_dir,
        root=root,
        labels=artifacts.target_labels,
    )
