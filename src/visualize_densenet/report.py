from __future__ import annotations

from pathlib import Path

from .data import load_manifest_for_visualization, load_visualization_artifacts
from .plots_data import create_data_plots
from .plots_errors import create_error_grids
from .plots_gradcam import create_gradcam_report
from .plots_subgroups import create_subgroup_plots
from .plots_test import create_test_plots
from .plots_training import create_training_plots
from .plots_validation import create_validation_plots


def generate_visualization_report(
    *,
    run_dir: Path,
    manifest_path: Path,
    target_labels_path: Path,
    root: Path,
    include_gradcam: bool = False,
    device: str | None = None,
) -> list[Path]:
    artifacts = load_visualization_artifacts(run_dir, target_labels_path)
    manifest = load_manifest_for_visualization(manifest_path, artifacts.target_labels)
    paths: list[Path] = []
    paths.extend(
        create_data_plots(
            manifest=manifest,
            class_weights=artifacts.class_weights,
            figures_dir=artifacts.figures_dir,
            labels=artifacts.target_labels,
        )
    )
    paths.extend(create_training_plots(artifacts.history, artifacts.figures_dir))
    paths.extend(
        create_validation_plots(
            predictions_val=artifacts.predictions_val,
            thresholds=artifacts.thresholds,
            figures_dir=artifacts.figures_dir,
            labels=artifacts.target_labels,
        )
    )
    paths.extend(
        create_test_plots(
            predictions_test=artifacts.predictions_test,
            metrics_test=artifacts.metrics_test,
            figures_dir=artifacts.figures_dir,
            labels=artifacts.target_labels,
        )
    )
    paths.extend(
        create_subgroup_plots(
            predictions_test=artifacts.predictions_test,
            figures_dir=artifacts.figures_dir,
            labels=artifacts.target_labels,
        )
    )
    paths.extend(
        create_error_grids(
            predictions_test=artifacts.predictions_test,
            figures_dir=artifacts.figures_dir,
            root=root,
            labels=artifacts.target_labels,
        )
    )
    if include_gradcam:
        paths.extend(
            create_gradcam_report(
                checkpoint_path=run_dir / "checkpoint_best.pt",
                predictions=artifacts.predictions_test,
                figures_dir=artifacts.figures_dir,
                root=root,
                labels=artifacts.target_labels,
                device_name=device,
            )
        )
    return paths
