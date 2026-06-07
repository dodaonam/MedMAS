from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessPaths:
    root: Path
    data_dir: Path
    csv_path: Path
    image_dir: Path
    eda_artifacts_dir: Path
    eda_image_metrics_path: Path
    artifacts_dir: Path
    figures_dir: Path
    config_path: Path
    target_labels_path: Path
    manifest_all_path: Path
    manifest_filtered_path: Path
    split_manifest_path: Path
    split_audit_path: Path
    notebook_path: Path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (
            candidate / "data" / "chest_x-ray" / "sample_labels.csv"
        ).exists():
            return candidate
    raise FileNotFoundError("Could not locate project root with pyproject.toml and chest X-ray data")


def resolve_image_dir(data_dir: Path) -> Path:
    candidates = [
        data_dir / "sample" / "images",
        data_dir / "sample" / "sample" / "images",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.png")):
            return candidate
    raise FileNotFoundError(f"Could not locate PNG images under {data_dir}")


def resolve_paths(start: Path | None = None, artifacts_dir: Path | None = None) -> PreprocessPaths:
    root = find_project_root(start)
    data_dir = root / "data" / "chest_x-ray"
    artifacts = artifacts_dir or (root / "artifacts" / "preprocess")
    figures = artifacts / "figures"
    eda_artifacts = root / "artifacts" / "eda_image"
    return PreprocessPaths(
        root=root,
        data_dir=data_dir,
        csv_path=data_dir / "sample_labels.csv",
        image_dir=resolve_image_dir(data_dir),
        eda_artifacts_dir=eda_artifacts,
        eda_image_metrics_path=eda_artifacts / "image_metrics.csv",
        artifacts_dir=artifacts,
        figures_dir=figures,
        config_path=artifacts / "preprocess_config.json",
        target_labels_path=artifacts / "target_labels.json",
        manifest_all_path=artifacts / "manifest_all.csv",
        manifest_filtered_path=artifacts / "manifest_filtered.csv",
        split_manifest_path=artifacts / "split_manifest.csv",
        split_audit_path=artifacts / "split_audit.json",
        notebook_path=root / "notebook" / "preprocess_ready_for_training.ipynb",
    )


def ensure_workspace(paths: PreprocessPaths) -> None:
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.notebook_path.parent.mkdir(parents=True, exist_ok=True)
