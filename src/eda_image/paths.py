from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EdaPaths:
    root: Path
    data_dir: Path
    csv_path: Path
    image_dir: Path
    artifacts_dir: Path
    baseline_path: Path
    image_metrics_path: Path
    image_metrics_meta_path: Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the repo root from a notebook or shell working directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (
            candidate / "data" / "chest_x-ray" / "sample_labels.csv"
        ).exists():
            return candidate
    raise FileNotFoundError("Could not locate project root with pyproject.toml and chest X-ray data")


def resolve_image_dir(data_dir: Path) -> Path:
    """Pick the first local image directory that contains PNG files."""
    candidates = [
        data_dir / "sample" / "images",
        data_dir / "sample" / "sample" / "images",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.png")):
            return candidate
    raise FileNotFoundError(f"Could not locate PNG images under {data_dir}")


def resolve_paths(start: Path | None = None, artifacts_dir: Path | None = None) -> EdaPaths:
    root = find_project_root(start)
    data_dir = root / "data" / "chest_x-ray"
    csv_path = data_dir / "sample_labels.csv"
    image_dir = resolve_image_dir(data_dir)
    artifacts = artifacts_dir or (root / "artifacts" / "eda_image")
    return EdaPaths(
        root=root,
        data_dir=data_dir,
        csv_path=csv_path,
        image_dir=image_dir,
        artifacts_dir=artifacts,
        baseline_path=artifacts / "baseline_summary.json",
        image_metrics_path=artifacts / "image_metrics.csv",
        image_metrics_meta_path=artifacts / "image_metrics.meta.json",
    )

