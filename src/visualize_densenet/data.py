from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from train_densenet.artifacts import TARGET_LABELS, load_target_labels, resolve_artifact_paths
from train_densenet.dataset import assert_patient_disjoint, load_split_manifest


@dataclass
class VisualizationArtifacts:
    run_dir: Path
    figures_dir: Path
    config: dict[str, Any]
    history: pd.DataFrame
    class_weights: dict[str, Any]
    thresholds: dict[str, Any]
    predictions_val: pd.DataFrame
    predictions_test: pd.DataFrame
    metrics_val: dict[str, Any]
    metrics_test: dict[str, Any]
    target_labels: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_id_from_frame(frame: pd.DataFrame, path: Path) -> str | None:
    if "run_id" not in frame.columns or frame.empty:
        return None
    values = frame["run_id"].dropna().astype(str).unique().tolist()
    if len(values) > 1:
        raise ValueError(f"Multiple run_id values in {path}: {values}")
    return values[0] if values else None


def _validate_run_ids(
    *,
    expected: str,
    history: pd.DataFrame,
    predictions_val: pd.DataFrame,
    predictions_test: pd.DataFrame,
    thresholds: dict[str, Any],
    metrics_val: dict[str, Any],
    metrics_test: dict[str, Any],
) -> None:
    checks = {
        "training_history.csv": _run_id_from_frame(history, Path("training_history.csv")),
        "predictions_val.csv": _run_id_from_frame(predictions_val, Path("predictions_val.csv")),
        "predictions_test.csv": _run_id_from_frame(predictions_test, Path("predictions_test.csv")),
        "thresholds.json": thresholds.get("run_id"),
        "metrics_val.json": metrics_val.get("run_id"),
        "metrics_test.json": metrics_test.get("run_id"),
    }
    mismatched = {name: value for name, value in checks.items() if value is not None and str(value) != expected}
    if mismatched:
        raise ValueError(f"Run ID mismatch. Expected {expected!r}, got {mismatched}")


def load_visualization_artifacts(run_dir: Path, target_labels_path: Path | None = None) -> VisualizationArtifacts:
    paths = resolve_artifact_paths(run_dir)
    config = _load_json(paths.config_path)
    run_id = str(config["run_id"])
    history = pd.read_csv(paths.training_history_path)
    class_weights = _load_json(paths.class_weights_path)
    thresholds = _load_json(paths.thresholds_path)
    predictions_val = pd.read_csv(paths.predictions_val_path)
    predictions_test = pd.read_csv(paths.predictions_test_path)
    metrics_val = _load_json(paths.metrics_val_path)
    metrics_test = _load_json(paths.metrics_test_path)
    labels = config.get("target_label_order") or TARGET_LABELS
    if target_labels_path is not None:
        labels = load_target_labels(target_labels_path)
    if labels != TARGET_LABELS:
        raise ValueError(f"Unexpected target label order: {labels!r}")
    _validate_run_ids(
        expected=run_id,
        history=history,
        predictions_val=predictions_val,
        predictions_test=predictions_test,
        thresholds=thresholds,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
    )
    return VisualizationArtifacts(
        run_dir=run_dir,
        figures_dir=paths.figures_dir,
        config=config,
        history=history,
        class_weights=class_weights,
        thresholds=thresholds,
        predictions_val=predictions_val,
        predictions_test=predictions_test,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        target_labels=list(labels),
    )


def load_manifest_for_visualization(manifest_path: Path, target_labels: list[str] | None = None) -> pd.DataFrame:
    manifest = load_split_manifest(manifest_path, target_labels or TARGET_LABELS)
    assert_patient_disjoint(manifest)
    return manifest
