from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from train_densenet import TARGET_LABELS, artifact_paths, load_manifest, load_target_labels, resolve_run_dir


@dataclass
class VisualizationArtifacts:
    run_dir: Path
    figures_dir: Path
    config: dict[str, Any]
    history: pd.DataFrame
    predictions_val: pd.DataFrame
    predictions_test: pd.DataFrame
    metrics_val: dict[str, Any]
    metrics_test: dict[str, Any]
    target_labels: list[str]


def load_visualization_artifacts(run_dir: Path, target_labels_path: Path | None = None) -> VisualizationArtifacts:
    resolved_run_dir = resolve_run_dir(run_dir)
    paths = artifact_paths(resolved_run_dir)
    config = _load_json(paths.config_path)
    labels = list(config.get("target_labels") or TARGET_LABELS)
    if target_labels_path is not None:
        expected = load_target_labels(target_labels_path)
        if labels != expected:
            raise ValueError(f"Run labels {labels!r} do not match {target_labels_path}: {expected!r}")

    history = pd.read_csv(paths.history_path)
    predictions_val = pd.read_csv(paths.predictions_val_path)
    predictions_test = pd.read_csv(paths.predictions_test_path)
    metrics_val = _load_json(paths.metrics_val_path)
    metrics_test = _load_json(paths.metrics_test_path)
    _validate_run_id(config, history, predictions_val, predictions_test, metrics_val, metrics_test)
    return VisualizationArtifacts(
        run_dir=resolved_run_dir,
        figures_dir=paths.figures_dir,
        config=config,
        history=history,
        predictions_val=predictions_val,
        predictions_test=predictions_test,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        target_labels=labels,
    )


def load_manifest_for_visualization(manifest_path: Path, labels: list[str]) -> pd.DataFrame:
    return load_manifest(manifest_path, labels)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_run_id(
    config: dict[str, Any],
    history: pd.DataFrame,
    predictions_val: pd.DataFrame,
    predictions_test: pd.DataFrame,
    metrics_val: dict[str, Any],
    metrics_test: dict[str, Any],
) -> None:
    expected = str(config.get("run_id", ""))
    if not expected:
        raise ValueError("config.json is missing run_id")
    checks = {
        "training_history.csv": _frame_run_id(history),
        "predictions_val.csv": _frame_run_id(predictions_val),
        "predictions_test.csv": _frame_run_id(predictions_test),
        "metrics_val.json": metrics_val.get("run_id"),
        "metrics_test.json": metrics_test.get("run_id"),
    }
    mismatched = {name: value for name, value in checks.items() if value is not None and str(value) != expected}
    if mismatched:
        raise ValueError(f"Run ID mismatch. Expected {expected!r}, got {mismatched}")


def _frame_run_id(frame: pd.DataFrame) -> str | None:
    if "run_id" not in frame.columns or frame.empty:
        return None
    values = frame["run_id"].dropna().astype(str).unique().tolist()
    if len(values) > 1:
        raise ValueError(f"Multiple run_id values found: {values}")
    return values[0] if values else None
