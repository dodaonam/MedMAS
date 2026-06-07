from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .artifacts import DISEASE_LABELS, TARGET_LABELS
from .metrics import confusion_counts, metrics_from_counts


def threshold_grid() -> np.ndarray:
    return np.arange(0.05, 0.951, 0.01)


def select_threshold_for_label(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    label: str,
    grid: Sequence[float] | np.ndarray | None = None,
    unstable_positive_min: int = 50,
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(y_prob, dtype=float).reshape(-1)
    thresholds = np.asarray(list(grid) if grid is not None else threshold_grid(), dtype=float)
    positive_count = int(np.sum(true == 1))

    best: dict[str, Any] | None = None
    for value in thresholds:
        pred = (prob >= value).astype(int)
        counts = confusion_counts(true, pred)
        thresholded = metrics_from_counts(counts)
        candidate = {
            "threshold": float(round(float(value), 4)),
            "precision": thresholded["precision"],
            "recall": thresholded["recall"],
            "f1": thresholded["f1"],
            **counts,
        }
        if best is None:
            best = candidate
            continue
        if candidate["f1"] > best["f1"]:
            best = candidate
            continue
        if label in DISEASE_LABELS and candidate["f1"] == best["f1"] and candidate["recall"] > best["recall"]:
            best = candidate

    if best is None:
        raise ValueError("Threshold grid is empty")
    best["positive_count"] = positive_count
    best["threshold_unstable"] = positive_count < unstable_positive_min
    best["selection_metric"] = "max_validation_f1"
    return best


def select_validation_thresholds(
    y_true: Sequence[Sequence[int]] | np.ndarray,
    y_prob: Sequence[Sequence[float]] | np.ndarray,
    labels: list[str] | None = None,
    run_id: str | None = None,
    grid: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    target_labels = labels or TARGET_LABELS
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    if true.shape != prob.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape}, y_prob {prob.shape}")
    if true.ndim != 2 or true.shape[1] != len(target_labels):
        raise ValueError(f"Expected shape [n, {len(target_labels)}], got {true.shape}")

    grid_values = np.asarray(list(grid) if grid is not None else threshold_grid(), dtype=float)
    per_label = {
        label: select_threshold_for_label(true[:, idx], prob[:, idx], label=label, grid=grid_values)
        for idx, label in enumerate(target_labels)
    }
    return {
        "run_id": run_id,
        "target_label_order": target_labels,
        "threshold_selection_metric": "max_validation_f1",
        "threshold_grid": "np.arange(0.05, 0.951, 0.01)",
        "threshold_grid_values": [float(round(value, 4)) for value in grid_values],
        "thresholds": {label: per_label[label]["threshold"] for label in target_labels},
        "per_label": per_label,
    }


def thresholds_by_label(threshold_payload: Mapping[str, Any], labels: list[str] | None = None) -> dict[str, float]:
    target_labels = labels or TARGET_LABELS
    if "thresholds" in threshold_payload:
        source = threshold_payload["thresholds"]
        return {label: float(source[label]) for label in target_labels}
    if "per_label" in threshold_payload:
        source = threshold_payload["per_label"]
        return {label: float(source[label]["threshold"]) for label in target_labels}
    raise ValueError("Threshold payload must contain `thresholds` or `per_label`")
