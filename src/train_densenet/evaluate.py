from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import DISEASE_LABELS, label_slug
from .metrics import compute_metrics_from_prediction_frame
from .progress import ProgressBar

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only on machines without torch
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for model inference.")


def _metadata_batch_to_rows(metadata: Any, batch_size: int) -> list[dict[str, Any]]:
    if isinstance(metadata, Mapping):
        rows: list[dict[str, Any]] = []
        for idx in range(batch_size):
            row = {}
            for key, values in metadata.items():
                if isinstance(values, Sequence) and not isinstance(values, str):
                    row[key] = values[idx]
                else:
                    try:
                        row[key] = values[idx].item()
                    except Exception:
                        row[key] = values
            rows.append(row)
        return rows
    return list(metadata)


def build_prediction_frame(
    *,
    metadata_rows: list[dict[str, Any]],
    targets: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Mapping[str, float],
    labels: list[str] | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    target_labels = labels or DISEASE_LABELS
    if "No Finding" in target_labels:
        raise ValueError("Recipe v2 prediction frames use disease labels only; `No Finding` is derived.")
    if targets.shape != logits.shape or targets.shape != probabilities.shape:
        raise ValueError("targets, logits, and probabilities must have identical shapes")
    rows: list[dict[str, Any]] = []
    for idx, metadata in enumerate(metadata_rows):
        row = dict(metadata)
        row["run_id"] = run_id
        if "No Finding" in metadata:
            row["true_no_finding_source"] = int(metadata["No Finding"])
        disease_predictions: list[int] = []
        for label_idx, label in enumerate(target_labels):
            slug = label_slug(label)
            threshold = float(thresholds[label])
            probability = float(probabilities[idx, label_idx])
            prediction = int(probability >= threshold)
            row[label] = int(targets[idx, label_idx])
            row[f"true_{slug}"] = int(targets[idx, label_idx])
            row[f"logit_{slug}"] = float(logits[idx, label_idx])
            row[f"prob_{slug}"] = probability
            row[f"threshold_{slug}"] = threshold
            row[f"pred_{slug}"] = prediction
            disease_predictions.append(prediction)
        row["true_no_finding_derived"] = int(np.sum(targets[idx, :]) == 0)
        row["pred_no_finding_derived"] = int(sum(disease_predictions) == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def run_inference(
    model: Any,
    dataloader: Any,
    *,
    device: Any,
    thresholds: Mapping[str, float],
    labels: list[str] | None = None,
    run_id: str | None = None,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    _require_torch()
    target_labels = labels or DISEASE_LABELS
    model.eval()
    metadata_rows: list[dict[str, Any]] = []
    targets_list: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    probabilities_list: list[np.ndarray] = []
    with torch.no_grad():
        with ProgressBar(total=len(dataloader), desc=progress_desc or "infer", enabled=progress_desc is not None) as progress:
            for images, targets, metadata in dataloader:
                images = images.to(device)
                logits = model(images).detach().cpu()
                probabilities = torch.sigmoid(logits)
                batch_size = int(logits.shape[0])
                metadata_rows.extend(_metadata_batch_to_rows(metadata, batch_size))
                targets_list.append(targets.detach().cpu().numpy())
                logits_list.append(logits.numpy())
                probabilities_list.append(probabilities.numpy())
                progress.update()
    return build_prediction_frame(
        metadata_rows=metadata_rows,
        targets=np.concatenate(targets_list, axis=0),
        logits=np.concatenate(logits_list, axis=0),
        probabilities=np.concatenate(probabilities_list, axis=0),
        thresholds=thresholds,
        labels=target_labels,
        run_id=run_id,
    )


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    *,
    labels: list[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return compute_metrics_from_prediction_frame(frame, labels=labels or DISEASE_LABELS, run_id=run_id)
