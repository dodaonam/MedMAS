from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ._backend import require_torch, torch
from .config import label_slug
from .constants import (
    DYNAMIC_THRESHOLD_PRIOR_LABELS,
    LABEL_THRESHOLD_PRIORS,
    METADATA_COLUMNS,
    MIN_TUNED_THRESHOLD_POSITIVES,
    THRESHOLD_PRIOR_SEARCH_RADIUS,
)


def predict(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    labels: list[str],
    *,
    threshold: float,
    run_id: str,
    desc: str | None = None,
) -> tuple[float, pd.DataFrame, np.ndarray, np.ndarray]:
    require_torch()
    model.eval()
    total_loss = 0.0
    total_rows = 0
    metadata_rows: list[dict[str, Any]] = []
    targets_list: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    with torch.no_grad():
        progress = tqdm(loader, desc=desc, total=len(loader), dynamic_ncols=True, leave=False) if desc else loader
        for images, targets, metadata in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            batch_size = int(images.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_rows += batch_size
            metadata_rows.extend(metadata_to_rows(metadata, batch_size))
            targets_list.append(targets.detach().cpu().numpy())
            logits_list.append(logits.detach().cpu().numpy())
            if desc:
                progress.set_postfix(loss=f"{total_loss / max(total_rows, 1):.4f}")
    y_true = np.concatenate(targets_list, axis=0)
    logits_np = np.concatenate(logits_list, axis=0)
    y_prob = 1.0 / (1.0 + np.exp(-logits_np))
    frame = prediction_frame(metadata_rows, y_true, y_prob, labels, threshold=threshold, run_id=run_id)
    return total_loss / max(total_rows, 1), frame, y_true, y_prob


def prediction_frame(
    metadata_rows: list[dict[str, Any]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    *,
    threshold: float | Mapping[str, float],
    run_id: str,
) -> pd.DataFrame:
    thresholds = resolve_thresholds(labels, threshold)
    rows: list[dict[str, Any]] = []
    for idx, metadata in enumerate(metadata_rows):
        row = dict(metadata)
        row["run_id"] = run_id
        for label_idx, label in enumerate(labels):
            slug = label_slug(label)
            probability = float(y_prob[idx, label_idx])
            label_threshold = thresholds[label]
            row[label] = int(y_true[idx, label_idx])
            row[f"true_{slug}"] = int(y_true[idx, label_idx])
            row[f"prob_{slug}"] = probability
            row[f"threshold_{slug}"] = label_threshold
            row[f"pred_{slug}"] = int(probability >= label_threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def metadata_to_rows(metadata: Any, batch_size: int) -> list[dict[str, Any]]:
    if not isinstance(metadata, dict):
        return list(metadata)
    rows: list[dict[str, Any]] = []
    for index in range(batch_size):
        row: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                row[key] = value[index]
            elif torch is not None and torch.is_tensor(value):
                item = value[index]
                row[key] = item.item() if item.ndim == 0 else item.detach().cpu().tolist()
            else:
                row[key] = value
        rows.append(row)
    return rows


def metadata_rows_from_prediction_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [column for column in METADATA_COLUMNS if column in frame.columns]
    return frame.loc[:, columns].to_dict(orient="records")


def compute_metrics(
    y_true: Any,
    y_prob: Any,
    labels: list[str],
    *,
    threshold: float | Mapping[str, float] = 0.5,
    run_id: str | None = None,
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    if true.shape != prob.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape}, y_prob {prob.shape}")
    if true.ndim != 2 or true.shape[1] != len(labels):
        raise ValueError(f"Expected shape [n, {len(labels)}], got {true.shape}")

    thresholds = resolve_thresholds(labels, threshold)
    per_label: dict[str, Any] = {}
    micro_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for index, label in enumerate(labels):
        label_threshold = thresholds[label]
        pred = (prob[:, index] >= label_threshold).astype(int)
        counts = confusion_counts(true[:, index], pred)
        for key in micro_counts:
            micro_counts[key] += counts[key]
        basic = metrics_from_counts(counts)
        per_label[label] = {
            **counts,
            **basic,
            "positive_count": int(np.sum(true[:, index] == 1)),
            "threshold": label_threshold,
            "average_precision": average_precision(true[:, index], prob[:, index]),
            "auroc": auroc(true[:, index], prob[:, index]),
        }

    common_threshold = shared_threshold(thresholds)
    return {
        "run_id": run_id,
        "labels": labels,
        "threshold": common_threshold,
        "threshold_mode": "fixed" if common_threshold is not None else "per_label",
        "thresholds": thresholds,
        "per_label": per_label,
        "macro_precision": mean_defined(item["precision"] for item in per_label.values()),
        "macro_recall": mean_defined(item["recall"] for item in per_label.values()),
        "macro_f1": mean_defined(item["f1"] for item in per_label.values()),
        "macro_average_precision": mean_defined(item["average_precision"] for item in per_label.values()),
        "macro_auroc": mean_defined(item["auroc"] for item in per_label.values()),
        "micro": {**micro_counts, **metrics_from_counts(micro_counts)},
    }


def attach_slice_metrics(
    metrics: dict[str, Any],
    frame: pd.DataFrame,
    labels: list[str],
    *,
    threshold: float | Mapping[str, float],
) -> dict[str, Any]:
    payload = dict(metrics)
    thresholds = resolve_thresholds(labels, threshold)
    disease_labels = [label for label in labels if label != "No Finding"]
    if disease_labels:
        disease_thresholds = {label: thresholds[label] for label in disease_labels}
        disease_true, disease_prob = frame_targets_and_probabilities(frame, disease_labels)
        disease_metrics = compute_metrics(
            disease_true,
            disease_prob,
            disease_labels,
            threshold=disease_thresholds,
            run_id=metrics.get("run_id"),
        )
        payload["disease_only"] = {
            "labels": disease_labels,
            "macro_precision": disease_metrics["macro_precision"],
            "macro_recall": disease_metrics["macro_recall"],
            "macro_f1": disease_metrics["macro_f1"],
            "macro_average_precision": disease_metrics["macro_average_precision"],
            "macro_auroc": disease_metrics["macro_auroc"],
            "micro": disease_metrics["micro"],
        }
    subsets: dict[str, Any] = {}
    if "has_out_of_scope_label" in frame.columns:
        for subset_name, subset_frame in [
            ("in_scope_only", frame.loc[~frame["has_out_of_scope_label"].astype(bool)].copy()),
            ("out_of_scope_only", frame.loc[frame["has_out_of_scope_label"].astype(bool)].copy()),
        ]:
            if subset_frame.empty:
                continue
            subset_true, subset_prob = frame_targets_and_probabilities(subset_frame, labels)
            subset_metrics = compute_metrics(
                subset_true,
                subset_prob,
                labels,
                threshold=thresholds,
                run_id=metrics.get("run_id"),
            )
            subsets[subset_name] = {"row_count": int(len(subset_frame)), "metrics": subset_metrics}
    if subsets:
        payload["subsets"] = subsets
    return payload


def frame_targets_and_probabilities(frame: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    true = np.stack([frame[f"true_{label_slug(label)}"].to_numpy(dtype=int) for label in labels], axis=1)
    prob = np.stack([frame[f"prob_{label_slug(label)}"].to_numpy(dtype=float) for label in labels], axis=1)
    return true, prob


def confusion_counts(y_true: Any, y_pred: Any) -> dict[str, int]:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    return {
        "tp": int(np.sum((true == 1) & (pred == 1))),
        "fp": int(np.sum((true == 0) & (pred == 1))),
        "fn": int(np.sum((true == 1) & (pred == 0))),
        "tn": int(np.sum((true == 0) & (pred == 0))),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def average_precision(y_true: Any, y_score: Any) -> float | None:
    true = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    positives = int(np.sum(true == 1))
    if positives == 0 or positives == len(true):
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_true = true[order]
    precision_at_k = np.cumsum(sorted_true == 1) / np.arange(1, len(sorted_true) + 1)
    return float(np.sum(precision_at_k * (sorted_true == 1)) / positives)


def auroc(y_true: Any, y_score: Any) -> float | None:
    true = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    positives = score[true == 1]
    negatives = score[true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    comparisons = positives[:, None] - negatives[None, :]
    wins = np.sum(comparisons > 0)
    ties = np.sum(comparisons == 0)
    return float((wins + 0.5 * ties) / (len(positives) * len(negatives)))


def mean_defined(values: Any) -> float | None:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return float(np.mean(defined))


def resolve_thresholds(labels: list[str], threshold: float | Mapping[str, float]) -> dict[str, float]:
    if isinstance(threshold, Mapping):
        missing = [label for label in labels if label not in threshold]
        if missing:
            raise ValueError(f"Missing thresholds for labels: {missing}")
        return {label: float(threshold[label]) for label in labels}
    return {label: float(threshold) for label in labels}


def threshold_priors(
    y_true: Any,
    y_prob: Any,
    labels: list[str],
    *,
    default_threshold: float = 0.5,
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    if true.shape != prob.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape}, y_prob {prob.shape}")
    if true.ndim != 2 or true.shape[1] != len(labels):
        raise ValueError(f"Expected shape [n, {len(labels)}], got {true.shape}")
    priors = resolve_thresholds(labels, default_threshold)
    for label, prior in LABEL_THRESHOLD_PRIORS.items():
        if label in priors:
            priors[label] = float(prior)
    for index, label in enumerate(labels):
        if label not in DYNAMIC_THRESHOLD_PRIOR_LABELS:
            continue
        priors[label] = prevalence_matched_threshold(true[:, index], prob[:, index], default_threshold=priors[label])
    return priors


def prevalence_matched_threshold(y_true: Any, y_prob: Any, *, default_threshold: float = 0.5) -> float:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    positives = int(np.sum(true == 1))
    negatives = int(np.sum(true == 0))
    if positives == 0 or negatives == 0:
        return float(default_threshold)
    sorted_prob = np.sort(prob)
    target_index = min(max(len(sorted_prob) - positives, 0), len(sorted_prob) - 1)
    return float(sorted_prob[target_index])


def shared_threshold(thresholds: Mapping[str, float]) -> float | None:
    values = [float(value) for value in thresholds.values()]
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values[1:]):
        return first
    return None


def tune_thresholds(
    y_true: Any,
    y_prob: Any,
    labels: list[str],
    *,
    default_threshold: float = 0.5,
    low_support_threshold: float | Mapping[str, float] | None = None,
    min_positives_for_tuning: int = 0,
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    if true.shape != prob.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape}, y_prob {prob.shape}")
    if true.ndim != 2 or true.shape[1] != len(labels):
        raise ValueError(f"Expected shape [n, {len(labels)}], got {true.shape}")
    fallback_thresholds = resolve_thresholds(
        labels,
        default_threshold if low_support_threshold is None else low_support_threshold,
    )
    return {
        label: tune_binary_threshold(
            true[:, index],
            prob[:, index],
            default_threshold=default_threshold,
            prior_threshold=None if low_support_threshold is None else fallback_thresholds[label],
            fallback_threshold=fallback_thresholds[label],
            min_positives_for_tuning=min_positives_for_tuning,
        )
        for index, label in enumerate(labels)
    }


def tune_binary_threshold(
    y_true: Any,
    y_prob: Any,
    *,
    default_threshold: float = 0.5,
    prior_threshold: float | None = None,
    fallback_threshold: float | None = None,
    min_positives_for_tuning: int = 0,
) -> float:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    positives = int(np.sum(true == 1))
    negatives = int(np.sum(true == 0))
    anchor_threshold = float(default_threshold if prior_threshold is None else prior_threshold)
    if positives < min_positives_for_tuning:
        return float(anchor_threshold if fallback_threshold is None else fallback_threshold)
    if positives == 0 or negatives == 0:
        return float(default_threshold)
    order = np.argsort(-prob, kind="mergesort")
    sorted_true = true[order]
    sorted_prob = prob[order]
    tp = 0
    fp = 0
    best_f1 = -1.0
    best_threshold = float(anchor_threshold)
    lower_bound = None
    upper_bound = None
    if prior_threshold is not None:
        lower_bound = max(0.0, anchor_threshold - THRESHOLD_PRIOR_SEARCH_RADIUS)
        upper_bound = min(1.0, anchor_threshold + THRESHOLD_PRIOR_SEARCH_RADIUS)
    index = 0
    while index < len(sorted_prob):
        threshold = float(sorted_prob[index])
        while index < len(sorted_prob) and sorted_prob[index] == threshold:
            if sorted_true[index] == 1:
                tp += 1
            else:
                fp += 1
            index += 1
        if lower_bound is not None and upper_bound is not None and (threshold < lower_bound or threshold > upper_bound):
            continue
        fn = positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_threshold = threshold
            continue
        if abs(f1 - best_f1) > 1e-12:
            continue
        current_distance = abs(best_threshold - anchor_threshold)
        candidate_distance = abs(threshold - anchor_threshold)
        if candidate_distance < current_distance or (
            candidate_distance == current_distance and threshold > best_threshold
        ):
            best_threshold = threshold
    return best_threshold


def checkpoint_score(metrics: dict[str, Any], val_loss: float) -> float:
    disease_only = metrics.get("disease_only")
    if isinstance(disease_only, Mapping):
        disease_macro_ap = disease_only.get("macro_average_precision")
        if disease_macro_ap is not None:
            return float(disease_macro_ap)
    macro_ap = metrics.get("macro_average_precision")
    if macro_ap is not None:
        return float(macro_ap)
    return -float(val_loss)


def threshold_strategy_name() -> str:
    if MIN_TUNED_THRESHOLD_POSITIVES <= 0:
        return "per_label_f1_from_val_bounded_by_label_priors"
    return f"per_label_f1_from_val_bounded_by_label_priors_min_positives_{MIN_TUNED_THRESHOLD_POSITIVES}"
