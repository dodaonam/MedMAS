from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .artifacts import DISEASE_LABELS, TARGET_LABELS, label_slug


def _as_1d_float(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _as_1d_int(values: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=int).reshape(-1)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def confusion_counts(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> dict[str, int]:
    true = _as_1d_int(y_true)
    pred = _as_1d_int(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape}, y_pred {pred.shape}")
    return {
        "tp": int(np.sum((true == 1) & (pred == 1))),
        "fp": int(np.sum((true == 0) & (pred == 1))),
        "tn": int(np.sum((true == 0) & (pred == 0))),
        "fn": int(np.sum((true == 1) & (pred == 0))),
    }


def metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    tp = float(counts["tp"])
    fp = float(counts["fp"])
    tn = float(counts["tn"])
    fn = float(counts["fn"])
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def safe_auroc(y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> dict[str, Any]:
    true = _as_1d_int(y_true)
    score = _as_1d_float(y_score)
    positives = int(np.sum(true == 1))
    negatives = int(np.sum(true == 0))
    if positives == 0 or negatives == 0:
        return {
            "value": None,
            "reason": "not_defined_one_class_y_true",
            "positive_count": positives,
            "negative_count": negatives,
        }
    ranks = _average_ranks(score)
    positive_rank_sum = float(np.sum(ranks[true == 1]))
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return {
        "value": float(auc),
        "reason": None,
        "positive_count": positives,
        "negative_count": negatives,
    }


def safe_average_precision(y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> dict[str, Any]:
    true = _as_1d_int(y_true)
    score = _as_1d_float(y_score)
    positives = int(np.sum(true == 1))
    negatives = int(np.sum(true == 0))
    if positives == 0 or negatives == 0:
        return {
            "value": None,
            "reason": "not_defined_one_class_y_true",
            "positive_count": positives,
            "negative_count": negatives,
        }
    order = np.argsort(-score, kind="mergesort")
    sorted_true = true[order]
    cumulative_tp = np.cumsum(sorted_true == 1)
    ranks = np.arange(1, len(sorted_true) + 1)
    precision_at_rank = cumulative_tp / ranks
    ap = float(np.sum(precision_at_rank[sorted_true == 1]) / positives)
    return {
        "value": ap,
        "reason": None,
        "positive_count": positives,
        "negative_count": negatives,
    }


def compute_label_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    true = _as_1d_int(y_true)
    prob = _as_1d_float(y_prob)
    pred = (prob >= float(threshold)).astype(int)
    counts = confusion_counts(true, pred)
    thresholded = metrics_from_counts(counts)
    auroc = safe_auroc(true, prob)
    average_precision = safe_average_precision(true, prob)
    return {
        "positive_count": int(np.sum(true == 1)),
        "negative_count": int(np.sum(true == 0)),
        "threshold": float(threshold),
        "auroc": auroc["value"],
        "auroc_reason": auroc["reason"],
        "average_precision": average_precision["value"],
        "average_precision_reason": average_precision["reason"],
        **counts,
        **thresholded,
    }


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return float(np.mean(defined))


def _threshold_sequence(thresholds: Mapping[str, float] | Sequence[float], labels: list[str]) -> list[float]:
    if isinstance(thresholds, Mapping):
        return [float(thresholds[label]) for label in labels]
    values = list(thresholds)
    if len(values) != len(labels):
        raise ValueError(f"Expected {len(labels)} thresholds, got {len(values)}")
    return [float(value) for value in values]


def compute_multilabel_metrics(
    y_true: Sequence[Sequence[int]] | np.ndarray,
    y_prob: Sequence[Sequence[float]] | np.ndarray,
    thresholds: Mapping[str, float] | Sequence[float],
    labels: list[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    target_labels = labels or TARGET_LABELS
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    if true.shape != prob.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape}, y_prob {prob.shape}")
    if true.ndim != 2 or true.shape[1] != len(target_labels):
        raise ValueError(f"Expected shape [n, {len(target_labels)}], got {true.shape}")

    threshold_values = _threshold_sequence(thresholds, target_labels)
    per_label: dict[str, dict[str, Any]] = {}
    for idx, label in enumerate(target_labels):
        per_label[label] = compute_label_metrics(true[:, idx], prob[:, idx], threshold_values[idx])

    disease_names = [label for label in DISEASE_LABELS if label in per_label]
    all_counts = {
        "tp": int(sum(per_label[label]["tp"] for label in target_labels)),
        "fp": int(sum(per_label[label]["fp"] for label in target_labels)),
        "tn": int(sum(per_label[label]["tn"] for label in target_labels)),
        "fn": int(sum(per_label[label]["fn"] for label in target_labels)),
    }
    micro = metrics_from_counts(all_counts)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "target_label_order": target_labels,
        "per_label": per_label,
        "disease_macro_average_precision": _mean_defined([per_label[label]["average_precision"] for label in disease_names]),
        "disease_macro_auroc": _mean_defined([per_label[label]["auroc"] for label in disease_names]),
        "disease_macro_f1": _mean_defined([per_label[label]["f1"] for label in disease_names]),
        "all_label_macro_average_precision": _mean_defined([per_label[label]["average_precision"] for label in target_labels]),
        "all_label_macro_auroc": _mean_defined([per_label[label]["auroc"] for label in target_labels]),
        "all_label_macro_f1": _mean_defined([per_label[label]["f1"] for label in target_labels]),
        "all_label_micro_precision": micro["precision"],
        "all_label_micro_recall": micro["recall"],
        "all_label_micro_specificity": micro["specificity"],
        "all_label_micro_f1": micro["f1"],
        "micro_counts": all_counts,
    }
    if target_labels and target_labels[0] == "No Finding":
        payload["raw_no_finding_metrics"] = per_label["No Finding"]
    return payload


def prediction_columns_for(labels: list[str]) -> dict[str, list[str]]:
    return {
        "true": [f"true_{label_slug(label)}" for label in labels],
        "logit": [f"logit_{label_slug(label)}" for label in labels],
        "prob": [f"prob_{label_slug(label)}" for label in labels],
        "threshold": [f"threshold_{label_slug(label)}" for label in labels],
        "pred": [f"pred_{label_slug(label)}" for label in labels],
    }


def compute_metrics_from_prediction_frame(
    frame: Any,
    labels: list[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    target_labels = labels or TARGET_LABELS
    columns = prediction_columns_for(target_labels)
    true = frame[columns["true"]].to_numpy(dtype=int)
    prob = frame[columns["prob"]].to_numpy(dtype=float)
    thresholds = {
        label: float(frame[f"threshold_{label_slug(label)}"].dropna().iloc[0])
        for label in target_labels
    }
    return compute_multilabel_metrics(true, prob, thresholds, target_labels, run_id=run_id)


def compute_subgroup_metrics_from_frame(
    frame: Any,
    group_column: str,
    labels: list[str] | None = None,
    min_positives: int = 20,
) -> dict[str, Any]:
    target_labels = labels or TARGET_LABELS
    result: dict[str, Any] = {"group_column": group_column, "groups": {}}
    columns = prediction_columns_for(target_labels)
    for group_value, group_frame in frame.groupby(group_column, dropna=False):
        group_key = "missing" if group_value != group_value else str(group_value)
        group_payload: dict[str, Any] = {"row_count": int(len(group_frame)), "per_label": {}}
        for label, true_col, prob_col, threshold_col in zip(
            target_labels,
            columns["true"],
            columns["prob"],
            columns["threshold"],
            strict=True,
        ):
            y_true = group_frame[true_col].to_numpy(dtype=int)
            positives = int(np.sum(y_true == 1))
            negatives = int(np.sum(y_true == 0))
            label_payload = {
                "positive_count": positives,
                "negative_count": negatives,
                "prevalence": _safe_div(float(positives), float(len(y_true))),
            }
            if positives >= min_positives and negatives > 0:
                threshold = float(group_frame[threshold_col].dropna().iloc[0])
                label_payload.update(compute_label_metrics(y_true, group_frame[prob_col].to_numpy(dtype=float), threshold))
                label_payload["reported"] = True
            else:
                label_payload["reported"] = False
                label_payload["reason"] = "insufficient_positives_or_no_negatives"
            group_payload["per_label"][label] = label_payload
        result["groups"][group_key] = group_payload
    return result
