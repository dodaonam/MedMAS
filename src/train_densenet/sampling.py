from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import DISEASE_LABELS, RARE_SAMPLER_WEAKCROP_VARIANT, V2_LOCKED_VARIANT

try:
    import torch
    from torch.utils.data import WeightedRandomSampler
except ModuleNotFoundError:  # pragma: no cover - exercised only on machines without torch
    torch = None  # type: ignore[assignment]
    WeightedRandomSampler = None  # type: ignore[assignment]


RARE_SAMPLER_LABELS = ["Nodule", "Mass"]


def disabled_sampler_config(recipe_variant: str = V2_LOCKED_VARIANT) -> dict[str, Any]:
    return {
        "recipe_variant": recipe_variant,
        "rare_sampler_enabled": False,
        "reason": "normal shuffled train batches",
    }


def rare_label_sampler_config_and_weights(
    train_frame: pd.DataFrame,
    *,
    labels: list[str] | None = None,
    rare_labels: list[str] | None = None,
    max_weight: float = 2.0,
    replacement: bool = True,
    num_samples: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    target_labels = labels or DISEASE_LABELS
    target_rare_labels = rare_labels or RARE_SAMPLER_LABELS
    missing = [label for label in [*target_labels, *target_rare_labels] if label not in train_frame.columns]
    if missing:
        raise ValueError(f"Missing labels for rare sampler: {missing}")
    if len(train_frame) == 0:
        raise ValueError("rare sampler requires a non-empty train frame")

    train_positive_counts = {
        label: int(train_frame[label].astype(int).sum())
        for label in target_labels
    }
    reference_label, reference_positive_count = max(
        train_positive_counts.items(),
        key=lambda item: item[1],
    )
    if reference_positive_count <= 0:
        raise ValueError("rare sampler requires at least one positive training label")

    boosts: dict[str, float] = {}
    weights = np.ones(len(train_frame), dtype=float)
    for label in target_rare_labels:
        positive_count = train_positive_counts[label]
        boost = 1.0 if positive_count <= 0 else math.sqrt(reference_positive_count / positive_count)
        boost = min(max(float(boost), 1.0), float(max_weight))
        boosts[label] = boost
        mask = train_frame[label].astype(int).to_numpy() == 1
        weights[mask] = np.maximum(weights[mask], boost)

    sample_count = int(num_samples or len(train_frame))
    weight_sum = float(weights.sum())
    expected_counts = {
        label: float(sample_count * np.sum(weights * train_frame[label].astype(int).to_numpy()) / weight_sum)
        for label in target_labels
    }
    config = {
        "recipe_variant": RARE_SAMPLER_WEAKCROP_VARIANT,
        "rare_sampler_enabled": True,
        "rare_sampler_labels": list(target_rare_labels),
        "rare_sampler_reference": "max_train_positive_count",
        "reference_label": reference_label,
        "reference_positive_count": int(reference_positive_count),
        "rare_sampler_boost": boosts,
        "rare_sampler_max_weight": float(max_weight),
        "rare_sampler_num_samples": sample_count,
        "rare_sampler_replacement": bool(replacement),
        "train_positive_counts": train_positive_counts,
        "expected_sampled_positive_counts_per_epoch": expected_counts,
        "sample_weight_stats": {
            "min": float(weights.min()),
            "max": float(weights.max()),
            "mean": float(weights.mean()),
            "weighted_row_count": int(np.sum(weights > 1.0)),
        },
    }
    return config, weights


def create_weighted_random_sampler(
    weights: np.ndarray,
    *,
    num_samples: int,
    replacement: bool,
    seed: int,
) -> Any:
    if torch is None or WeightedRandomSampler is None:
        raise ModuleNotFoundError("PyTorch is required for WeightedRandomSampler.")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples),
        replacement=bool(replacement),
        generator=generator,
    )
