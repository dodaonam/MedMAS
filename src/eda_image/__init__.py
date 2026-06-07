"""Reusable helpers for the NIH Chest X-ray image EDA notebook."""

from .paths import EdaPaths, resolve_paths
from .metadata import (
    AGE_BIN_LABELS,
    add_metadata_features,
    load_metadata,
    parse_age_to_years,
)
from .labels import (
    build_multi_hot,
    compute_cooccurrence,
    compute_age_bin_prevalence,
    compute_binary_group_risk,
    compute_label_summary,
    compute_labelset_summary,
    compute_pairwise_associations,
)
from .splits import compute_patient_dynamics, simulate_split_leakage
from .image_quality import (
    QUALITY_COLS,
    ahash_bits,
    compute_or_load_image_metrics,
    compute_rgba_consistency,
    load_gray,
    summarize_ahash_collisions,
)

__all__ = [
    "AGE_BIN_LABELS",
    "EdaPaths",
    "QUALITY_COLS",
    "add_metadata_features",
    "ahash_bits",
    "build_multi_hot",
    "compute_cooccurrence",
    "compute_age_bin_prevalence",
    "compute_binary_group_risk",
    "compute_label_summary",
    "compute_labelset_summary",
    "compute_or_load_image_metrics",
    "compute_pairwise_associations",
    "compute_patient_dynamics",
    "compute_rgba_consistency",
    "load_gray",
    "load_metadata",
    "parse_age_to_years",
    "resolve_paths",
    "simulate_split_leakage",
    "summarize_ahash_collisions",
]
