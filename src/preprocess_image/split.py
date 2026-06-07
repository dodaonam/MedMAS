from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .targets import TARGET_LABELS


@dataclass(frozen=True)
class SplitSearchResult:
    manifest: pd.DataFrame
    selected_seed: int
    diagnostics: list[dict[str, object]]


def assign_patient_split(
    df: pd.DataFrame,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac must be between 0 and 1, got {train_frac}")
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be between 0 and 1, got {val_frac}")
    if train_frac + val_frac >= 1:
        raise ValueError(f"train_frac + val_frac must be less than 1, got {train_frac + val_frac}")

    rng = np.random.default_rng(seed)
    patients = df["Patient ID"].drop_duplicates().to_numpy()
    rng.shuffle(patients)

    n_patients = len(patients)
    n_train = int(n_patients * train_frac)
    n_val = int(n_patients * val_frac)
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients = set(patients[n_train + n_val :])

    split_by_patient = {patient: "train" for patient in train_patients}
    split_by_patient.update({patient: "val" for patient in val_patients})
    split_by_patient.update({patient: "test" for patient in test_patients})

    split_manifest = df.copy()
    split_manifest["split"] = split_manifest["Patient ID"].map(split_by_patient)
    return split_manifest


def patient_overlap_matrix(split_manifest: pd.DataFrame) -> pd.DataFrame:
    splits = ["train", "val", "test"]
    patient_sets = {
        split: set(split_manifest.loc[split_manifest["split"] == split, "Patient ID"])
        for split in splits
    }
    rows = []
    for split_a in splits:
        rows.append([len(patient_sets[split_a] & patient_sets[split_b]) for split_b in splits])
    return pd.DataFrame(rows, index=splits, columns=splits)


def target_counts_by_split(split_manifest: pd.DataFrame, target_labels: list[str] | None = None) -> pd.DataFrame:
    targets = target_labels or TARGET_LABELS
    return split_manifest.groupby("split", observed=False)[targets].sum().reindex(["train", "val", "test"]).fillna(0).astype(int)


def validate_split(split_manifest: pd.DataFrame, target_labels: list[str] | None = None) -> tuple[bool, dict[str, object]]:
    targets = target_labels or TARGET_LABELS
    split_values = set(split_manifest["split"].dropna().unique().tolist())
    expected_splits = {"train", "val", "test"}
    overlap = patient_overlap_matrix(split_manifest)
    counts = target_counts_by_split(split_manifest, targets)
    missing_by_split = {
        split: [label for label in targets if int(counts.loc[split, label]) == 0]
        for split in ["train", "val", "test"]
    }
    diagnostics: dict[str, object] = {
        "split_values": sorted(split_values),
        "row_counts": split_manifest["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
        "patient_counts": split_manifest.groupby("split", observed=False)["Patient ID"].nunique().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
        "off_diagonal_patient_overlap": int(overlap.values.sum() - np.trace(overlap.values)),
        "missing_labels_by_split": missing_by_split,
    }
    passes = (
        split_values == expected_splits
        and min(diagnostics["row_counts"].values()) > 0
        and diagnostics["off_diagonal_patient_overlap"] == 0
        and all(len(labels) == 0 for labels in missing_by_split.values())
    )
    return passes, diagnostics


def search_patient_split(
    df: pd.DataFrame,
    target_labels: list[str] | None = None,
    initial_start: int = 0,
    initial_end: int = 100,
    fallback_end: int = 1000,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitSearchResult:
    targets = target_labels or TARGET_LABELS
    diagnostics: list[dict[str, object]] = []
    for seed in range(initial_start, fallback_end + 1):
        split_manifest = assign_patient_split(df, seed, train_frac=train_frac, val_frac=val_frac)
        passes, seed_diagnostics = validate_split(split_manifest, targets)
        seed_diagnostics = {"seed": seed, **seed_diagnostics}
        diagnostics.append(seed_diagnostics)
        if passes:
            return SplitSearchResult(split_manifest, seed, diagnostics)
        if seed == initial_end:
            continue

    ranked = sorted(
        diagnostics,
        key=lambda item: (
            sum(len(labels) for labels in item["missing_labels_by_split"].values()),
            item["off_diagonal_patient_overlap"],
            -min(item["row_counts"].values()),
        ),
    )[:5]
    raise RuntimeError(f"No patient-wise split passed gates through seed {fallback_end}. Best candidates: {ranked}")
