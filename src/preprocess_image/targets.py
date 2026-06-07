from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_LABELS: list[str] = [
    "No Finding",
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
]

TARGET_LABEL_SET = set(TARGET_LABELS)


def parse_labels(value: object) -> list[str]:
    if value is None:
        return []
    labels = [part.strip() for part in str(value).split("|")]
    return [label for label in labels if label]


def serialize_labels(labels: Iterable[str]) -> str:
    return json.dumps(list(labels), ensure_ascii=True)


def build_target_frame(label_series: pd.Series, target_labels: list[str] | None = None) -> pd.DataFrame:
    targets = target_labels or TARGET_LABELS
    target_set = set(targets)
    label_lists = label_series.map(parse_labels)
    label_sets = label_lists.map(set)
    target_df = pd.DataFrame(index=label_series.index)
    for label in targets:
        target_df[label] = label_sets.map(lambda labels: int(label in labels)).astype(np.uint8)
    target_df["has_any_target"] = target_df[targets].sum(axis=1).gt(0)
    target_df["keep_for_mvp"] = target_df["has_any_target"]
    target_df["out_of_scope_labels"] = label_lists.map(
        lambda labels: serialize_labels(label for label in labels if label not in target_set)
    )
    target_df["has_out_of_scope_label"] = target_df["out_of_scope_labels"].map(lambda value: len(json.loads(value)) > 0)
    return target_df


def add_target_columns(df: pd.DataFrame, target_labels: list[str] | None = None) -> pd.DataFrame:
    targets = target_labels or TARGET_LABELS
    enriched = df.copy()
    label_lists = enriched["Finding Labels"].map(parse_labels)
    enriched["LabelList"] = label_lists.map(serialize_labels)
    target_df = build_target_frame(enriched["Finding Labels"], targets)
    for column in target_df.columns:
        enriched[column] = target_df[column]
    return enriched


def filter_to_target_scope(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["keep_for_mvp"].astype(bool)].copy()


def validate_target_columns(df: pd.DataFrame, target_labels: list[str] | None = None) -> None:
    targets = target_labels or TARGET_LABELS
    missing = [label for label in targets if label not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    for label in targets:
        values = set(df[label].dropna().unique().tolist())
        if not values.issubset({0, 1, np.uint8(0), np.uint8(1)}):
            raise ValueError(f"Target column {label!r} contains non-binary values: {values}")

    no_finding_mixed = df["Finding Labels"].map(lambda value: "No Finding" in parse_labels(value) and len(parse_labels(value)) > 1)
    if bool(no_finding_mixed.any()):
        examples = df.loc[no_finding_mixed, "Finding Labels"].head(5).tolist()
        raise ValueError(f"`No Finding` appears with disease labels: {examples}")


def write_target_labels(path: Path, target_labels: list[str] | None = None) -> None:
    targets = target_labels or TARGET_LABELS
    path.write_text(json.dumps({"target_labels": targets}, indent=2), encoding="utf-8")
