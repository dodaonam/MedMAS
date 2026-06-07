from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import PreprocessPaths
from .targets import TARGET_LABELS, add_target_columns, filter_to_target_scope, validate_target_columns


AGE_BINS = [0, 18, 40, 60, 80, 150, 500]
AGE_BIN_LABELS = ["0-18", "19-40", "41-60", "61-80", "81-150", "150+"]


def parse_age_to_years(age_value: object) -> tuple[float, str | None]:
    if age_value is None:
        return np.nan, None
    text = str(age_value).strip()
    if not text or text.lower() == "nan":
        return np.nan, None

    unit = text[-1].upper() if text[-1].isalpha() else "Y"
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return np.nan, unit

    value = float(digits)
    if unit == "Y":
        return value, unit
    if unit == "M":
        return value / 12.0, unit
    if unit == "D":
        return value / 365.0, unit
    return np.nan, unit


def load_source_metadata(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        dtype={
            "Image Index": "string",
            "Finding Labels": "string",
            "Follow-up #": "string",
            "Patient ID": "string",
            "Patient Age": "string",
            "Patient Gender": "string",
            "View Position": "string",
        },
    )


def compute_integrity_summary(df: pd.DataFrame, image_dir: Path) -> dict[str, int]:
    image_files = list(image_dir.glob("*.png"))
    csv_images = set(df["Image Index"].astype(str))
    disk_images = {path.name for path in image_files}
    return {
        "rows_csv": int(len(df)),
        "cols_csv": int(df.shape[1]),
        "unique_images_csv": int(df["Image Index"].nunique()),
        "images_on_disk": int(len(image_files)),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "csv_images_not_in_disk": int(len(csv_images - disk_images)),
        "disk_images_not_in_csv": int(len(disk_images - csv_images)),
    }


def add_metadata_features(df: pd.DataFrame, image_dir: Path, root: Path) -> pd.DataFrame:
    enriched = df.copy()
    age_parsed = enriched["Patient Age"].map(parse_age_to_years)
    enriched["AgeYears"] = age_parsed.map(lambda item: item[0])
    enriched["AgeUnit"] = age_parsed.map(lambda item: item[1])
    enriched["AgeBin"] = pd.cut(
        enriched["AgeYears"],
        bins=AGE_BINS,
        labels=AGE_BIN_LABELS,
        include_lowest=True,
    ).astype("string")
    enriched["AgeOutlierFlag"] = enriched["AgeYears"].gt(120).fillna(False)
    enriched["FollowUpNum"] = pd.to_numeric(enriched["Follow-up #"], errors="coerce")
    enriched["NumImagesForPatient"] = enriched.groupby("Patient ID")["Image Index"].transform("count").astype(int)
    image_dir_rel = image_dir.relative_to(root)
    enriched["image_path"] = enriched["Image Index"].map(lambda name: str(image_dir_rel / str(name)))
    enriched["image_path_exists"] = enriched["image_path"].map(lambda path: (root / path).exists())
    return enriched


def merge_optional_image_metrics(df: pd.DataFrame, image_metrics_path: Path) -> pd.DataFrame:
    if not image_metrics_path.exists():
        return df
    metrics = pd.read_csv(image_metrics_path)
    return df.merge(metrics, on="Image Index", how="left")


def build_manifest_all(paths: PreprocessPaths) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = load_source_metadata(paths.csv_path)
    integrity = compute_integrity_summary(raw, paths.image_dir)
    manifest = add_metadata_features(raw, paths.image_dir, paths.root)
    manifest = add_target_columns(manifest, TARGET_LABELS)
    manifest["NumLabels"] = manifest["Finding Labels"].map(lambda value: len(str(value).split("|"))).astype(int)
    manifest = merge_optional_image_metrics(manifest, paths.eda_image_metrics_path)
    validate_target_columns(manifest, TARGET_LABELS)
    return manifest, integrity


def build_manifest_filtered(manifest_all: pd.DataFrame) -> pd.DataFrame:
    filtered = filter_to_target_scope(manifest_all)
    validate_target_columns(filtered, TARGET_LABELS)
    return filtered


def write_preprocess_config(path: Path) -> None:
    config = {
        "phase": "filter_and_split_only",
        "source_package": "preprocess_image",
        "workspace": "artifacts/preprocess",
        "target_labels": TARGET_LABELS,
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "split_unit": "Patient ID",
        "seed_search": {"initial_start": 0, "initial_end": 100, "fallback_end": 1000},
        "keep_rule": "at_least_one_target_label",
        "drop_rule": "out_of_scope_only",
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
