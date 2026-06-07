from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AGE_BINS = [0, 18, 40, 60, 80, 150, 500]
AGE_BIN_LABELS = ["0-18", "19-40", "41-60", "61-80", "81-150", "150+"]


def parse_age_to_years(age_value: object) -> tuple[float, str | None]:
    """Parse NIH-style age strings such as 060Y, 013M, and 001D into years."""
    if age_value is None:
        return np.nan, None
    s = str(age_value).strip()
    if not s or s.lower() == "nan":
        return np.nan, None

    unit = s[-1].upper() if s[-1].isalpha() else "Y"
    digits = "".join(ch for ch in s if ch.isdigit())
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


def load_metadata(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    age_parsed = enriched["Patient Age"].apply(parse_age_to_years)
    enriched["AgeYears"] = age_parsed.map(lambda item: item[0])
    enriched["AgeUnit"] = age_parsed.map(lambda item: item[1])
    enriched["FollowUpNum"] = pd.to_numeric(enriched["Follow-up #"], errors="coerce")
    enriched["LabelList"] = enriched["Finding Labels"].str.split("|")
    enriched["NumLabels"] = enriched["LabelList"].str.len()
    enriched["OriginalResolution"] = (
        enriched[["OriginalImageWidth", "OriginalImageHeight"]]
        .astype(int)
        .astype(str)
        .agg("x".join, axis=1)
    )
    enriched["AgeBin"] = pd.cut(
        enriched["AgeYears"],
        bins=AGE_BINS,
        labels=AGE_BIN_LABELS,
        include_lowest=True,
    )
    return enriched


def compute_integrity_summary(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    image_files = list(image_dir.glob("*.png"))
    csv_images = set(df["Image Index"])
    disk_images = {p.name for p in image_files}
    rows = [
        ("rows_csv", len(df)),
        ("cols_csv", df.shape[1]),
        ("unique_images_csv", df["Image Index"].nunique()),
        ("images_on_disk", len(image_files)),
        ("missing_values_total", int(df.isna().sum().sum())),
        ("duplicated_rows", int(df.duplicated().sum())),
        ("csv_images_not_in_disk", len(csv_images - disk_images)),
        ("disk_images_not_in_csv", len(disk_images - csv_images)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])

