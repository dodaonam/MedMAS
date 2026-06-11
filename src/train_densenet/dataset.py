from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .artifacts import DISEASE_LABELS, TARGET_LABELS

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised only on machines without torch
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


REQUIRED_METADATA_COLUMNS = [
    "image_path",
    "split",
    "Patient ID",
    "Image Index",
    "Patient Gender",
    "View Position",
    "AgeBin",
    "has_out_of_scope_label",
]

REQUIRED_COLUMNS = [*REQUIRED_METADATA_COLUMNS, *TARGET_LABELS]

METADATA_COLUMNS = [
    "Image Index",
    "Patient ID",
    "split",
    "image_path",
    "Patient Gender",
    "View Position",
    "AgeBin",
    "has_out_of_scope_label",
    "No Finding",
]


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for DenseNet dataset loading. Install torch and torchvision first.")


def load_split_manifest(path: Path, target_labels: list[str] | None = None) -> pd.DataFrame:
    labels = target_labels or DISEASE_LABELS
    frame = pd.read_csv(
        path,
        dtype={
            "Image Index": "string",
            "Patient ID": "string",
            "Patient Gender": "string",
            "View Position": "string",
            "AgeBin": "string",
            "split": "string",
            "image_path": "string",
        },
    )
    validate_manifest_columns(frame, labels)
    return frame


def validate_manifest_columns(frame: pd.DataFrame, target_labels: list[str] | None = None) -> None:
    labels = target_labels or DISEASE_LABELS
    required = [*REQUIRED_METADATA_COLUMNS, "No Finding", *labels]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required manifest columns: {missing}")
    split_values = set(frame["split"].dropna().astype(str).unique().tolist())
    expected = {"train", "val", "test"}
    if split_values != expected:
        raise ValueError(f"Expected split values {sorted(expected)}, got {sorted(split_values)}")
    for label in labels:
        values = set(frame[label].dropna().astype(int).unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"Target column {label!r} contains non-binary values: {values}")


def assert_patient_disjoint(frame: pd.DataFrame) -> None:
    patient_sets = {
        split: set(frame.loc[frame["split"].astype(str) == split, "Patient ID"].astype(str))
        for split in ["train", "val", "test"]
    }
    overlaps = {
        f"{left}_{right}": sorted(patient_sets[left] & patient_sets[right])[:10]
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]
        if patient_sets[left] & patient_sets[right]
    }
    if overlaps:
        raise ValueError(f"Patient IDs overlap across splits: {overlaps}")


class ChestXrayMultiLabelDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        root: Path,
        split: str,
        target_labels: list[str] | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        _require_torch()
        self.target_labels = target_labels or DISEASE_LABELS
        validate_manifest_columns(manifest, self.target_labels)
        self.root = root
        self.split = split
        self.transform = transform
        self.frame = manifest.loc[manifest["split"].astype(str) == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No rows found for split {split!r}")

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        _require_torch()
        row = self.frame.iloc[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = self.root / image_path
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            image_tensor = self.transform(image_rgb) if self.transform is not None else image_rgb
        target_tensor = torch.tensor(row[self.target_labels].astype(float).to_numpy(), dtype=torch.float32)
        metadata = {column: row[column] for column in METADATA_COLUMNS if column in row.index}
        metadata["resolved_image_path"] = str(image_path)
        return image_tensor, target_tensor, metadata


def create_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    sampler: Any | None = None,
) -> Any:
    _require_torch()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
