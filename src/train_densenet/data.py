from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ._backend import DataLoader, Dataset, WeightedRandomSampler, require_torch, torch
from .config import TrainConfig
from .constants import IMAGENET_MEAN, IMAGENET_STD, METADATA_COLUMNS


def load_manifest(path: Path, labels: list[str]) -> pd.DataFrame:
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
    required = [*METADATA_COLUMNS, *labels]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    split_values = set(frame["split"].dropna().astype(str))
    if split_values != {"train", "val", "test"}:
        raise ValueError(f"Expected train/val/test splits, got {sorted(split_values)}")
    for label in labels:
        values = set(frame[label].dropna().astype(int).unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"Label column {label!r} is not binary: {values}")
        frame[label] = frame[label].astype(int)
    assert_patient_disjoint(frame)
    return frame


def assert_patient_disjoint(frame: pd.DataFrame) -> None:
    patient_sets = {
        split: set(frame.loc[frame["split"].astype(str) == split, "Patient ID"].astype(str))
        for split in ["train", "val", "test"]
    }
    overlaps = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = patient_sets[left] & patient_sets[right]
        if overlap:
            overlaps[f"{left}_{right}"] = sorted(overlap)[:10]
    if overlaps:
        raise ValueError(f"Patient IDs overlap across splits: {overlaps}")


class ChestXrayDataset(Dataset):  # type: ignore[misc]
    def __init__(self, frame: pd.DataFrame, *, root: Path, split: str, labels: list[str], transform: Any) -> None:
        require_torch()
        self.frame = frame.loc[frame["split"].astype(str) == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No rows found for split {split!r}")
        self.root = root
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = self.root / image_path
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        target = torch.tensor(row[self.labels].to_numpy(dtype=np.float32), dtype=torch.float32)
        metadata = {column: row[column] for column in METADATA_COLUMNS if column in row.index}
        return image_tensor, target, metadata


def build_transforms(image_size: int) -> tuple[Any, Any]:
    from torchvision import transforms

    resize_size = image_size + 32
    train_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def denormalize_image_tensor(image_tensor: Any) -> Any:
    require_torch()
    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)


def build_dataloaders(config: TrainConfig, frame: pd.DataFrame, labels: list[str]) -> tuple[Any, Any, Any]:
    require_torch()
    train_transform, eval_transform = build_transforms(config.image_size)
    datasets = {
        "train": ChestXrayDataset(frame, root=config.root, split="train", labels=labels, transform=train_transform),
        "val": ChestXrayDataset(frame, root=config.root, split="val", labels=labels, transform=eval_transform),
        "test": ChestXrayDataset(frame, root=config.root, split="test", labels=labels, transform=eval_transform),
    }
    pin_memory = bool(torch.cuda.is_available())
    train_sampler = None
    if config.balanced_sampler:
        train_weights = balanced_sample_weights(frame, labels)
        train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    return (
        DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            datasets["val"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            datasets["test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
    )


def positive_weights(frame: pd.DataFrame, labels: list[str]) -> Any:
    require_torch()
    train = frame.loc[frame["split"].astype(str) == "train", labels].to_numpy(dtype=np.float32)
    positives = train.sum(axis=0)
    negatives = train.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def balanced_sample_weights(frame: pd.DataFrame, labels: list[str]) -> Any:
    require_torch()
    train = frame.loc[frame["split"].astype(str) == "train", labels].to_numpy(dtype=np.float32)
    positives = train.sum(axis=0)
    negatives = train.shape[0] - positives
    label_weights = np.maximum(negatives / np.maximum(positives, 1.0), 1.0)
    sample_weights = np.ones(train.shape[0], dtype=np.float64)
    positive_mask = train == 1
    for index, row_mask in enumerate(positive_mask):
        if np.any(row_mask):
            sample_weights[index] = float(np.max(label_weights[row_mask]))
    return torch.tensor(sample_weights, dtype=torch.double)
