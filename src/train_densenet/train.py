from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ModuleNotFoundError:  # pragma: no cover - only happens before torch is installed
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    WeightedRandomSampler = None  # type: ignore[assignment]


MODEL_NAME = "densenet121"
TARGET_LABELS = ["No Finding", "Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass"]
METADATA_COLUMNS = [
    "Image Index",
    "Patient ID",
    "split",
    "image_path",
    "Patient Gender",
    "View Position",
    "AgeBin",
    "has_out_of_scope_label",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MIN_TUNED_THRESHOLD_POSITIVES = 50
THRESHOLD_PRIOR_SEARCH_RADIUS = 0.10


@dataclass
class TrainConfig:
    root: Path
    manifest_path: Path
    target_labels_path: Path
    output_dir: Path
    seed: int = 0
    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    warmup_start_factor: float = 0.1
    min_lr: float = 1e-6
    threshold: float = 0.5
    balanced_sampler: bool = True
    pretrained: bool = True
    device: str | None = None


@dataclass(frozen=True)
class ArtifactPaths:
    run_dir: Path
    config_path: Path
    history_path: Path
    checkpoint_path: Path
    predictions_val_path: Path
    predictions_test_path: Path
    metrics_val_path: Path
    metrics_test_path: Path
    figures_dir: Path


def default_train_config(root: Path, output_dir: Path | None = None) -> TrainConfig:
    return TrainConfig(
        root=root,
        manifest_path=root / "artifacts" / "preprocess" / "split_manifest.csv",
        target_labels_path=root / "artifacts" / "preprocess" / "target_labels.json",
        output_dir=output_dir or root / "artifacts" / "training" / MODEL_NAME,
    )


def label_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower().strip())
    return slug.strip("_")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")


def load_target_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("target_labels", payload) if isinstance(payload, dict) else payload
    if labels != TARGET_LABELS:
        raise ValueError(f"Unexpected target labels in {path}: {labels!r}")
    return list(labels)


def create_run_id(seed: int, timestamp: datetime | None = None) -> str:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{MODEL_NAME}_seed{seed}_{stamp}"


def artifact_paths(run_dir: Path) -> ArtifactPaths:
    return ArtifactPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        history_path=run_dir / "training_history.csv",
        checkpoint_path=run_dir / "checkpoint_best.pt",
        predictions_val_path=run_dir / "predictions_val.csv",
        predictions_test_path=run_dir / "predictions_test.csv",
        metrics_val_path=run_dir / "metrics_val.json",
        metrics_test_path=run_dir / "metrics_test.json",
        figures_dir=run_dir / "figures",
    )


def load_checkpoint(path: Path, *, map_location: Any) -> dict[str, Any]:
    _require_torch()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)


def resolve_run_dir(path: Path) -> Path:
    if (path / "config.json").is_file():
        return path
    if path.is_dir():
        candidates = [child for child in path.iterdir() if (child / "config.json").is_file()]
        if candidates:
            return max(candidates, key=lambda child: child.stat().st_mtime)
    return path


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
        _require_torch()
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
    _require_torch()
    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)


def build_model(num_labels: int, *, pretrained: bool = True) -> Any:
    _require_torch()
    from torchvision.models import DenseNet121_Weights, densenet121

    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_labels)
    return model


def build_lr_scheduler(optimizer: Any, config: TrainConfig) -> Any:
    _require_torch()
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if config.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if config.warmup_epochs >= config.epochs:
        raise ValueError("warmup_epochs must be smaller than epochs")
    if not 0.0 < config.warmup_start_factor <= 1.0:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if config.min_lr < 0.0:
        raise ValueError("min_lr must be non-negative")

    cosine_t_max = _cosine_t_max(config)
    if config.warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=config.min_lr)

    warmup = LinearLR(
        optimizer,
        start_factor=config.warmup_start_factor,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=config.min_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs])


def train_model(config: TrainConfig) -> dict[str, Any]:
    _require_torch()
    labels = load_target_labels(config.target_labels_path)
    frame = load_manifest(config.manifest_path, labels)
    set_seed(config.seed)
    device = resolve_device(config.device)

    run_id = create_run_id(config.seed)
    paths = artifact_paths(config.output_dir / run_id)
    paths.run_dir.mkdir(parents=True, exist_ok=False)

    train_loader, val_loader, test_loader = build_dataloaders(config, frame, labels)
    model = build_model(len(labels), pretrained=config.pretrained).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weights(frame, labels).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_lr_scheduler(optimizer, config)

    config_payload = {
        **_config_to_json(config),
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "lr_scheduler": _scheduler_name(config),
        "cosine_t_max": _cosine_t_max(config),
        "threshold_strategy": _threshold_strategy_name(),
        "threshold_tuning_min_positives": MIN_TUNED_THRESHOLD_POSITIVES,
        "target_labels": labels,
        "threshold": config.threshold,
        "best_checkpoint": str(paths.checkpoint_path),
        "torch_version": str(getattr(torch, "__version__", "")),
        "device": str(device),
    }
    save_json(paths.config_path, config_payload)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        start = time.time()
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            desc=f"epoch {epoch}/{config.epochs} train",
        )
        val_loss, val_frame, y_true, y_prob = predict(
            model,
            val_loader,
            criterion,
            device,
            labels,
            threshold=config.threshold,
            run_id=run_id,
            desc=f"epoch {epoch}/{config.epochs} val",
        )
        val_thresholds = tune_thresholds(
            y_true,
            y_prob,
            labels,
            default_threshold=config.threshold,
            low_support_threshold=derive_threshold_priors(y_true, y_prob, labels, default_threshold=config.threshold),
            min_positives_for_tuning=MIN_TUNED_THRESHOLD_POSITIVES,
        )
        tuned_val_frame = prediction_frame(
            metadata_rows_from_prediction_frame(val_frame),
            y_true,
            y_prob,
            labels,
            threshold=val_thresholds,
            run_id=run_id,
        )
        val_metrics = compute_metrics(y_true, y_prob, labels, threshold=val_thresholds, run_id=run_id)
        val_metrics = attach_slice_metrics(val_metrics, tuned_val_frame, labels, threshold=val_thresholds)
        score = _score_for_checkpoint(val_metrics, val_loss)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": labels,
                    "config": config_payload,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                paths.checkpoint_path,
            )
            tuned_val_frame.to_csv(paths.predictions_val_path, index=False)
            save_json(paths.metrics_val_path, val_metrics)

        row = {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_average_precision": val_metrics["macro_average_precision"],
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": epoch_lr,
            "epoch_seconds": round(time.time() - start, 3),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(paths.history_path, index=False)
        print(
            f"epoch {epoch:03d}/{config.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_ap={_fmt_metric(row['val_macro_average_precision'])}"
        )
        scheduler.step()

    checkpoint = load_checkpoint(paths.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_frame, val_true, val_prob = predict(
        model,
        val_loader,
        criterion,
        device,
        labels,
        threshold=config.threshold,
        run_id=run_id,
        desc="best val",
    )
    tuned_thresholds = tune_thresholds(
        val_true,
        val_prob,
        labels,
        default_threshold=config.threshold,
        low_support_threshold=derive_threshold_priors(val_true, val_prob, labels, default_threshold=config.threshold),
        min_positives_for_tuning=MIN_TUNED_THRESHOLD_POSITIVES,
    )
    val_frame = prediction_frame(
        metadata_rows_from_prediction_frame(val_frame),
        val_true,
        val_prob,
        labels,
        threshold=tuned_thresholds,
        run_id=run_id,
    )
    test_loss, test_frame, test_true, test_prob = predict(
        model,
        test_loader,
        criterion,
        device,
        labels,
        threshold=config.threshold,
        run_id=run_id,
        desc="test",
    )
    test_frame = prediction_frame(
        metadata_rows_from_prediction_frame(test_frame),
        test_true,
        test_prob,
        labels,
        threshold=tuned_thresholds,
        run_id=run_id,
    )
    val_metrics = compute_metrics(val_true, val_prob, labels, threshold=tuned_thresholds, run_id=run_id)
    test_metrics = compute_metrics(test_true, test_prob, labels, threshold=tuned_thresholds, run_id=run_id)
    val_metrics = attach_slice_metrics(val_metrics, val_frame, labels, threshold=tuned_thresholds)
    test_metrics = attach_slice_metrics(test_metrics, test_frame, labels, threshold=tuned_thresholds)
    val_metrics["loss"] = val_loss
    test_metrics["loss"] = test_loss
    config_payload["selected_thresholds"] = tuned_thresholds
    config_payload["threshold_tuning_min_positives"] = MIN_TUNED_THRESHOLD_POSITIVES
    save_json(paths.config_path, config_payload)

    val_frame.to_csv(paths.predictions_val_path, index=False)
    test_frame.to_csv(paths.predictions_test_path, index=False)
    save_json(paths.metrics_val_path, val_metrics)
    save_json(paths.metrics_test_path, test_metrics)

    return {
        "run_id": run_id,
        "run_dir": str(paths.run_dir),
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "test_loss": test_loss,
        "test_macro_average_precision": test_metrics["macro_average_precision"],
        "test_macro_auroc": test_metrics["macro_auroc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "selected_thresholds": tuned_thresholds,
    }


def finalize_run(config: TrainConfig, run_dir: Path) -> dict[str, Any]:
    _require_torch()
    paths = artifact_paths(resolve_run_dir(run_dir))
    run_config = json.loads(paths.config_path.read_text(encoding="utf-8"))
    run_root = Path(run_config.get("root", config.root))
    manifest_path = Path(run_config.get("manifest_path", config.manifest_path))
    target_labels_path = Path(run_config.get("target_labels_path", config.target_labels_path))
    labels = list(run_config.get("target_labels") or load_target_labels(target_labels_path))
    device = resolve_device(config.device)
    run_id = str(run_config["run_id"])
    threshold = float(run_config.get("threshold", config.threshold))
    eval_config = replace(
        config,
        root=run_root,
        manifest_path=manifest_path,
        target_labels_path=target_labels_path,
        image_size=int(run_config.get("image_size", config.image_size)),
        threshold=threshold,
        pretrained=False,
    )
    frame = load_manifest(eval_config.manifest_path, labels)
    _train_loader, val_loader, test_loader = build_dataloaders(eval_config, frame, labels)
    model = build_model(len(labels), pretrained=False).to(device)
    checkpoint = load_checkpoint(paths.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weights(frame, labels).to(device))

    val_loss, val_frame, val_true, val_prob = predict(
        model,
        val_loader,
        criterion,
        device,
        labels,
        threshold=threshold,
        run_id=run_id,
        desc="best val",
    )
    tuned_thresholds = tune_thresholds(
        val_true,
        val_prob,
        labels,
        default_threshold=threshold,
        low_support_threshold=derive_threshold_priors(val_true, val_prob, labels, default_threshold=threshold),
        min_positives_for_tuning=MIN_TUNED_THRESHOLD_POSITIVES,
    )
    val_frame = prediction_frame(
        metadata_rows_from_prediction_frame(val_frame),
        val_true,
        val_prob,
        labels,
        threshold=tuned_thresholds,
        run_id=run_id,
    )
    test_loss, test_frame, test_true, test_prob = predict(
        model,
        test_loader,
        criterion,
        device,
        labels,
        threshold=threshold,
        run_id=run_id,
        desc="test",
    )
    test_frame = prediction_frame(
        metadata_rows_from_prediction_frame(test_frame),
        test_true,
        test_prob,
        labels,
        threshold=tuned_thresholds,
        run_id=run_id,
    )
    val_metrics = compute_metrics(val_true, val_prob, labels, threshold=tuned_thresholds, run_id=run_id)
    test_metrics = compute_metrics(test_true, test_prob, labels, threshold=tuned_thresholds, run_id=run_id)
    val_metrics = attach_slice_metrics(val_metrics, val_frame, labels, threshold=tuned_thresholds)
    test_metrics = attach_slice_metrics(test_metrics, test_frame, labels, threshold=tuned_thresholds)
    val_metrics["loss"] = val_loss
    test_metrics["loss"] = test_loss
    run_config["selected_thresholds"] = tuned_thresholds
    run_config["threshold_strategy"] = _threshold_strategy_name()
    run_config["threshold_tuning_min_positives"] = MIN_TUNED_THRESHOLD_POSITIVES
    save_json(paths.config_path, run_config)
    val_frame.to_csv(paths.predictions_val_path, index=False)
    test_frame.to_csv(paths.predictions_test_path, index=False)
    save_json(paths.metrics_val_path, val_metrics)
    save_json(paths.metrics_test_path, test_metrics)
    return {
        "run_id": run_id,
        "run_dir": str(paths.run_dir),
        "test_loss": test_loss,
        "test_macro_average_precision": test_metrics["macro_average_precision"],
        "test_macro_auroc": test_metrics["macro_auroc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "selected_thresholds": tuned_thresholds,
    }


def smoke_check(config: TrainConfig) -> dict[str, Any]:
    _require_torch()
    labels = load_target_labels(config.target_labels_path)
    frame = load_manifest(config.manifest_path, labels)
    train_loader, _, _ = build_dataloaders(config, frame, labels)
    device = resolve_device(config.device)
    model = build_model(len(labels), pretrained=config.pretrained).to(device)
    images, targets, _metadata = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
    return {
        "batch_shape": list(images.shape),
        "target_shape": list(targets.shape),
        "logit_shape": list(logits.shape),
        "device": str(device),
    }


def build_dataloaders(config: TrainConfig, frame: pd.DataFrame, labels: list[str]) -> tuple[Any, Any, Any]:
    _require_torch()
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


def train_one_epoch(model: Any, loader: Any, criterion: Any, optimizer: Any, device: Any, *, desc: str = "train") -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    progress = tqdm(loader, desc=desc, total=len(loader), dynamic_ncols=True, leave=False)
    for images, targets, _metadata in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = int(images.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_rows += batch_size
        progress.set_postfix(loss=f"{total_loss / max(total_rows, 1):.4f}")
    return total_loss / max(total_rows, 1)


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
    _require_torch()
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


def positive_weights(frame: pd.DataFrame, labels: list[str]) -> Any:
    _require_torch()
    train = frame.loc[frame["split"].astype(str) == "train", labels].to_numpy(dtype=np.float32)
    positives = train.sum(axis=0)
    negatives = train.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def balanced_sample_weights(frame: pd.DataFrame, labels: list[str]) -> Any:
    _require_torch()
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
            subsets[subset_name] = {
                "row_count": int(len(subset_frame)),
                "metrics": subset_metrics,
            }
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


def derive_threshold_priors(
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
    return {
        label: prevalence_matched_threshold(true[:, index], prob[:, index], default_threshold=default_threshold)
        for index, label in enumerate(labels)
    }


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


def set_seed(seed: int) -> None:
    _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> Any:
    _require_torch()
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _score_for_checkpoint(metrics: dict[str, Any], val_loss: float) -> float:
    disease_only = metrics.get("disease_only")
    if isinstance(disease_only, Mapping):
        disease_macro_ap = disease_only.get("macro_average_precision")
        if disease_macro_ap is not None:
            return float(disease_macro_ap)
    macro_ap = metrics.get("macro_average_precision")
    if macro_ap is not None:
        return float(macro_ap)
    return -float(val_loss)


def _scheduler_name(config: TrainConfig) -> str:
    return "linear_warmup_cosine_annealing" if config.warmup_epochs > 0 else "cosine_annealing"


def _threshold_strategy_name() -> str:
    if MIN_TUNED_THRESHOLD_POSITIVES <= 0:
        return "per_label_f1_from_val_bounded_by_rate_matched_prior"
    return f"per_label_f1_from_val_bounded_by_rate_matched_prior_min_positives_{MIN_TUNED_THRESHOLD_POSITIVES}"


def _cosine_t_max(config: TrainConfig) -> int:
    return max(config.epochs - config.warmup_epochs - 1, 1)


def _config_to_json(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _fmt_metric(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch and torchvision are required to train DenseNet121.")
