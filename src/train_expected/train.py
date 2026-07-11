from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    DataLoader = None
    Dataset = object


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
LABEL_THRESHOLD_PRIORS = {
    "No Finding": 0.31707045435905457,
    "Infiltration": 0.46902427077293396,
    "Effusion": 0.5468099117279053,
    "Atelectasis": 0.36542871594429016,
    "Nodule": 0.6589330434799194,
    "Mass": 0.5728916525840759,
}
DYNAMIC_THRESHOLD_PRIOR_LABELS = {"Infiltration"}

TRAINING_RECIPE: dict[str, Any] = {
    "model_name": MODEL_NAME,
    "seed": 0,
    "epochs": 10,
    "batch_size": 32,
    "num_workers": 4,
    "image_size": 320,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 1,
    "warmup_start_factor": 0.1,
    "min_lr": 1e-6,
    "threshold": 0.5,
    "pretrained": True,
    "device": "cuda",
    "classifier_head": "linear",
    "loss": "bce_with_logits_pos_weight",
    "sampler": "shuffle",
}


@dataclass(frozen=True)
class TrainConfig:
    root: Path
    manifest_path: Path
    target_labels_path: Path
    output_dir: Path
    seed: int = field(default=0, init=False)
    epochs: int = field(default=10, init=False)
    batch_size: int = field(default=32, init=False)
    num_workers: int = field(default=4, init=False)
    image_size: int = field(default=320, init=False)
    lr: float = field(default=1e-4, init=False)
    weight_decay: float = field(default=1e-4, init=False)
    warmup_epochs: int = field(default=1, init=False)
    warmup_start_factor: float = field(default=0.1, init=False)
    min_lr: float = field(default=1e-6, init=False)
    threshold: float = field(default=0.5, init=False)
    pretrained: bool = field(default=True, init=False)
    device: str = field(default="cuda", init=False)


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


def default_config(root: Path) -> TrainConfig:
    return TrainConfig(
        root=root,
        manifest_path=root / "artifacts" / "preprocess" / "split_manifest.csv",
        target_labels_path=root / "artifacts" / "preprocess" / "target_labels.json",
        output_dir=root / "artifacts" / "training" / MODEL_NAME,
    )


def train_model(config: TrainConfig) -> dict[str, Any]:
    require_torch()
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

    config_payload = build_config_payload(config, labels, run_id, paths.checkpoint_path, device)
    save_json(paths.config_path, config_payload)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        start = time.time()
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, desc=f"epoch {epoch} train")
        val_loss, val_frame, y_true, y_prob = predict(
            model, val_loader, criterion, device, labels, threshold=config.threshold, run_id=run_id, desc=f"epoch {epoch} val"
        )
        val_thresholds = tune_thresholds(
            y_true,
            y_prob,
            labels,
            default_threshold=config.threshold,
            low_support_threshold=threshold_priors(y_true, y_prob, labels, default_threshold=config.threshold),
            min_positives_for_tuning=MIN_TUNED_THRESHOLD_POSITIVES,
        )
        val_frame = prediction_frame(
            metadata_rows_from_prediction_frame(val_frame),
            y_true,
            y_prob,
            labels,
            threshold=val_thresholds,
            run_id=run_id,
        )
        val_metrics = compute_metrics(y_true, y_prob, labels, threshold=val_thresholds, run_id=run_id)
        val_metrics = attach_slice_metrics(val_metrics, val_frame, labels, threshold=val_thresholds)
        score = checkpoint_score(val_metrics, val_loss)
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
            val_frame.to_csv(paths.predictions_val_path, index=False)
            save_json(paths.metrics_val_path, val_metrics)

        history.append(
            {
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
        )
        pd.DataFrame(history).to_csv(paths.history_path, index=False)
        print(
            f"epoch {epoch:03d}/{config.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_ap={format_metric(val_metrics['macro_average_precision'])}"
        )
        scheduler.step()

    checkpoint = load_checkpoint(paths.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_frame, val_true, val_prob = predict(
        model, val_loader, criterion, device, labels, threshold=config.threshold, run_id=run_id, desc="best val"
    )
    tuned_thresholds = tune_thresholds(
        val_true,
        val_prob,
        labels,
        default_threshold=config.threshold,
        low_support_threshold=threshold_priors(val_true, val_prob, labels, default_threshold=config.threshold),
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
        model, test_loader, criterion, device, labels, threshold=config.threshold, run_id=run_id, desc="test"
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


def smoke_check(config: TrainConfig) -> dict[str, Any]:
    require_torch()
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


def build_config_payload(config: TrainConfig, labels: list[str], run_id: str, checkpoint_path: Path, device: Any) -> dict[str, Any]:
    payload = asdict(config)
    payload.update(
        {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "lr_scheduler": scheduler_name(config),
            "cosine_t_max": cosine_t_max(config),
            "threshold_strategy": threshold_strategy_name(),
            "threshold_tuning_min_positives": MIN_TUNED_THRESHOLD_POSITIVES,
            "target_labels": labels,
            "best_checkpoint": str(checkpoint_path),
            "torch_version": str(getattr(torch, "__version__", "")) if torch is not None else None,
            "device": str(device),
            "recipe": TRAINING_RECIPE,
        }
    )
    return payload


def require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch and torchvision are required to train DenseNet121.")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=json_default), encoding="utf-8")


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
    )


def load_checkpoint(path: Path, *, map_location: Any) -> dict[str, Any]:
    require_torch()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)


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


def build_dataloaders(config: TrainConfig, frame: pd.DataFrame, labels: list[str]) -> tuple[Any, Any, Any]:
    require_torch()
    train_transform, eval_transform = build_transforms(config.image_size)
    datasets = {
        "train": ChestXrayDataset(frame, root=config.root, split="train", labels=labels, transform=train_transform),
        "val": ChestXrayDataset(frame, root=config.root, split="val", labels=labels, transform=eval_transform),
        "test": ChestXrayDataset(frame, root=config.root, split="test", labels=labels, transform=eval_transform),
    }
    pin_memory = bool(torch.cuda.is_available())
    return (
        DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=pin_memory),
        DataLoader(datasets["val"], batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=pin_memory),
        DataLoader(datasets["test"], batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=pin_memory),
    )


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


def build_model(num_labels: int, *, pretrained: bool = True) -> Any:
    require_torch()
    from torchvision.models import DenseNet121_Weights, densenet121

    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_labels)
    return model


def positive_weights(frame: pd.DataFrame, labels: list[str]) -> Any:
    require_torch()
    train = frame.loc[frame["split"].astype(str) == "train", labels].to_numpy(dtype=np.float32)
    positives = train.sum(axis=0)
    negatives = train.shape[0] - positives
    return torch.tensor(negatives / np.maximum(positives, 1.0), dtype=torch.float32)


def build_lr_scheduler(optimizer: Any, config: TrainConfig) -> Any:
    require_torch()
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup = LinearLR(optimizer, start_factor=config.warmup_start_factor, end_factor=1.0, total_iters=config.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_t_max(config), eta_min=config.min_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs])


def train_one_epoch(model: Any, loader: Any, criterion: Any, optimizer: Any, device: Any, *, desc: str) -> float:
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
    threshold: float | Mapping[str, float],
    run_id: str,
    desc: str,
) -> tuple[float, pd.DataFrame, np.ndarray, np.ndarray]:
    require_torch()
    model.eval()
    total_loss = 0.0
    total_rows = 0
    metadata_rows: list[dict[str, Any]] = []
    targets_list: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    with torch.no_grad():
        progress = tqdm(loader, desc=desc, total=len(loader), dynamic_ncols=True, leave=False)
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
            progress.set_postfix(loss=f"{total_loss / max(total_rows, 1):.4f}")
    y_true = np.concatenate(targets_list, axis=0)
    y_prob = 1.0 / (1.0 + np.exp(-np.concatenate(logits_list, axis=0)))
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
    for row_index, metadata in enumerate(metadata_rows):
        row = dict(metadata)
        row["run_id"] = run_id
        for label_index, label in enumerate(labels):
            slug = label_slug(label)
            probability = float(y_prob[row_index, label_index])
            row[label] = int(y_true[row_index, label_index])
            row[f"true_{slug}"] = int(y_true[row_index, label_index])
            row[f"prob_{slug}"] = probability
            row[f"threshold_{slug}"] = thresholds[label]
            row[f"pred_{slug}"] = int(probability >= thresholds[label])
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


def compute_metrics(
    y_true: Any,
    y_prob: Any,
    labels: list[str],
    *,
    threshold: float | Mapping[str, float],
    run_id: str | None = None,
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    if true.shape != prob.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape}, y_prob {prob.shape}")
    thresholds = resolve_thresholds(labels, threshold)
    per_label: dict[str, Any] = {}
    micro_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for index, label in enumerate(labels):
        pred = (prob[:, index] >= thresholds[label]).astype(int)
        counts = confusion_counts(true[:, index], pred)
        for key in micro_counts:
            micro_counts[key] += counts[key]
        per_label[label] = {
            **counts,
            **metrics_from_counts(counts),
            "positive_count": int(np.sum(true[:, index] == 1)),
            "threshold": thresholds[label],
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


def attach_slice_metrics(metrics: dict[str, Any], frame: pd.DataFrame, labels: list[str], *, threshold: Mapping[str, float]) -> dict[str, Any]:
    payload = dict(metrics)
    disease_labels = [label for label in labels if label != "No Finding"]
    if disease_labels:
        disease_true, disease_prob = frame_targets_and_probabilities(frame, disease_labels)
        disease_metrics = compute_metrics(
            disease_true,
            disease_prob,
            disease_labels,
            threshold={label: threshold[label] for label in disease_labels},
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
    return payload


def frame_targets_and_probabilities(frame: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    true = np.stack([frame[f"true_{label_slug(label)}"].to_numpy(dtype=int) for label in labels], axis=1)
    prob = np.stack([frame[f"prob_{label_slug(label)}"].to_numpy(dtype=float) for label in labels], axis=1)
    return true, prob


def tune_thresholds(
    y_true: Any,
    y_prob: Any,
    labels: list[str],
    *,
    default_threshold: float,
    low_support_threshold: float | Mapping[str, float],
    min_positives_for_tuning: int,
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    fallbacks = resolve_thresholds(labels, low_support_threshold)
    return {
        label: tune_binary_threshold(
            true[:, index],
            prob[:, index],
            default_threshold=default_threshold,
            prior_threshold=fallbacks[label],
            fallback_threshold=fallbacks[label],
            min_positives_for_tuning=min_positives_for_tuning,
        )
        for index, label in enumerate(labels)
    }


def tune_binary_threshold(
    y_true: Any,
    y_prob: Any,
    *,
    default_threshold: float,
    prior_threshold: float,
    fallback_threshold: float,
    min_positives_for_tuning: int,
) -> float:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    positives = int(np.sum(true == 1))
    negatives = int(np.sum(true == 0))
    if positives < min_positives_for_tuning:
        return float(fallback_threshold)
    if positives == 0 or negatives == 0:
        return float(default_threshold)

    lower_bound = max(0.0, prior_threshold - THRESHOLD_PRIOR_SEARCH_RADIUS)
    upper_bound = min(1.0, prior_threshold + THRESHOLD_PRIOR_SEARCH_RADIUS)
    best_f1 = -1.0
    best_threshold = float(prior_threshold)
    for threshold in np.unique(prob):
        threshold = float(threshold)
        if threshold < lower_bound or threshold > upper_bound:
            continue
        pred = (prob >= threshold).astype(int)
        f1 = metrics_from_counts(confusion_counts(true, pred))["f1"]
        if f1 > best_f1 or (f1 == best_f1 and abs(threshold - prior_threshold) < abs(best_threshold - prior_threshold)):
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def threshold_priors(y_true: Any, y_prob: Any, labels: list[str], *, default_threshold: float) -> dict[str, float]:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    priors = resolve_thresholds(labels, default_threshold)
    for label, prior in LABEL_THRESHOLD_PRIORS.items():
        if label in priors:
            priors[label] = float(prior)
    for index, label in enumerate(labels):
        if label in DYNAMIC_THRESHOLD_PRIOR_LABELS:
            priors[label] = prevalence_matched_threshold(true[:, index], prob[:, index], default_threshold=priors[label])
    return priors


def prevalence_matched_threshold(y_true: Any, y_prob: Any, *, default_threshold: float) -> float:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    positives = int(np.sum(true == 1))
    negatives = int(np.sum(true == 0))
    if positives == 0 or negatives == 0:
        return float(default_threshold)
    sorted_prob = np.sort(prob)
    target_index = min(max(len(sorted_prob) - positives, 0), len(sorted_prob) - 1)
    return float(sorted_prob[target_index])


def checkpoint_score(metrics: dict[str, Any], val_loss: float) -> float:
    disease_only = metrics.get("disease_only")
    if isinstance(disease_only, Mapping) and disease_only.get("macro_average_precision") is not None:
        return float(disease_only["macro_average_precision"])
    if metrics.get("macro_average_precision") is not None:
        return float(metrics["macro_average_precision"])
    return -float(val_loss)


def resolve_thresholds(labels: list[str], threshold: float | Mapping[str, float]) -> dict[str, float]:
    if isinstance(threshold, Mapping):
        missing = [label for label in labels if label not in threshold]
        if missing:
            raise ValueError(f"Missing thresholds for labels: {missing}")
        return {label: float(threshold[label]) for label in labels}
    return {label: float(threshold) for label in labels}


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
    return float(np.mean(defined)) if defined else None


def shared_threshold(thresholds: Mapping[str, float]) -> float | None:
    values = [float(value) for value in thresholds.values()]
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def label_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower().strip())
    return slug.strip("_")


def set_seed(seed: int) -> None:
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> Any:
    require_torch()
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scheduler_name(config: TrainConfig) -> str:
    return "linear_warmup_cosine_annealing" if config.warmup_epochs > 0 else "cosine_annealing"


def cosine_t_max(config: TrainConfig) -> int:
    return max(config.epochs - config.warmup_epochs - 1, 1)


def threshold_strategy_name() -> str:
    return f"per_label_f1_from_val_bounded_by_label_priors_min_positives_{MIN_TUNED_THRESHOLD_POSITIVES}"


def format_metric(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
