from __future__ import annotations

import json
import random
import re
import time
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
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - only happens before torch is installed
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


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


@dataclass
class TrainConfig:
    root: Path
    manifest_path: Path
    target_labels_path: Path
    output_dir: Path
    seed: int = 0
    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    lr: float = 1e-4
    weight_decay: float = 1e-4
    threshold: float = 0.5
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

    config_payload = {
        **_config_to_json(config),
        "run_id": run_id,
        "model_name": MODEL_NAME,
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
        val_metrics = compute_metrics(y_true, y_prob, labels, threshold=config.threshold, run_id=run_id)
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
            val_frame.to_csv(paths.predictions_val_path, index=False)
            save_json(paths.metrics_val_path, val_metrics)

        row = {
            "run_id": run_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_average_precision": val_metrics["macro_average_precision"],
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - start, 3),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(paths.history_path, index=False)
        print(
            f"epoch {epoch:03d}/{config.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_ap={_fmt_metric(row['val_macro_average_precision'])}"
        )

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
    val_metrics = compute_metrics(val_true, val_prob, labels, threshold=config.threshold, run_id=run_id)
    test_metrics = compute_metrics(test_true, test_prob, labels, threshold=config.threshold, run_id=run_id)
    val_metrics["loss"] = val_loss
    test_metrics["loss"] = test_loss

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
    val_metrics = compute_metrics(val_true, val_prob, labels, threshold=threshold, run_id=run_id)
    test_metrics = compute_metrics(test_true, test_prob, labels, threshold=threshold, run_id=run_id)
    val_metrics["loss"] = val_loss
    test_metrics["loss"] = test_loss
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
    return (
        DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
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
    threshold: float,
    run_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, metadata in enumerate(metadata_rows):
        row = dict(metadata)
        row["run_id"] = run_id
        for label_idx, label in enumerate(labels):
            slug = label_slug(label)
            probability = float(y_prob[idx, label_idx])
            row[label] = int(y_true[idx, label_idx])
            row[f"true_{slug}"] = int(y_true[idx, label_idx])
            row[f"prob_{slug}"] = probability
            row[f"threshold_{slug}"] = float(threshold)
            row[f"pred_{slug}"] = int(probability >= threshold)
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


def positive_weights(frame: pd.DataFrame, labels: list[str]) -> Any:
    _require_torch()
    train = frame.loc[frame["split"].astype(str) == "train", labels].to_numpy(dtype=np.float32)
    positives = train.sum(axis=0)
    negatives = train.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(
    y_true: Any,
    y_prob: Any,
    labels: list[str],
    *,
    threshold: float = 0.5,
    run_id: str | None = None,
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    if true.shape != prob.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape}, y_prob {prob.shape}")
    if true.ndim != 2 or true.shape[1] != len(labels):
        raise ValueError(f"Expected shape [n, {len(labels)}], got {true.shape}")

    per_label: dict[str, Any] = {}
    micro_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for index, label in enumerate(labels):
        pred = (prob[:, index] >= threshold).astype(int)
        counts = confusion_counts(true[:, index], pred)
        for key in micro_counts:
            micro_counts[key] += counts[key]
        basic = metrics_from_counts(counts)
        per_label[label] = {
            **counts,
            **basic,
            "positive_count": int(np.sum(true[:, index] == 1)),
            "threshold": float(threshold),
            "average_precision": average_precision(true[:, index], prob[:, index]),
            "auroc": auroc(true[:, index], prob[:, index]),
        }

    return {
        "run_id": run_id,
        "labels": labels,
        "threshold": float(threshold),
        "per_label": per_label,
        "macro_precision": mean_defined(item["precision"] for item in per_label.values()),
        "macro_recall": mean_defined(item["recall"] for item in per_label.values()),
        "macro_f1": mean_defined(item["f1"] for item in per_label.values()),
        "macro_average_precision": mean_defined(item["average_precision"] for item in per_label.values()),
        "macro_auroc": mean_defined(item["auroc"] for item in per_label.values()),
        "micro": {**micro_counts, **metrics_from_counts(micro_counts)},
    }


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
    macro_ap = metrics.get("macro_average_precision")
    if macro_ap is not None:
        return float(macro_ap)
    return -float(val_loss)


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
