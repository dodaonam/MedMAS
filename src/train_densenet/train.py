from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    TARGET_LABELS,
    RunConfig,
    build_class_weights_payload_from_manifest,
    create_run_id,
    default_training_output_dir,
    ensure_artifact_tree,
    load_target_labels,
    label_slug,
    resolve_artifact_paths,
    save_json,
    selected_pos_weights_from_payload,
    write_class_weights,
    write_run_config,
)
from .dataset import ChestXrayMultiLabelDataset, assert_patient_disjoint, create_dataloader, load_split_manifest
from .evaluate import evaluate_prediction_frame, run_inference
from .metrics import compute_multilabel_metrics
from .model import (
    DenseNet121CNNHead,
    configure_stage1,
    configure_stage2,
    set_backbone_batchnorm_eval,
    stage1_optimizer_parameters,
    stage2_optimizer_parameters,
)
from .thresholds import select_validation_thresholds, thresholds_by_label
from .transforms import build_eval_transform, build_train_transform

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only on machines without torch
    torch = None  # type: ignore[assignment]


@dataclass
class TrainingConfig:
    root: Path
    manifest_path: Path
    target_labels_path: Path
    output_dir: Path
    seed: int = 0
    batch_size: int = 16
    num_workers: int = 0
    stage1_epochs: int = 5
    stage2_epochs: int = 20
    stage2_min_epochs_before_early_stop: int = 5
    early_stopping_patience: int = 5
    head_lr_stage1: float = 3e-4
    head_lr_stage2: float = 1e-4
    backbone_lr_stage2: float = 1e-5
    weight_decay: float = 1e-4
    use_amp: bool = True
    device: str | None = None


def default_training_config(root: Path, output_dir: Path | None = None) -> TrainingConfig:
    return TrainingConfig(
        root=root,
        manifest_path=root / "artifacts" / "preprocess" / "split_manifest.csv",
        target_labels_path=root / "artifacts" / "preprocess" / "target_labels.json",
        output_dir=output_dir or default_training_output_dir(root),
    )


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to train DenseNet121. Install torch and torchvision first.")


def set_seed(seed: int) -> None:
    _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> Any:
    _require_torch()
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_criterion(device: Any, pos_weights: dict[str, float], labels: list[str]) -> Any:
    _require_torch()
    weights = torch.tensor([pos_weights[label] for label in labels], dtype=torch.float32, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=weights)


def build_dataloaders(config: TrainingConfig, labels: list[str]) -> dict[str, Any]:
    manifest = load_split_manifest(config.manifest_path, labels)
    assert_patient_disjoint(manifest)
    datasets = {
        "train": ChestXrayMultiLabelDataset(
            manifest,
            root=config.root,
            split="train",
            target_labels=labels,
            transform=build_train_transform(),
        ),
        "val": ChestXrayMultiLabelDataset(
            manifest,
            root=config.root,
            split="val",
            target_labels=labels,
            transform=build_eval_transform(),
        ),
        "test": ChestXrayMultiLabelDataset(
            manifest,
            root=config.root,
            split="test",
            target_labels=labels,
            transform=build_eval_transform(),
        ),
    }
    pin_memory = bool(torch is not None and torch.cuda.is_available())
    return {
        "train": create_dataloader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        "val": create_dataloader(
            datasets["val"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        "test": create_dataloader(
            datasets["test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
    }


def run_stage0_smoke(model: Any, dataloader: Any, criterion: Any, device: Any, labels: list[str]) -> None:
    _require_torch()
    model.train()
    images, targets, _metadata = next(iter(dataloader))
    images = images.to(device)
    targets = targets.to(device)
    assert images.ndim == 4
    assert tuple(images.shape[1:]) == (3, 224, 224)
    assert images.dtype == torch.float32
    assert targets.ndim == 2
    assert targets.shape[1] == len(labels)
    assert targets.dtype == torch.float32
    logits = model(images)
    assert logits.shape == targets.shape
    assert logits.dtype == torch.float32
    loss = criterion(logits, targets)
    loss.backward()
    model.zero_grad(set_to_none=True)


def train_one_epoch(
    *,
    model: Any,
    dataloader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    use_amp: bool,
) -> float:
    _require_torch()
    model.train()
    set_backbone_batchnorm_eval(model)
    total_loss = 0.0
    total_rows = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    for images, targets, _metadata in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(images.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_rows += batch_size
    return total_loss / max(total_rows, 1)


def evaluate_loss(
    *,
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
    labels: list[str],
) -> tuple[float, dict[str, Any]]:
    _require_torch()
    model.eval()
    total_loss = 0.0
    total_rows = 0
    true_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []
    with torch.no_grad():
        for images, targets, _metadata in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            probabilities = torch.sigmoid(logits)
            batch_size = int(images.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_rows += batch_size
            true_batches.append(targets.detach().cpu().numpy())
            prob_batches.append(probabilities.detach().cpu().numpy())
    y_true = np.concatenate(true_batches, axis=0)
    y_prob = np.concatenate(prob_batches, axis=0)
    default_thresholds = {label: 0.5 for label in labels}
    metrics = compute_multilabel_metrics(y_true, y_prob, default_thresholds, labels)
    return total_loss / max(total_rows, 1), metrics


def _current_lrs(optimizer: Any) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "learning_rate_cnn_head": None,
        "learning_rate_denseblock4_norm5": None,
    }
    for group in optimizer.param_groups:
        name = group.get("name")
        if name == "cnn_head":
            result["learning_rate_cnn_head"] = float(group["lr"])
        elif name == "denseblock4_norm5":
            result["learning_rate_denseblock4_norm5"] = float(group["lr"])
    return result


def train_model(config: TrainingConfig) -> dict[str, Any]:
    _require_torch()
    labels = load_target_labels(config.target_labels_path)
    set_seed(config.seed)
    device = resolve_device(config.device)
    paths = resolve_artifact_paths(config.output_dir)
    ensure_artifact_tree(paths)
    run_id = create_run_id(config.seed)
    manifest = load_split_manifest(config.manifest_path, labels)
    class_weights_payload = build_class_weights_payload_from_manifest(manifest, labels)
    write_class_weights(paths.class_weights_path, class_weights_payload)
    run_config = RunConfig(
        run_id=run_id,
        output_dir=str(paths.output_dir),
        seed=config.seed,
        split_manifest_path=str(config.manifest_path),
        target_labels_path=str(config.target_labels_path),
        target_label_order=labels,
        stage1_epochs_planned=[config.stage1_epochs, config.stage1_epochs],
        stage2_min_epochs_before_early_stop=config.stage2_min_epochs_before_early_stop,
    )
    write_run_config(paths.config_path, run_config)

    dataloaders = build_dataloaders(config, labels)
    model = DenseNet121CNNHead(num_classes=len(labels)).to(device)
    selected_pos_weights = selected_pos_weights_from_payload(class_weights_payload, labels)
    criterion = create_criterion(device, selected_pos_weights, labels)
    configure_stage1(model)
    run_stage0_smoke(model, dataloaders["train"], criterion, device, labels)

    history: list[dict[str, Any]] = []
    best_metric = -np.inf
    best_tiebreaker_auroc = -np.inf
    best_tiebreaker_loss = np.inf
    best_epoch = 0
    global_epoch = 0

    optimizer = torch.optim.AdamW(
        stage1_optimizer_parameters(model, config.head_lr_stage1, config.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    for _ in range(config.stage1_epochs):
        global_epoch += 1
        configure_stage1(model)
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=config.use_amp,
        )
        val_loss, val_metrics = evaluate_loss(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            labels=labels,
        )
        scheduler.step(val_metrics["disease_macro_average_precision"] or 0.0)
        lrs = _current_lrs(optimizer)
        history.append(
            {
                "run_id": run_id,
                "epoch": global_epoch,
                "stage": "cnn_head_only",
                "train_loss": train_loss,
                "val_loss": val_loss,
                **lrs,
                "learning_rate_denseblock4_norm5": np.nan,
                "val_disease_macro_auroc": val_metrics["disease_macro_auroc"],
                "val_disease_macro_average_precision": val_metrics["disease_macro_average_precision"],
                "val_all_label_macro_auroc": val_metrics["all_label_macro_auroc"],
                "val_all_label_macro_average_precision": val_metrics["all_label_macro_average_precision"],
            }
        )
        metric = val_metrics["disease_macro_average_precision"] or -np.inf
        auroc = val_metrics["disease_macro_auroc"] or -np.inf
        if metric > best_metric or (metric == best_metric and (auroc > best_tiebreaker_auroc or val_loss < best_tiebreaker_loss)):
            best_metric = metric
            best_tiebreaker_auroc = auroc
            best_tiebreaker_loss = val_loss
            best_epoch = global_epoch
            torch.save(model.state_dict(), paths.checkpoint_best_path)

    stage2_start_epoch = global_epoch + 1
    run_config.stage2_start_epoch = stage2_start_epoch
    write_run_config(paths.config_path, run_config)
    configure_stage2(model)
    optimizer = torch.optim.AdamW(
        stage2_optimizer_parameters(
            model,
            head_lr=config.head_lr_stage2,
            backbone_lr=config.backbone_lr_stage2,
            weight_decay=config.weight_decay,
        )
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    patience_counter = 0
    for stage2_epoch in range(1, config.stage2_epochs + 1):
        global_epoch += 1
        configure_stage2(model)
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=config.use_amp,
        )
        val_loss, val_metrics = evaluate_loss(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            labels=labels,
        )
        scheduler.step(val_metrics["disease_macro_average_precision"] or 0.0)
        lrs = _current_lrs(optimizer)
        history.append(
            {
                "run_id": run_id,
                "epoch": global_epoch,
                "stage": "denseblock4_norm5_finetune",
                "train_loss": train_loss,
                "val_loss": val_loss,
                **lrs,
                "val_disease_macro_auroc": val_metrics["disease_macro_auroc"],
                "val_disease_macro_average_precision": val_metrics["disease_macro_average_precision"],
                "val_all_label_macro_auroc": val_metrics["all_label_macro_auroc"],
                "val_all_label_macro_average_precision": val_metrics["all_label_macro_average_precision"],
            }
        )
        metric = val_metrics["disease_macro_average_precision"] or -np.inf
        auroc = val_metrics["disease_macro_auroc"] or -np.inf
        improved = metric > best_metric or (
            metric == best_metric and (auroc > best_tiebreaker_auroc or val_loss < best_tiebreaker_loss)
        )
        if improved:
            best_metric = metric
            best_tiebreaker_auroc = auroc
            best_tiebreaker_loss = val_loss
            best_epoch = global_epoch
            patience_counter = 0
            torch.save(model.state_dict(), paths.checkpoint_best_path)
        else:
            patience_counter += 1
        if stage2_epoch >= config.stage2_min_epochs_before_early_stop and patience_counter >= config.early_stopping_patience:
            break

    pd.DataFrame(history).to_csv(paths.training_history_path, index=False)
    model.load_state_dict(torch.load(paths.checkpoint_best_path, map_location=device))
    val_frame_05 = run_inference(
        model,
        dataloaders["val"],
        device=device,
        thresholds={label: 0.5 for label in labels},
        labels=labels,
        run_id=run_id,
    )
    y_true = val_frame_05[[f"true_{label_slug(label)}" for label in labels]].to_numpy(dtype=int)
    y_prob = val_frame_05[[f"prob_{label_slug(label)}" for label in labels]].to_numpy(dtype=float)
    thresholds_payload = select_validation_thresholds(y_true, y_prob, labels=labels, run_id=run_id)
    save_json(paths.thresholds_path, thresholds_payload)
    selected_thresholds = thresholds_by_label(thresholds_payload, labels)
    val_frame = run_inference(model, dataloaders["val"], device=device, thresholds=selected_thresholds, labels=labels, run_id=run_id)
    test_frame = run_inference(model, dataloaders["test"], device=device, thresholds=selected_thresholds, labels=labels, run_id=run_id)
    val_frame.to_csv(paths.predictions_val_path, index=False)
    test_frame.to_csv(paths.predictions_test_path, index=False)
    save_json(paths.metrics_val_path, evaluate_prediction_frame(val_frame, labels=labels, run_id=run_id))
    save_json(paths.metrics_test_path, evaluate_prediction_frame(test_frame, labels=labels, run_id=run_id))
    return {
        "run_id": run_id,
        "output_dir": str(paths.output_dir),
        "best_epoch": best_epoch,
        "best_validation_disease_macro_average_precision": best_metric,
    }
