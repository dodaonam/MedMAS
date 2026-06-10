from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    DISEASE_LABELS,
    RunConfig,
    create_run_id,
    default_training_output_dir,
    ensure_artifact_tree,
    labels_for_target_mode,
    load_target_labels,
    label_slug,
    resolve_artifact_paths,
    resolve_run_output_dir,
    save_json,
    write_run_config,
)
from .dataset import ChestXrayMultiLabelDataset, assert_patient_disjoint, create_dataloader, load_split_manifest
from .evaluate import evaluate_prediction_frame, run_inference
from .losses import AsymmetricLossConfig, build_asymmetric_loss
from .metrics import checkpoint_candidate_improved, compute_multilabel_metrics
from .model import (
    DenseNet121CNNHead,
    configure_stage1,
    configure_stage2,
    set_backbone_batchnorm_eval,
    stage1_optimizer_parameters,
    stage2_optimizer_parameters,
)
from .progress import ProgressBar
from .schedulers import (
    accumulation_window_example_count,
    build_warmup_cosine_scheduler,
    optimizer_steps_per_epoch,
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
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    input_size: int = 320
    resize_size: int = 352
    target_mode: str = "disease_only"
    stage1_epochs: int = 5
    stage2_epochs: int = 30
    stage2_min_epochs_before_early_stop: int = 5
    early_stopping_patience: int = 5
    min_delta: float = 0.001
    head_lr_stage1: float = 3e-4
    head_lr_stage2: float = 1e-4
    backbone_lr_stage2: float = 1e-5
    weight_decay: float = 1e-4
    scheduler: str = "warmup_cosine"
    warmup_ratio: float = 0.10
    min_lr_factor: float = 0.01
    loss: str = "asymmetric"
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
    asl_eps: float = 1e-8
    use_amp: bool = True
    device: str | None = None

    @property
    def effective_batch_size(self) -> int:
        return int(self.batch_size * self.gradient_accumulation_steps)


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


def runtime_environment_config(device: Any, *, amp_enabled: bool) -> dict[str, Any]:
    _require_torch()
    try:
        import torchvision
    except ModuleNotFoundError:
        torchvision_version = None
    else:
        torchvision_version = getattr(torchvision, "__version__", None)

    device_name = str(device)
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(device)
        except Exception:
            device_name = str(device)

    return {
        "amp_enabled": bool(amp_enabled),
        "torch_version": getattr(torch, "__version__", None),
        "torchvision_version": torchvision_version,
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_name": device_name,
    }


def create_criterion(config: TrainingConfig) -> Any:
    _require_torch()
    if config.loss != "asymmetric":
        raise ValueError("Recipe v2 supports only Asymmetric Loss via --loss asymmetric.")
    return build_asymmetric_loss(
        AsymmetricLossConfig(
            gamma_pos=config.asl_gamma_pos,
            gamma_neg=config.asl_gamma_neg,
            clip=config.asl_clip,
            eps=config.asl_eps,
            reduction="mean",
        )
    )


def build_dataloaders(config: TrainingConfig, labels: list[str]) -> dict[str, Any]:
    manifest = load_split_manifest(config.manifest_path, labels)
    assert_patient_disjoint(manifest)
    datasets = {
        "train": ChestXrayMultiLabelDataset(
            manifest,
            root=config.root,
            split="train",
            target_labels=labels,
            transform=build_train_transform(input_size=config.input_size, resize_size=config.resize_size),
        ),
        "val": ChestXrayMultiLabelDataset(
            manifest,
            root=config.root,
            split="val",
            target_labels=labels,
            transform=build_eval_transform(input_size=config.input_size, resize_size=config.resize_size),
        ),
        "test": ChestXrayMultiLabelDataset(
            manifest,
            root=config.root,
            split="test",
            target_labels=labels,
            transform=build_eval_transform(input_size=config.input_size, resize_size=config.resize_size),
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


def run_stage0_smoke(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
    labels: list[str],
    *,
    input_size: int = 320,
) -> None:
    _require_torch()
    model.train()
    set_backbone_batchnorm_eval(model)
    images, targets, _metadata = next(iter(dataloader))
    images = images.to(device)
    targets = targets.to(device)
    assert images.ndim == 4
    assert tuple(images.shape[1:]) == (3, input_size, input_size)
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
    scheduler: Any,
    device: Any,
    use_amp: bool,
    gradient_accumulation_steps: int,
    progress_desc: str,
) -> float:
    _require_torch()
    model.train()
    set_backbone_batchnorm_eval(model)
    total_loss = 0.0
    total_rows = 0
    amp_enabled = use_amp and device.type == "cuda"
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device_type, enabled=amp_enabled)
    configured_batch_size = int(getattr(dataloader, "batch_size", 0) or 0)
    total_examples = int(len(dataloader.dataset)) if hasattr(dataloader, "dataset") else 0
    optimizer.zero_grad(set_to_none=True)
    with ProgressBar(total=len(dataloader), desc=progress_desc) as progress:
        for batch_idx, (images, targets, _metadata) in enumerate(dataloader, start=1):
            images = images.to(device)
            targets = targets.to(device)
            batch_size = int(images.shape[0])
            effective_batch_size = configured_batch_size or batch_size
            effective_total_examples = total_examples or len(dataloader) * effective_batch_size
            with torch.amp.autocast(amp_device_type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
                window_examples = accumulation_window_example_count(
                    batch_index=batch_idx,
                    total_batches=len(dataloader),
                    total_examples=effective_total_examples,
                    batch_size=effective_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
                scaled_loss = loss * (batch_size / window_examples)
            scaler.scale(scaled_loss).backward()
            should_step = batch_idx % gradient_accumulation_steps == 0 or batch_idx == len(dataloader)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_rows += batch_size
            progress.update(loss=total_loss / max(total_rows, 1))
    return total_loss / max(total_rows, 1)


def evaluate_loss(
    *,
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
    labels: list[str],
    progress_desc: str,
) -> tuple[float, dict[str, Any]]:
    _require_torch()
    model.eval()
    total_loss = 0.0
    total_rows = 0
    true_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []
    with torch.no_grad():
        with ProgressBar(total=len(dataloader), desc=progress_desc) as progress:
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
                progress.update(loss=total_loss / max(total_rows, 1))
    y_true = np.concatenate(true_batches, axis=0)
    y_prob = np.concatenate(prob_batches, axis=0)
    default_thresholds = {label: 0.5 for label in labels}
    metrics = compute_multilabel_metrics(y_true, y_prob, default_thresholds, labels)
    return total_loss / max(total_rows, 1), metrics


def _require_config_value(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"Recipe v2 requires {name}={expected!r}, got {actual!r}.")


def _require_config_float(actual: float, expected: float, name: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=1e-12, atol=0.0):
        raise ValueError(f"Recipe v2 requires {name}={expected!r}, got {actual!r}.")


def validate_recipe_config(config: TrainingConfig, labels: list[str]) -> None:
    if labels != DISEASE_LABELS:
        raise ValueError(f"Recipe v2 expects disease labels {DISEASE_LABELS!r}, got {labels!r}")
    _require_config_value(config.target_mode, "disease_only", "target_mode")
    _require_config_value(config.input_size, 320, "input_size")
    _require_config_value(config.resize_size, 352, "resize_size")
    _require_config_value(config.stage1_epochs, 5, "stage1_epochs")
    _require_config_value(config.stage2_epochs, 30, "stage2_epochs")
    _require_config_value(config.stage2_min_epochs_before_early_stop, 5, "stage2_min_epochs_before_early_stop")
    _require_config_value(config.early_stopping_patience, 5, "early_stopping_patience")
    _require_config_float(config.min_delta, 0.001, "min_delta")
    _require_config_float(config.head_lr_stage1, 3e-4, "head_lr_stage1")
    _require_config_float(config.head_lr_stage2, 1e-4, "head_lr_stage2")
    _require_config_float(config.backbone_lr_stage2, 1e-5, "backbone_lr_stage2")
    _require_config_float(config.weight_decay, 1e-4, "weight_decay")
    _require_config_value(config.scheduler, "warmup_cosine", "scheduler")
    _require_config_float(config.warmup_ratio, 0.10, "warmup_ratio")
    _require_config_float(config.min_lr_factor, 0.01, "min_lr_factor")
    _require_config_value(config.loss, "asymmetric", "loss")
    _require_config_float(config.asl_gamma_pos, 0.0, "asl_gamma_pos")
    _require_config_float(config.asl_gamma_neg, 4.0, "asl_gamma_neg")
    _require_config_float(config.asl_clip, 0.05, "asl_clip")
    _require_config_float(config.asl_eps, 1e-8, "asl_eps")
    if (config.batch_size, config.gradient_accumulation_steps) not in {(256, 1), (128, 2)}:
        raise ValueError(
            "Recipe v2 supports only physical batch 256 with accumulation 1, "
            "or OOM fallback physical batch 128 with accumulation 2."
        )
    if config.effective_batch_size != 256:
        raise ValueError(
            f"Recipe v2 requires effective batch size 256, got {config.effective_batch_size} "
            f"from batch_size={config.batch_size} and gradient_accumulation_steps={config.gradient_accumulation_steps}."
        )


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


def _format_lr(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2e}"


def _print_epoch_summary(
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: dict[str, Any],
    lrs: dict[str, float | None],
    improved: bool,
    patience_counter: int | None = None,
) -> None:
    patience_suffix = "" if patience_counter is None else f" patience={patience_counter}"
    print(
        "Epoch "
        f"{epoch} summary: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"val_disease_macro_AP={val_metrics['disease_macro_average_precision'] or 0.0:.4f} "
        f"val_disease_macro_AUROC={val_metrics['disease_macro_auroc'] or 0.0:.4f} "
        f"head_lr={_format_lr(lrs.get('learning_rate_cnn_head'))} "
        f"backbone_lr={_format_lr(lrs.get('learning_rate_denseblock4_norm5'))} "
        f"checkpoint={'best' if improved else 'no'}"
        f"{patience_suffix}",
        flush=True,
    )


def train_model(config: TrainingConfig) -> dict[str, Any]:
    _require_torch()
    source_labels = load_target_labels(config.target_labels_path)
    labels = labels_for_target_mode(config.target_mode)
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    validate_recipe_config(config, labels)
    set_seed(config.seed)
    device = resolve_device(config.device)
    amp_enabled = config.use_amp and device.type == "cuda"
    run_id = create_run_id(config.seed)
    run_output_dir = resolve_run_output_dir(config.output_dir, run_id)
    paths = resolve_artifact_paths(run_output_dir)
    ensure_artifact_tree(paths)

    loss_config = AsymmetricLossConfig(
        gamma_pos=config.asl_gamma_pos,
        gamma_neg=config.asl_gamma_neg,
        clip=config.asl_clip,
        eps=config.asl_eps,
        reduction="mean",
    ).to_dict()
    run_config = RunConfig(
        run_id=run_id,
        output_dir=str(paths.output_dir),
        seed=config.seed,
        split_manifest_path=str(config.manifest_path),
        target_labels_path=str(config.target_labels_path),
        target_mode=config.target_mode,
        source_target_label_order=source_labels,
        target_labels=labels,
        target_label_order=labels,
        checkpoint_metric="validation_disease_macro_average_precision",
        checkpoint_selection_metric="validation_disease_macro_average_precision",
        stage1_epochs_planned=config.stage1_epochs,
        stage2_min_epochs_before_early_stop=config.stage2_min_epochs_before_early_stop,
        stage1_config={
            "stage_name": "cnn_head_only",
            "epochs": config.stage1_epochs,
            "head_max_lr": config.head_lr_stage1,
            "trainable": ["cnn_head", "classifier"],
            "frozen": ["backbone"],
            "backbone_batchnorm": "eval",
            "cnn_head_batchnorm": "train",
        },
        stage2_config={
            "stage_name": "denseblock4_norm5_finetune",
            "epochs": config.stage2_epochs,
            "head_max_lr": config.head_lr_stage2,
            "backbone_max_lr": config.backbone_lr_stage2,
            "trainable": ["cnn_head", "classifier", "backbone.denseblock4", "backbone.norm5"],
            "frozen": [
                "backbone.conv0",
                "backbone.norm0",
                "backbone.denseblock1",
                "backbone.transition1",
                "backbone.denseblock2",
                "backbone.transition2",
                "backbone.denseblock3",
                "backbone.transition3",
            ],
            "backbone_batchnorm": "eval",
            "cnn_head_batchnorm": "train",
        },
        input_size=config.input_size,
        resize_size=config.resize_size,
        physical_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        effective_batch_size=config.effective_batch_size,
        **runtime_environment_config(device, amp_enabled=amp_enabled),
        loss_config=loss_config,
        optimizer_config={
            "optimizer": "AdamW",
            "weight_decay": config.weight_decay,
            "stage1_head_lr": config.head_lr_stage1,
            "stage2_head_lr": config.head_lr_stage2,
            "stage2_backbone_lr": config.backbone_lr_stage2,
        },
        scheduler_config={
            "scheduler": "warmup_cosine",
            "warmup_ratio": config.warmup_ratio,
            "min_lr_factor": config.min_lr_factor,
            "step_unit": "optimizer_step",
        },
        early_stopping_config={
            "patience": config.early_stopping_patience,
            "min_delta": config.min_delta,
            "stage2_min_epochs_before_early_stop": config.stage2_min_epochs_before_early_stop,
        },
    )
    write_run_config(paths.config_path, run_config)
    save_json(paths.loss_config_path, loss_config)

    dataloaders = build_dataloaders(config, labels)
    model = DenseNet121CNNHead(num_classes=len(labels)).to(device)
    criterion = create_criterion(config)
    configure_stage1(model)
    run_stage0_smoke(model, dataloaders["train"], criterion, device, labels, input_size=config.input_size)

    history: list[dict[str, Any]] = []
    best_metric = -np.inf
    best_auroc: float | None = None
    best_val_loss: float | None = None
    best_epoch = 0
    global_epoch = 0

    train_optimizer_steps_per_epoch = optimizer_steps_per_epoch(
        len(dataloaders["train"]),
        config.gradient_accumulation_steps,
    )
    optimizer = torch.optim.AdamW(
        stage1_optimizer_parameters(model, config.head_lr_stage1, config.weight_decay)
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_optimizer_steps=max(config.stage1_epochs * train_optimizer_steps_per_epoch, 1),
        warmup_ratio=config.warmup_ratio,
        min_lr_factor=config.min_lr_factor,
    )
    for _ in range(config.stage1_epochs):
        global_epoch += 1
        print(f"Epoch {global_epoch} | Stage 1/{config.stage1_epochs} cnn_head_only", flush=True)
        epoch_started_at = time.perf_counter()
        configure_stage1(model)
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=config.use_amp,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            progress_desc=f"train e{global_epoch}",
        )
        val_loss, val_metrics = evaluate_loss(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            labels=labels,
            progress_desc=f"val   e{global_epoch}",
        )
        lrs = _current_lrs(optimizer)
        metric = val_metrics["disease_macro_average_precision"]
        improved = checkpoint_candidate_improved(
            candidate_ap=metric,
            best_ap=best_metric,
            min_delta=config.min_delta,
            has_best=best_epoch != 0,
            candidate_auroc=val_metrics["disease_macro_auroc"],
            best_auroc=best_auroc,
            candidate_val_loss=val_loss,
            best_val_loss=best_val_loss,
        )
        if improved:
            best_metric = metric if metric is not None else -np.inf
            best_auroc = val_metrics["disease_macro_auroc"]
            best_val_loss = val_loss
            best_epoch = global_epoch
            torch.save(model.state_dict(), paths.checkpoint_best_path)
        row = {
            "run_id": run_id,
            "epoch": global_epoch,
            "stage": "cnn_head_only",
            "train_loss": train_loss,
            "val_loss": val_loss,
            **lrs,
            "learning_rate_denseblock4_norm5": np.nan,
            "val_disease_macro_auroc": val_metrics["disease_macro_auroc"],
            "val_disease_macro_average_precision": val_metrics["disease_macro_average_precision"],
            "is_best_checkpoint": improved,
            "early_stopping_counter": np.nan,
            "epoch_duration": time.perf_counter() - epoch_started_at,
        }
        for label in labels:
            slug = label_slug(label)
            row[f"val_{slug}_average_precision"] = val_metrics["per_label"][label]["average_precision"]
            row[f"val_{slug}_auroc"] = val_metrics["per_label"][label]["auroc"]
        history.append(row)
        _print_epoch_summary(
            epoch=global_epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            lrs=lrs,
            improved=improved,
        )

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
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_optimizer_steps=max(config.stage2_epochs * train_optimizer_steps_per_epoch, 1),
        warmup_ratio=config.warmup_ratio,
        min_lr_factor=config.min_lr_factor,
    )
    patience_counter = 0
    for stage2_epoch in range(1, config.stage2_epochs + 1):
        global_epoch += 1
        print(f"Epoch {global_epoch} | Stage 2/{config.stage2_epochs} denseblock4_norm5_finetune", flush=True)
        epoch_started_at = time.perf_counter()
        configure_stage2(model)
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=config.use_amp,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            progress_desc=f"train e{global_epoch}",
        )
        val_loss, val_metrics = evaluate_loss(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            labels=labels,
            progress_desc=f"val   e{global_epoch}",
        )
        lrs = _current_lrs(optimizer)
        metric = val_metrics["disease_macro_average_precision"]
        improved = checkpoint_candidate_improved(
            candidate_ap=metric,
            best_ap=best_metric,
            min_delta=config.min_delta,
            has_best=best_epoch != 0,
            candidate_auroc=val_metrics["disease_macro_auroc"],
            best_auroc=best_auroc,
            candidate_val_loss=val_loss,
            best_val_loss=best_val_loss,
        )
        if improved:
            best_metric = metric if metric is not None else -np.inf
            best_auroc = val_metrics["disease_macro_auroc"]
            best_val_loss = val_loss
            best_epoch = global_epoch
            patience_counter = 0
            torch.save(model.state_dict(), paths.checkpoint_best_path)
        else:
            patience_counter += 1
        row = {
            "run_id": run_id,
            "epoch": global_epoch,
            "stage": "denseblock4_norm5_finetune",
            "train_loss": train_loss,
            "val_loss": val_loss,
            **lrs,
            "val_disease_macro_auroc": val_metrics["disease_macro_auroc"],
            "val_disease_macro_average_precision": val_metrics["disease_macro_average_precision"],
            "is_best_checkpoint": improved,
            "early_stopping_counter": patience_counter,
            "epoch_duration": time.perf_counter() - epoch_started_at,
        }
        for label in labels:
            slug = label_slug(label)
            row[f"val_{slug}_average_precision"] = val_metrics["per_label"][label]["average_precision"]
            row[f"val_{slug}_auroc"] = val_metrics["per_label"][label]["auroc"]
        history.append(row)
        _print_epoch_summary(
            epoch=global_epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            lrs=lrs,
            improved=improved,
            patience_counter=patience_counter,
        )
        if stage2_epoch >= config.stage2_min_epochs_before_early_stop and patience_counter >= config.early_stopping_patience:
            print(
                "Early stopping: "
                f"stage2_epoch={stage2_epoch} patience={patience_counter}/{config.early_stopping_patience}",
                flush=True,
            )
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
        progress_desc="infer val thresholds",
    )
    y_true = val_frame_05[[f"true_{label_slug(label)}" for label in labels]].to_numpy(dtype=int)
    y_prob = val_frame_05[[f"prob_{label_slug(label)}" for label in labels]].to_numpy(dtype=float)
    thresholds_payload = select_validation_thresholds(y_true, y_prob, labels=labels, run_id=run_id)
    save_json(paths.thresholds_path, thresholds_payload)
    selected_thresholds = thresholds_by_label(thresholds_payload, labels)
    val_frame = run_inference(
        model,
        dataloaders["val"],
        device=device,
        thresholds=selected_thresholds,
        labels=labels,
        run_id=run_id,
        progress_desc="infer val final",
    )
    test_frame = run_inference(
        model,
        dataloaders["test"],
        device=device,
        thresholds=selected_thresholds,
        labels=labels,
        run_id=run_id,
        progress_desc="infer test final",
    )
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
