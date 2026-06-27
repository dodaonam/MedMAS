from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from ._backend import require_torch, torch
from .config import (
    TrainConfig,
    artifact_paths,
    config_to_json,
    create_run_id,
    git_commit_hash,
    load_checkpoint,
    load_target_labels,
    resolve_run_dir,
    save_json,
    torchvision_version,
)
from .constants import (
    DYNAMIC_THRESHOLD_PRIOR_LABELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_THRESHOLD_PRIORS,
    MIN_TUNED_THRESHOLD_POSITIVES,
    MODEL_NAME,
    THRESHOLD_PRIOR_SEARCH_RADIUS,
)
from .data import build_dataloaders, load_manifest
from .evaluation import (
    attach_slice_metrics,
    checkpoint_score,
    compute_metrics,
    metadata_rows_from_prediction_frame,
    predict,
    prediction_frame,
    threshold_priors,
    threshold_strategy_name,
    tune_thresholds,
)
from .modeling import (
    ExponentialMovingAverage,
    build_criterion,
    build_lr_scheduler,
    build_model,
    cosine_t_max,
    resolve_device,
    scheduler_name,
    set_backbone_trainable,
    set_seed,
    train_one_epoch,
)


def train_model(config: TrainConfig) -> dict[str, Any]:
    require_torch()
    from .config import validate_train_config

    validate_train_config(config)
    labels = load_target_labels(config.target_labels_path)
    frame = load_manifest(config.manifest_path, labels)
    set_seed(config.seed)
    device = resolve_device(config.device)

    run_id = create_run_id(config.seed)
    paths = artifact_paths(config.output_dir / run_id)
    paths.run_dir.mkdir(parents=True, exist_ok=False)

    train_loader, val_loader, test_loader = build_dataloaders(config, frame, labels)
    model = build_model(
        len(labels),
        pretrained=config.pretrained,
        classifier_dropout=config.classifier_dropout,
    ).to(device)
    backbone_frozen = config.head_only or config.freeze_backbone_epochs > 0
    if backbone_frozen:
        set_backbone_trainable(model, trainable=False)
    criterion = build_criterion(config, frame, labels, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_lr_scheduler(optimizer, config)
    ema = ExponentialMovingAverage(model, decay=config.ema_decay) if config.ema_decay > 0.0 else None

    config_payload = {
        **config_to_json(config),
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "lr_scheduler": scheduler_name(config),
        "cosine_t_max": cosine_t_max(config),
        "threshold_strategy": threshold_strategy_name(),
        "threshold_tuning_min_positives": MIN_TUNED_THRESHOLD_POSITIVES,
        "target_labels": labels,
        "threshold": config.threshold,
        "head_only": config.head_only,
        "freeze_backbone_epochs": config.freeze_backbone_epochs,
        "best_checkpoint": str(paths.checkpoint_path),
        "torch_version": str(getattr(torch, "__version__", "")),
        "torchvision_version": torchvision_version(),
        "device": str(device),
        "git_commit": git_commit_hash(config.root),
        "techniques": techniques_payload(config, labels),
    }
    save_json(paths.config_path, config_payload)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        if backbone_frozen and not config.head_only and epoch == config.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            backbone_frozen = False
        start = time.time()
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            ema=ema,
            backbone_trainable=not backbone_frozen,
            desc=f"epoch {epoch}/{config.epochs} train",
        )
        eval_model = ema.model if ema is not None else model
        val_loss, val_frame, y_true, y_prob = predict(
            eval_model,
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
            low_support_threshold=threshold_priors(y_true, y_prob, labels, default_threshold=config.threshold),
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
        score = checkpoint_score(val_metrics, val_loss)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.model.state_dict() if ema is not None else None,
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
            "backbone_trainable": not backbone_frozen,
            "epoch_seconds": round(time.time() - start, 3),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(paths.history_path, index=False)
        print(
            f"epoch {epoch:03d}/{config.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_ap={fmt_metric(row['val_macro_average_precision'])}"
        )
        scheduler.step()

    checkpoint = load_checkpoint(paths.checkpoint_path, map_location=device)
    state_dict_key = "ema_state_dict" if checkpoint.get("ema_state_dict") is not None else "model_state_dict"
    model.load_state_dict(checkpoint[state_dict_key])
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
    require_torch()
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
    model = build_model(
        len(labels),
        pretrained=False,
        classifier_dropout=float(run_config.get("classifier_dropout", 0.0)),
    ).to(device)
    checkpoint = load_checkpoint(paths.checkpoint_path, map_location=device)
    state_dict_key = "ema_state_dict" if checkpoint.get("ema_state_dict") is not None else "model_state_dict"
    model.load_state_dict(checkpoint[state_dict_key])
    eval_config = replace(
        eval_config,
        loss_name=str(run_config.get("loss_name", config.loss_name)),
        asl_gamma_neg=float(run_config.get("asl_gamma_neg", config.asl_gamma_neg)),
        asl_gamma_pos=float(run_config.get("asl_gamma_pos", config.asl_gamma_pos)),
        asl_clip=float(run_config.get("asl_clip", config.asl_clip)),
    )
    criterion = build_criterion(eval_config, frame, labels, device)
    val_loss, val_frame, val_true, val_prob = predict(
        model, val_loader, criterion, device, labels, threshold=threshold, run_id=run_id, desc="best val"
    )
    tuned_thresholds = tune_thresholds(
        val_true,
        val_prob,
        labels,
        default_threshold=threshold,
        low_support_threshold=threshold_priors(val_true, val_prob, labels, default_threshold=threshold),
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
        model, test_loader, criterion, device, labels, threshold=threshold, run_id=run_id, desc="test"
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
    run_config["threshold_strategy"] = threshold_strategy_name()
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
    require_torch()
    labels = load_target_labels(config.target_labels_path)
    frame = load_manifest(config.manifest_path, labels)
    train_loader, _, _ = build_dataloaders(config, frame, labels)
    device = resolve_device(config.device)
    model = build_model(
        len(labels),
        pretrained=config.pretrained,
        classifier_dropout=config.classifier_dropout,
    ).to(device)
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


def classifier_head_type(config: TrainConfig) -> str:
    if config.classifier_dropout > 0.0:
        return "dropout_linear"
    return "linear"


def sampling_strategy_name(config: TrainConfig) -> str:
    return "weighted_random_sampler" if config.balanced_sampler else "shuffle"


def backbone_strategy_name(config: TrainConfig) -> str:
    if config.head_only:
        return "classifier_only_frozen_backbone"
    if config.freeze_backbone_epochs > 0:
        return "classifier_only_then_full_finetune"
    return "full_finetune"


def loss_payload(config: TrainConfig) -> dict[str, Any]:
    payload = {"name": config.loss_name}
    if config.loss_name == "asl":
        payload["asymmetric_loss"] = {
            "gamma_neg": config.asl_gamma_neg,
            "gamma_pos": config.asl_gamma_pos,
            "clip": config.asl_clip,
        }
    elif config.loss_name == "bce":
        payload["positive_weight_strategy"] = "train_split_negative_positive_ratio"
    return payload


def transforms_payload(config: TrainConfig) -> dict[str, Any]:
    resize_size = config.image_size + 32
    return {
        "train": {
            "resize": resize_size,
            "random_resized_crop": {"size": config.image_size, "scale": [0.9, 1.0], "ratio": [0.95, 1.05]},
            "random_rotation_degrees": 5,
            "random_affine": {"degrees": 0, "translate": [0.03, 0.03], "scale": [0.97, 1.03]},
            "color_jitter": {"brightness": 0.08, "contrast": 0.08},
            "normalize_mean": IMAGENET_MEAN,
            "normalize_std": IMAGENET_STD,
            "grayscale_to_rgb": True,
        },
        "eval": {
            "resize": resize_size,
            "center_crop": config.image_size,
            "normalize_mean": IMAGENET_MEAN,
            "normalize_std": IMAGENET_STD,
            "grayscale_to_rgb": True,
        },
    }


def techniques_payload(config: TrainConfig, labels: list[str]) -> dict[str, Any]:
    applied_techniques = [
        "torchvision_densenet121_backbone",
        "imagenet_rgb_normalization",
        "validation_per_label_threshold_selection",
        "best_checkpoint_selection_from_validation_metrics",
    ]
    if config.pretrained:
        applied_techniques.append("imagenet_pretrained_backbone")
    if config.classifier_dropout > 0.0:
        applied_techniques.append("classifier_dropout")
    if config.loss_name == "asl":
        applied_techniques.append("asymmetric_loss")
    if config.loss_name == "bce":
        applied_techniques.append("bce_with_logits_loss")
    if config.balanced_sampler:
        applied_techniques.append("weighted_random_sampler")
    if config.ema_decay > 0.0:
        applied_techniques.append("ema_evaluation_weights")
    if config.freeze_backbone_epochs > 0:
        applied_techniques.append("classifier_only_warm_start")
    if config.head_only:
        applied_techniques.append("head_only_training")
    if config.warmup_epochs > 0:
        applied_techniques.append("linear_lr_warmup")
    applied_techniques.append("cosine_lr_decay")
    return {
        "applied_techniques": applied_techniques,
        "model": {
            "architecture": MODEL_NAME,
            "backbone": {"source": "torchvision.models.densenet121", "pretrained": config.pretrained},
            "classifier_head": {
                "type": classifier_head_type(config),
                "dropout": config.classifier_dropout,
                "num_outputs": len(labels),
            },
        },
        "data": {
            "target_labels": labels,
            "sampling_strategy": sampling_strategy_name(config),
            "transforms": transforms_payload(config),
        },
        "optimization": {
            "optimizer": "AdamW",
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "scheduler": {
                "name": scheduler_name(config),
                "warmup_epochs": config.warmup_epochs,
                "warmup_start_factor": config.warmup_start_factor,
                "min_lr": config.min_lr,
                "cosine_t_max": cosine_t_max(config),
            },
            "loss": loss_payload(config),
            "ema": {"enabled": config.ema_decay > 0.0, "decay": config.ema_decay},
            "backbone_training_strategy": {
                "name": backbone_strategy_name(config),
                "head_only": config.head_only,
                "freeze_backbone_epochs": config.freeze_backbone_epochs,
            },
        },
        "evaluation": {
            "default_threshold": config.threshold,
            "threshold_strategy": threshold_strategy_name(),
            "threshold_priors": LABEL_THRESHOLD_PRIORS,
            "dynamic_threshold_prior_labels": sorted(DYNAMIC_THRESHOLD_PRIOR_LABELS),
            "threshold_prior_search_radius": THRESHOLD_PRIOR_SEARCH_RADIUS,
            "threshold_tuning_min_positives": MIN_TUNED_THRESHOLD_POSITIVES,
            "checkpoint_selection_metric": "disease_only_macro_average_precision_then_macro_average_precision_then_negative_val_loss",
            "slice_metrics": {
                "disease_only_excludes_no_finding": True,
                "scope_subsets": ["in_scope_only", "out_of_scope_only"],
            },
        },
    }


def fmt_metric(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"
