from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


MODEL_NAME = "densenet121_cnn_head"

TARGET_LABELS: list[str] = [
    "No Finding",
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
]

DISEASE_LABELS: list[str] = TARGET_LABELS[1:]
SUPPORTED_TARGET_MODES = {"disease_only"}


def label_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    return slug.strip("_")


def default_training_output_dir(root: Path) -> Path:
    return root / "artifacts" / "training" / MODEL_NAME


def create_run_id(seed: int, timestamp: datetime | None = None) -> str:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{MODEL_NAME}_disease5_320_asl_seed{seed}_{stamp}"


def resolve_run_output_dir(output_base: Path, run_id: str) -> Path:
    if output_base.name == run_id:
        return output_base
    return output_base / run_id


def load_target_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        labels = payload
    else:
        labels = payload.get("target_labels")
    if labels != TARGET_LABELS:
        raise ValueError(f"Unexpected target label order in {path}: {labels!r}")
    return list(labels)


def labels_for_target_mode(target_mode: str) -> list[str]:
    if target_mode != "disease_only":
        raise ValueError(f"Unsupported target_mode {target_mode!r}. Recipe v2 supports only 'disease_only'.")
    return list(DISEASE_LABELS)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class TrainingArtifactPaths:
    output_dir: Path
    config_path: Path
    loss_config_path: Path
    thresholds_path: Path
    metrics_val_path: Path
    metrics_test_path: Path
    predictions_val_path: Path
    predictions_test_path: Path
    checkpoint_best_path: Path
    training_history_path: Path
    figures_dir: Path


def resolve_artifact_paths(output_dir: Path) -> TrainingArtifactPaths:
    return TrainingArtifactPaths(
        output_dir=output_dir,
        config_path=output_dir / "config.json",
        loss_config_path=output_dir / "loss_config.json",
        thresholds_path=output_dir / "thresholds.json",
        metrics_val_path=output_dir / "metrics_val.json",
        metrics_test_path=output_dir / "metrics_test.json",
        predictions_val_path=output_dir / "predictions_val.csv",
        predictions_test_path=output_dir / "predictions_test.csv",
        checkpoint_best_path=output_dir / "checkpoint_best.pt",
        training_history_path=output_dir / "training_history.csv",
        figures_dir=output_dir / "figures",
    )


def ensure_artifact_tree(paths: TrainingArtifactPaths) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in [
        "00_data",
        "01_training",
        "02_validation",
        "03_test",
        "04_subgroups",
        "05_errors",
        "06_gradcam",
    ]:
        (paths.figures_dir / subdir).mkdir(parents=True, exist_ok=True)


@dataclass
class RunConfig:
    run_id: str
    model_name: str = MODEL_NAME
    output_dir: str = ""
    seed: int = 0
    split_manifest_path: str = "artifacts/preprocess/split_manifest.csv"
    target_labels_path: str = "artifacts/preprocess/target_labels.json"
    target_mode: str = "disease_only"
    source_target_label_order: list[str] = field(default_factory=lambda: list(TARGET_LABELS))
    target_labels: list[str] = field(default_factory=lambda: list(DISEASE_LABELS))
    target_label_order: list[str] = field(default_factory=lambda: list(DISEASE_LABELS))
    no_finding_derivation_policy: dict[str, str] = field(
        default_factory=lambda: {
            "true_no_finding_source": "Manifest No Finding column retained as source metadata.",
            "true_no_finding_derived": "1 when all five disease true labels are 0.",
            "pred_no_finding_derived": "1 when all five disease predictions are 0 after validation-selected thresholds.",
        }
    )
    checkpoint_metric: str = "validation_disease_macro_average_precision"
    checkpoint_selection_metric: str = "validation_disease_macro_average_precision"
    stage1_name: str = "cnn_head_only"
    stage2_name: str = "denseblock4_norm5_finetune"
    stage2_start_epoch: int | None = None
    stage1_epochs_planned: int = 5
    stage2_min_epochs_before_early_stop: int = 5
    stage1_config: dict[str, Any] = field(
        default_factory=lambda: {
            "stage_name": "cnn_head_only",
            "epochs": 5,
            "head_max_lr": 3e-4,
            "trainable": ["cnn_head", "classifier"],
            "frozen": ["backbone"],
        }
    )
    stage2_config: dict[str, Any] = field(
        default_factory=lambda: {
            "stage_name": "denseblock4_norm5_finetune",
            "epochs": 30,
            "head_max_lr": 1e-4,
            "backbone_max_lr": 1e-5,
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
        }
    )
    dense_net_variant: str = "torchvision.models.densenet121"
    input_size: int = 320
    resize_size: int = 352
    physical_batch_size: int = 256
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 256
    amp_enabled: bool = True
    torch_version: str | None = None
    torchvision_version: str | None = None
    cuda_version: str | None = None
    device_name: str | None = None
    loss_config: dict[str, Any] = field(
        default_factory=lambda: {
            "loss": "asymmetric",
            "gamma_pos": 0.0,
            "gamma_neg": 4.0,
            "clip": 0.05,
            "eps": 1e-8,
            "reduction": "mean",
            "reduction_scope": "mean over batch and labels",
        }
    )
    optimizer_config: dict[str, Any] = field(default_factory=lambda: {"optimizer": "AdamW", "weight_decay": 1e-4})
    scheduler_config: dict[str, Any] = field(
        default_factory=lambda: {
            "scheduler": "warmup_cosine",
            "warmup_ratio": 0.10,
            "min_lr_factor": 0.01,
            "step_unit": "optimizer_step",
        }
    )
    early_stopping_config: dict[str, Any] = field(
        default_factory=lambda: {"patience": 5, "min_delta": 0.001, "stage2_min_epochs_before_early_stop": 5}
    )
    test_leakage_policy: str = (
        "Validation selects checkpoint and thresholds; test is evaluated only after architecture, checkpoint, "
        "thresholds, and reporting code are fixed."
    )
    image_normalization: dict[str, list[float]] = field(
        default_factory=lambda: {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    )
    cnn_head_config: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "DenseNet121CNNHead",
            "backbone": "torchvision.models.densenet121",
            "pretrained_weights": "DenseNet121_Weights.IMAGENET1K_V1",
            "output_classes": 5,
            "architecture": [
                "DenseNet121.features",
                "ReLU",
                "Conv2d(1024, 256, kernel_size=1, bias=False)",
                "BatchNorm2d(256)",
                "ReLU",
                "Dropout2d(p=0.10)",
                "Conv2d(256, 128, kernel_size=3, padding=1, bias=False)",
                "BatchNorm2d(128)",
                "ReLU",
                "Conv2d(128, 128, kernel_size=3, padding=1, bias=False)",
                "BatchNorm2d(128)",
                "ReLU",
                "AdaptiveAvgPool2d((1, 1))",
                "Dropout(p=0.45)",
                "Linear(128, 5)",
            ],
        }
    )
    cnn_head: str = "1024->256->128->128->GAP->Dropout->Linear(5)"
    batchnorm_policy: str = "backbone BN running statistics frozen/eval in both training stages"
    backbone_batchnorm_policy: str = "backbone BN running statistics frozen in both training stages"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_run_config(path: Path, config: RunConfig) -> None:
    save_json(path, config.to_dict())


def slug_columns(prefix: str, labels: list[str]) -> list[str]:
    return [f"{prefix}_{label_slug(label)}" for label in labels]
