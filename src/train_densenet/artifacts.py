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

RAW_POS_WEIGHTS: dict[str, float] = {
    "No Finding": 0.68,
    "Infiltration": 4.41,
    "Effusion": 6.52,
    "Atelectasis": 8.79,
    "Nodule": 15.13,
    "Mass": 16.91,
}

SELECTED_POS_WEIGHTS: dict[str, float] = {
    "No Finding": 1.00,
    "Infiltration": 4.41,
    "Effusion": 6.52,
    "Atelectasis": 8.79,
    "Nodule": 10.00,
    "Mass": 10.00,
}


def label_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    return slug.strip("_")


def default_training_output_dir(root: Path) -> Path:
    return root / "artifacts" / "training" / MODEL_NAME


def create_run_id(seed: int, timestamp: datetime | None = None) -> str:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{MODEL_NAME}_seed{seed}_{stamp}"


def load_target_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        labels = payload
    else:
        labels = payload.get("target_labels")
    if labels != TARGET_LABELS:
        raise ValueError(f"Unexpected target label order in {path}: {labels!r}")
    return list(labels)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class TrainingArtifactPaths:
    output_dir: Path
    config_path: Path
    class_weights_path: Path
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
        class_weights_path=output_dir / "class_weights.json",
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
    target_label_order: list[str] = field(default_factory=lambda: list(TARGET_LABELS))
    checkpoint_selection_metric: str = "validation_disease_macro_average_precision"
    stage1_name: str = "cnn_head_only"
    stage2_name: str = "denseblock4_norm5_finetune"
    stage2_start_epoch: int | None = None
    stage1_epochs_planned: list[int] = field(default_factory=lambda: [3, 5])
    stage2_min_epochs_before_early_stop: int = 5
    dense_net_variant: str = "torchvision.models.densenet121"
    input_size: int = 224
    image_normalization: dict[str, list[float]] = field(
        default_factory=lambda: {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    )
    cnn_head: str = "1024->256->128->128->GAP->Dropout->Linear(6)"
    backbone_batchnorm_policy: str = "backbone BN running statistics frozen in both training stages"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_run_config(path: Path, config: RunConfig) -> None:
    save_json(path, config.to_dict())


def write_class_weights(path: Path, target_labels: list[str] | None = None) -> None:
    labels = target_labels or TARGET_LABELS
    payload = {
        "target_label_order": labels,
        "raw_train_only_pos_weight": {label: RAW_POS_WEIGHTS[label] for label in labels},
        "selected_clipped_pos_weight": {label: SELECTED_POS_WEIGHTS[label] for label in labels},
        "selection_rule": "No Finding set to 1.00; rare disease labels clipped at 10.00",
    }
    save_json(path, payload)


def slug_columns(prefix: str, labels: list[str]) -> list[str]:
    return [f"{prefix}_{label_slug(label)}" for label in labels]
