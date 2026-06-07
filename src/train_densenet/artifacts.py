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


def select_pos_weights(raw_pos_weights: dict[str, float], target_labels: list[str] | None = None) -> dict[str, float]:
    labels = target_labels or TARGET_LABELS
    selected: dict[str, float] = {}
    for label in labels:
        raw_value = float(raw_pos_weights[label])
        if label == "No Finding":
            selected[label] = max(raw_value, 1.0)
        else:
            selected[label] = min(raw_value, 10.0)
    return selected


def build_class_weights_payload(
    *,
    target_labels: list[str],
    train_positive_counts: dict[str, int],
    train_negative_counts: dict[str, int],
) -> dict[str, Any]:
    raw_pos_weights: dict[str, float] = {}
    for label in target_labels:
        positives = int(train_positive_counts[label])
        negatives = int(train_negative_counts[label])
        if positives <= 0:
            raise ValueError(f"Cannot compute pos_weight for {label!r}: train split has no positive samples.")
        raw_pos_weights[label] = negatives / positives
    selected = select_pos_weights(raw_pos_weights, target_labels)
    return {
        "target_label_order": target_labels,
        "train_positive_counts": train_positive_counts,
        "train_negative_counts": train_negative_counts,
        "raw_train_only_pos_weight": raw_pos_weights,
        "selected_clipped_pos_weight": selected,
        "selection_rule": "Computed from train split only; No Finding floored at 1.00; disease labels capped at 10.00",
    }


def build_class_weights_payload_from_manifest(manifest: Any, target_labels: list[str]) -> dict[str, Any]:
    train_frame = manifest.loc[manifest["split"].astype(str) == "train"]
    if train_frame.empty:
        raise ValueError("Cannot compute class weights: train split is empty.")
    positive_counts = train_frame[target_labels].sum(axis=0).astype(int).to_dict()
    negative_counts = (len(train_frame) - train_frame[target_labels].sum(axis=0)).astype(int).to_dict()
    return build_class_weights_payload(
        target_labels=target_labels,
        train_positive_counts={label: int(positive_counts[label]) for label in target_labels},
        train_negative_counts={label: int(negative_counts[label]) for label in target_labels},
    )


def write_class_weights(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)


def selected_pos_weights_from_payload(payload: dict[str, Any], target_labels: list[str] | None = None) -> dict[str, float]:
    labels = target_labels or TARGET_LABELS
    weights = payload["selected_clipped_pos_weight"]
    missing = [label for label in labels if label not in weights]
    if missing:
        raise ValueError(f"Missing selected pos_weight values for labels: {missing}")
    return {label: float(weights[label]) for label in labels}


def slug_columns(prefix: str, labels: list[str]) -> list[str]:
    return [f"{prefix}_{label_slug(label)}" for label in labels]
