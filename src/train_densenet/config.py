from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ._backend import require_torch, torch
from .constants import MODEL_NAME, TARGET_LABELS


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
    classifier_dropout: float = 0.2
    loss_name: str = "asl"
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    ema_decay: float = 0.999
    threshold: float = 0.5
    head_only: bool = False
    freeze_backbone_epochs: int = 0
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
    require_torch()
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


def validate_train_config(config: TrainConfig) -> None:
    if config.head_only and config.freeze_backbone_epochs != 0:
        raise ValueError("head_only cannot be combined with freeze_backbone_epochs")
    if config.freeze_backbone_epochs < 0:
        raise ValueError("freeze_backbone_epochs must be non-negative")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not config.head_only and config.freeze_backbone_epochs >= config.epochs:
        raise ValueError("freeze_backbone_epochs must be smaller than epochs")


def config_to_json(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def git_commit_hash(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def torchvision_version() -> str | None:
    try:
        import torchvision
    except ModuleNotFoundError:
        return None
    return str(getattr(torchvision, "__version__", "")) or None


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
