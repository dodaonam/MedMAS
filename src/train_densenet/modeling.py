from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ._backend import require_torch, torch
from .config import TrainConfig
from .data import positive_weights


def build_model(num_labels: int, *, pretrained: bool = True, classifier_dropout: float = 0.0) -> Any:
    require_torch()
    from torchvision.models import DenseNet121_Weights, densenet121

    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    in_features = model.classifier.in_features
    if classifier_dropout > 0.0:
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=classifier_dropout),
            torch.nn.Linear(in_features, num_labels),
        )
    else:
        model.classifier = torch.nn.Linear(in_features, num_labels)
    return model


class AsymmetricLoss(torch.nn.Module):
    def __init__(self, *, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: Any, targets: Any) -> Any:
        probs = torch.sigmoid(logits)
        probs_neg = 1.0 - probs
        if self.clip > 0.0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)
        loss = targets * torch.log(probs.clamp(min=self.eps))
        loss = loss + (1.0 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0.0 or self.gamma_pos > 0.0:
            pt = targets * probs + (1.0 - targets) * probs_neg
            gamma = targets * self.gamma_pos + (1.0 - targets) * self.gamma_neg
            loss = loss * torch.pow(1.0 - pt, gamma)
        return -loss.mean()


class ExponentialMovingAverage:
    def __init__(self, model: Any, *, decay: float) -> None:
        self.decay = float(decay)
        self.model = deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def update(self, model: Any) -> None:
        with torch.no_grad():
            ema_state = self.model.state_dict()
            model_state = model.state_dict()
            for key, value in ema_state.items():
                source = model_state[key].detach()
                if torch.is_floating_point(value) or torch.is_complex(value):
                    value.copy_(value * self.decay + source * (1.0 - self.decay))
                    continue
                value.copy_(source)


def build_criterion(config: TrainConfig, frame: pd.DataFrame, labels: list[str], device: Any) -> Any:
    require_torch()
    if config.loss_name == "asl":
        return AsymmetricLoss(
            gamma_neg=config.asl_gamma_neg,
            gamma_pos=config.asl_gamma_pos,
            clip=config.asl_clip,
        )
    if config.loss_name == "bce":
        return torch.nn.BCEWithLogitsLoss(pos_weight=positive_weights(frame, labels).to(device))
    raise ValueError(f"Unsupported loss_name: {config.loss_name!r}")


def build_lr_scheduler(optimizer: Any, config: TrainConfig) -> Any:
    require_torch()
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

    cosine_t_max_value = cosine_t_max(config)
    if config.warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=cosine_t_max_value, eta_min=config.min_lr)
    warmup = LinearLR(
        optimizer,
        start_factor=config.warmup_start_factor,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_t_max_value, eta_min=config.min_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs])


def set_backbone_trainable(model: Any, *, trainable: bool) -> None:
    backbone = getattr(model, "features", None)
    if backbone is None:
        raise ValueError("DenseNet model is expected to expose .features as the backbone.")
    for parameter in backbone.parameters():
        parameter.requires_grad_(trainable)
    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        for parameter in classifier.parameters():
            parameter.requires_grad_(True)


def train_one_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    *,
    ema: ExponentialMovingAverage | None = None,
    backbone_trainable: bool = True,
    desc: str = "train",
) -> float:
    model.train()
    if not backbone_trainable:
        backbone = getattr(model, "features", None)
        if backbone is not None:
            backbone.eval()
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
        if ema is not None:
            ema.update(model)
        batch_size = int(images.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_rows += batch_size
        progress.set_postfix(loss=f"{total_loss / max(total_rows, 1):.4f}")
    return total_loss / max(total_rows, 1)


def set_seed(seed: int) -> None:
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> Any:
    require_torch()
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scheduler_name(config: TrainConfig) -> str:
    return "linear_warmup_cosine_annealing" if config.warmup_epochs > 0 else "cosine_annealing"


def cosine_t_max(config: TrainConfig) -> int:
    return max(config.epochs - config.warmup_epochs - 1, 1)
