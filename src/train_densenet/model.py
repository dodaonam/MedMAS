from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121CNNHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        weights: DenseNet121_Weights | None = DenseNet121_Weights.IMAGENET1K_V1,
    ) -> None:
        super().__init__()
        base = densenet121(weights=weights)
        self.backbone = base.features
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.10),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.45)
        self.classifier = nn.Linear(128, num_classes)
        self._init_head()

    def _init_head(self) -> None:
        for module in self.cnn_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.relu(x, inplace=True)
        x = self.cnn_head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def set_backbone_batchnorm_eval(model: nn.Module) -> None:
    for module in model.backbone.modules():  # type: ignore[attr-defined]
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def configure_stage1(model: nn.Module) -> None:
    for param in model.backbone.parameters():  # type: ignore[attr-defined]
        param.requires_grad = False
    for param in model.cnn_head.parameters():  # type: ignore[attr-defined]
        param.requires_grad = True
    for param in model.classifier.parameters():  # type: ignore[attr-defined]
        param.requires_grad = True

    model.train()
    model.backbone.eval()  # type: ignore[attr-defined]
    model.cnn_head.train()  # type: ignore[attr-defined]
    model.dropout.train()  # type: ignore[attr-defined]
    model.classifier.train()  # type: ignore[attr-defined]
    set_backbone_batchnorm_eval(model)


def configure_stage2(model: nn.Module) -> None:
    for param in model.backbone.parameters():  # type: ignore[attr-defined]
        param.requires_grad = False
    for param in model.backbone.denseblock4.parameters():  # type: ignore[attr-defined]
        param.requires_grad = True
    for param in model.backbone.norm5.parameters():  # type: ignore[attr-defined]
        param.requires_grad = True
    for param in model.cnn_head.parameters():  # type: ignore[attr-defined]
        param.requires_grad = True
    for param in model.classifier.parameters():  # type: ignore[attr-defined]
        param.requires_grad = True

    model.train()
    set_backbone_batchnorm_eval(model)
    model.cnn_head.train()  # type: ignore[attr-defined]
    model.dropout.train()  # type: ignore[attr-defined]
    model.classifier.train()  # type: ignore[attr-defined]


def cnn_head_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    yield from model.cnn_head.parameters()  # type: ignore[attr-defined]
    yield from model.classifier.parameters()  # type: ignore[attr-defined]


def denseblock4_norm5_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    yield from model.backbone.denseblock4.parameters()  # type: ignore[attr-defined]
    yield from model.backbone.norm5.parameters()  # type: ignore[attr-defined]


def stage1_optimizer_parameters(model: nn.Module, lr: float, weight_decay: float) -> list[dict[str, Any]]:
    return [
        {
            "name": "cnn_head",
            "params": [param for param in cnn_head_parameters(model) if param.requires_grad],
            "lr": lr,
            "weight_decay": weight_decay,
        }
    ]


def stage2_optimizer_parameters(
    model: nn.Module,
    *,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "cnn_head",
            "params": [param for param in cnn_head_parameters(model) if param.requires_grad],
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
        {
            "name": "denseblock4_norm5",
            "params": [param for param in denseblock4_norm5_parameters(model) if param.requires_grad],
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        },
    ]


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def final_cnn_head_conv(model: nn.Module) -> nn.Conv2d:
    for module in reversed(list(model.cnn_head.modules())):  # type: ignore[attr-defined]
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in model.cnn_head")
