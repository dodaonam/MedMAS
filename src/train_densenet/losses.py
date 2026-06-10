from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AsymmetricLossConfig:
    loss: str = "asymmetric"
    gamma_pos: float = 0.0
    gamma_neg: float = 4.0
    clip: float = 0.05
    eps: float = 1e-8
    reduction: str = "mean"
    reduction_scope: str = "mean over batch and labels"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        *,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported ASL reduction: {reduction!r}")
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(f"ASL shape mismatch: logits {tuple(logits.shape)}, targets {tuple(targets.shape)}")
        targets = targets.to(dtype=logits.dtype)
        probs_pos = torch.sigmoid(logits)
        probs_neg = 1.0 - probs_pos
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        log_pos = torch.log(probs_pos.clamp(min=self.eps))
        log_neg = torch.log(probs_neg.clamp(min=self.eps))
        loss = targets * log_pos + (1.0 - targets) * log_neg

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = probs_pos * targets + probs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow((1.0 - pt).clamp(min=0.0), gamma)

        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_asymmetric_loss(config: AsymmetricLossConfig | None = None) -> AsymmetricLoss:
    cfg = config or AsymmetricLossConfig()
    return AsymmetricLoss(
        gamma_pos=cfg.gamma_pos,
        gamma_neg=cfg.gamma_neg,
        clip=cfg.clip,
        eps=cfg.eps,
        reduction=cfg.reduction,
    )
