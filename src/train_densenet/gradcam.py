from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised only on machines without torch
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or F is None:
        raise ModuleNotFoundError("PyTorch is required for Grad-CAM generation.")


class GradCAM:
    def __init__(self, model: Any, target_layer: Any) -> None:
        _require_torch()
        self.model = model
        self.target_layer = target_layer
        self.activations: Any | None = None
        self.gradients: Any | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def _save_activation(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module: Any, _grad_input: Any, grad_output: Any) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image_batch: Any, class_index: int) -> np.ndarray:
        _require_torch()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_batch)
        score = logits[:, class_index].sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_batch.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        cam_min = flat.min(dim=1).values.view(-1, 1, 1)
        cam_max = flat.max(dim=1).values.view(-1, 1, 1)
        normalized = (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-8)
        return normalized.detach().cpu().numpy()


def load_model_for_gradcam(
    checkpoint_path: Path,
    *,
    device: Any,
    num_classes: int = 5,
) -> Any:
    _require_torch()
    from .model import DenseNet121CNNHead

    model = DenseNet121CNNHead(num_classes=num_classes, weights=None)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def default_target_layer(model: Any) -> Any:
    from .model import final_cnn_head_conv

    return final_cnn_head_conv(model)


def target_layer_by_name(model: Any, target_layer_name: str) -> Any:
    if target_layer_name in {"cnn_head_final_conv", "cnn_head"}:
        return default_target_layer(model)
    if target_layer_name in {"backbone.denseblock4", "denseblock4"}:
        return model.backbone.denseblock4
    raise ValueError(f"Unsupported Grad-CAM target layer: {target_layer_name!r}")
