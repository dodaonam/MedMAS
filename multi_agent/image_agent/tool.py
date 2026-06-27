from __future__ import annotations

import base64
import io
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.models import densenet121

from multi_agent.shared_context import PredictionResult

CONFIG_PATH = "artifacts/training/densenet121/densenet121_seed0_20260618_173103/config.json"
CHECKPOINT_PATH = "weight/checkpoint_best.pt"

with open(CONFIG_PATH) as f:
    _cfg = json.load(f)

LABELS: list[str] = _cfg["target_labels"]
THRESHOLDS: dict[str, float] = _cfg["selected_thresholds"]

# English label → Vietnamese display name
LABEL_VI: dict[str, str] = {
    "No Finding":    "Không phát hiện bất thường",
    "Infiltration":  "Thâm nhiễm phổi",
    "Effusion":      "Tràn dịch màng phổi",
    "Atelectasis":   "Xẹp phổi",
    "Nodule":        "Nốt phổi",
    "Mass":          "Khối u phổi",
}

preprocess = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model: torch.nn.Module | None = None


def get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        key = "ema_state_dict" if checkpoint.get("ema_state_dict") is not None else "model_state_dict"
        model.load_state_dict(checkpoint[key])
        model.eval()
        _model = model
    return _model


def run_inference(image_path: str) -> tuple[dict[str, PredictionResult], dict[str, str]]:
    """Returns (predictions, gradcam_outputs).

    gradcam_outputs: base64-encoded PNG per positive label (excluding 'No Finding').
    """
    model = get_model()
    original_pil = Image.open(image_path).convert("RGB")
    tensor = preprocess(original_pil).unsqueeze(0)  # (1, 3, 320, 320)

    with torch.no_grad():
        logits = model(tensor)
    scores = torch.sigmoid(logits).squeeze()  # (6,)

    predictions: dict[str, PredictionResult] = {
        label: PredictionResult(
            score=float(scores[i]),
            positive=float(scores[i]) > THRESHOLDS[label],
        )
        for i, label in enumerate(LABELS)
    }

    # Grad-CAM: Resize+CenterCrop only (no normalize) to match model input spatial size
    pil_for_cam = transforms.Compose(preprocess.transforms[:2])(original_pil)  # PIL 320×320
    rgb_img = np.float32(pil_for_cam) / 255  # (320, 320, 3) in [0, 1]

    gradcam_outputs: dict[str, str] = {}
    with GradCAM(model=model, target_layers=[model.features.norm5]) as cam:
        for i, label in enumerate(LABELS):
            if predictions[label]["positive"] and label != "No Finding":
                grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(i)])
                visualization = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)
                buf = io.BytesIO()
                Image.fromarray(visualization).save(buf, format="PNG")
                gradcam_outputs[label] = base64.b64encode(buf.getvalue()).decode()

    return predictions, gradcam_outputs
