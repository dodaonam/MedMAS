from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from train_densenet.artifacts import DISEASE_LABELS, label_slug


def create_gradcam_report(
    *,
    checkpoint_path: Path,
    predictions: Any,
    figures_dir: Path,
    root: Path,
    labels: list[str] | None = None,
    max_cases_per_label: int = 2,
    device_name: str | None = None,
    target_layer_name: str = "cnn_head_final_conv",
) -> list[Path]:
    import torch
    from PIL import Image

    from train_densenet.gradcam import GradCAM, load_model_for_gradcam, target_layer_by_name
    from train_densenet.transforms import build_eval_transform, denormalize_image_tensor

    target_labels = labels or DISEASE_LABELS
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_for_gradcam(checkpoint_path, device=device, num_classes=len(target_labels))
    transform = build_eval_transform()
    cam = GradCAM(model, target_layer_by_name(model, target_layer_name))
    target_layer_slug = target_layer_name.replace(".", "_")
    paths: list[Path] = []
    try:
        for label_idx, label in enumerate(target_labels):
            slug = label_slug(label)
            prob_col = f"prob_{slug}"
            true_col = f"true_{slug}"
            pred_col = f"pred_{slug}"
            threshold_col = f"threshold_{slug}"
            case_sets = {
                "true_positive": predictions.loc[
                    (predictions[true_col] == 1) & (predictions[pred_col] == 1)
                ].sort_values(prob_col, ascending=False),
                "false_positive": predictions.loc[
                    (predictions[true_col] == 0) & (predictions[pred_col] == 1)
                ].sort_values(prob_col, ascending=False),
                "false_negative": predictions.loc[
                    (predictions[true_col] == 1) & (predictions[pred_col] == 0)
                ].sort_values(prob_col, ascending=True),
            }
            for case_type, cases in case_sets.items():
                for case_idx, (_, row) in enumerate(cases.head(max_cases_per_label).iterrows(), start=1):
                    image_path = Path(str(row["image_path"]))
                    if not image_path.is_absolute():
                        image_path = root / image_path
                    with Image.open(image_path) as image:
                        image_rgb = image.convert("RGB")
                        image_tensor = transform(image_rgb).unsqueeze(0).to(device)
                        heatmap = cam(image_tensor, label_idx)[0]
                        display_image = (
                            denormalize_image_tensor(image_tensor.squeeze(0).detach().cpu())
                            .permute(1, 2, 0)
                            .numpy()
                        )
                        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
                        axes[0].imshow(display_image)
                        axes[0].set_title("image")
                        axes[0].axis("off")
                        axes[1].imshow(display_image)
                        axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
                        probability = float(row[prob_col])
                        threshold = float(row[threshold_col])
                        axes[1].set_title(
                            f"{label} {case_type.replace('_', ' ')}\n"
                            f"p={probability:.3f} thr={threshold:.2f}",
                            fontsize=9,
                        )
                        axes[1].axis("off")
                        output_path = (
                            figures_dir
                            / "06_gradcam"
                            / f"gradcam_{target_layer_slug}_{slug}_{case_type}_{case_idx}.png"
                        )
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        fig.tight_layout()
                        fig.savefig(output_path, dpi=160)
                        plt.close(fig)
                        paths.append(output_path)
    finally:
        cam.close()
    return paths
