from __future__ import annotations

from typing import Any

from .artifacts import RARE_SAMPLER_WEAKCROP_VARIANT, V2_LOCKED_VARIANT

try:
    import torch
    from torchvision import transforms
except ModuleNotFoundError:  # pragma: no cover - exercised only on machines without torchvision
    torch = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
V2_TRAIN_TRANSFORM = "v2_random_resized_crop"
LESION_PRESERVING_TRAIN_TRANSFORM = "lesion_preserving_center_crop_v1"


def _require_torchvision() -> None:
    if transforms is None:
        raise ModuleNotFoundError("torchvision is required for DenseNet image transforms.")


def train_transform_variant_for_recipe(recipe_variant: str) -> str:
    if recipe_variant == V2_LOCKED_VARIANT:
        return V2_TRAIN_TRANSFORM
    if recipe_variant == RARE_SAMPLER_WEAKCROP_VARIANT:
        return LESION_PRESERVING_TRAIN_TRANSFORM
    raise ValueError(f"Unsupported recipe_variant {recipe_variant!r}.")


def build_train_transform(
    *,
    input_size: int = 320,
    resize_size: int = 352,
    variant: str = V2_TRAIN_TRANSFORM,
) -> Any:
    _require_torchvision()
    if variant == V2_TRAIN_TRANSFORM:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.RandomResizedCrop(input_size, scale=(0.90, 1.00), ratio=(0.97, 1.03)),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    if variant == LESION_PRESERVING_TRAIN_TRANSFORM:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(input_size),
                transforms.RandomRotation(degrees=3),
                transforms.ColorJitter(brightness=0.03, contrast=0.03),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    raise ValueError(f"Unsupported train transform variant: {variant!r}")


def build_eval_transform(*, input_size: int = 320, resize_size: int = 352) -> Any:
    _require_torchvision()
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_transform(
    split: str,
    *,
    input_size: int = 320,
    resize_size: int = 352,
    train_variant: str = V2_TRAIN_TRANSFORM,
) -> Any:
    if split == "train":
        return build_train_transform(input_size=input_size, resize_size=resize_size, variant=train_variant)
    if split in {"val", "test"}:
        return build_eval_transform(input_size=input_size, resize_size=resize_size)
    raise ValueError(f"Unsupported split: {split!r}")


def denormalize_image_tensor(image_tensor: Any) -> Any:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to denormalize image tensors.")
    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)
