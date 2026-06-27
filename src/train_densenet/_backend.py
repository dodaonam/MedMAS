from __future__ import annotations

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ModuleNotFoundError:
    torch = None
    DataLoader = None
    Dataset = object
    WeightedRandomSampler = None


def require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch and torchvision are required to train DenseNet121.")
