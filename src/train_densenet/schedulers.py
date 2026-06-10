from __future__ import annotations

import math
from typing import Any


def optimizer_steps_per_epoch(num_batches: int, gradient_accumulation_steps: int) -> int:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    return int(math.ceil(num_batches / gradient_accumulation_steps))


def accumulation_window_microbatch_count(
    *,
    batch_index: int,
    total_batches: int,
    gradient_accumulation_steps: int,
) -> int:
    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if batch_index < 1 or batch_index > total_batches:
        raise ValueError(f"batch_index must be in [1, {total_batches}], got {batch_index}")
    window_start = batch_index - ((batch_index - 1) % gradient_accumulation_steps)
    window_end = min(window_start + gradient_accumulation_steps - 1, total_batches)
    return int(window_end - window_start + 1)


def batch_example_count(
    *,
    batch_index: int,
    total_batches: int,
    total_examples: int,
    batch_size: int,
) -> int:
    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_index < 1 or batch_index > total_batches:
        raise ValueError(f"batch_index must be in [1, {total_batches}], got {batch_index}")
    if batch_index < total_batches:
        return int(batch_size)
    remaining = total_examples - batch_size * (batch_index - 1)
    if remaining <= 0:
        raise ValueError("total_examples is inconsistent with batch_index and batch_size")
    return int(min(batch_size, remaining))


def accumulation_window_example_count(
    *,
    batch_index: int,
    total_batches: int,
    total_examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if batch_index < 1 or batch_index > total_batches:
        raise ValueError(f"batch_index must be in [1, {total_batches}], got {batch_index}")
    window_start = batch_index - ((batch_index - 1) % gradient_accumulation_steps)
    window_end = min(window_start + gradient_accumulation_steps - 1, total_batches)
    return int(
        sum(
            batch_example_count(
                batch_index=index,
                total_batches=total_batches,
                total_examples=total_examples,
                batch_size=batch_size,
            )
            for index in range(window_start, window_end + 1)
        )
    )


def build_warmup_cosine_scheduler(
    optimizer: Any,
    *,
    total_optimizer_steps: int,
    warmup_ratio: float = 0.10,
    min_lr_factor: float = 0.01,
) -> Any:
    if total_optimizer_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0 <= min_lr_factor <= 1:
        raise ValueError("min_lr_factor must be in [0, 1]")

    import torch

    warmup_steps = int(round(total_optimizer_steps * warmup_ratio))

    def lr_factor(step_index: int) -> float:
        current_step = min(step_index + 1, total_optimizer_steps)
        if warmup_steps > 0 and current_step <= warmup_steps:
            return max(current_step / warmup_steps, min_lr_factor)
        decay_steps = max(total_optimizer_steps - warmup_steps, 1)
        progress = min(max((current_step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
