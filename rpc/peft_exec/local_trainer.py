from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@dataclass
class TrainResult:
    local_loss: float
    num_examples: int
    actual_steps: int
    train_time_sec: float
    peak_gpu_memory_gb: float

def local_train(
    model: nn.Module,
    dataloader: DataLoader,
    local_steps: int,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    device: torch.device | str = "cuda",
) -> TrainResult:
    model.to(device).train()
    torch.cuda.reset_peak_memory_stats(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return TrainResult(
            local_loss=0.0, num_examples=0, actual_steps=0,
            train_time_sec=0.0, peak_gpu_memory_gb=0.0,
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    total_loss = 0.0
    total_examples = 0
    step_count = 0
    t_start = time.time()

    data_iter = iter(dataloader)
    for _ in range(local_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()

        bs = batch.get("input_ids", batch.get("pixel_values", batch.get("labels"))).shape[0]
        total_loss += loss.item() * bs
        total_examples += bs
        step_count += 1

    train_time = time.time() - t_start
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    avg_loss = total_loss / max(total_examples, 1)
    return TrainResult(
        local_loss=avg_loss,
        num_examples=total_examples,
        actual_steps=step_count,
        train_time_sec=train_time,
        peak_gpu_memory_gb=peak_mem,
    )
