from __future__ import annotations

from dataclasses import dataclass

import torch

@dataclass
class ClientUpdate:
    round_id: int
    client_id: int
    worker_id: str
    method: str
    num_examples: int
    local_steps: int
    delta: dict[str, torch.Tensor]
    trained_param_names: list[str]
    train_time_sec: float = 0.0
    upload_time_sec: float = 0.0
    payload_bytes: int = 0
    peak_gpu_memory_gb: float = 0.0
    deadline_miss: bool = False
    local_loss: float = 0.0

def aggregate_partial_deltas(
    global_state: dict[str, torch.Tensor],
    updates: list[ClientUpdate],
    server_lr: float = 1.0,
    weight_by: str = "num_examples",
) -> dict[str, torch.Tensor]:
    param_deltas: dict[str, list[tuple[float, torch.Tensor]]] = {}

    for update in updates:
        if update.deadline_miss:
            continue
        w = float(update.num_examples) if weight_by == "num_examples" else 1.0
        for name, delta in update.delta.items():
            if name not in param_deltas:
                param_deltas[name] = []
            param_deltas[name].append((w, delta.cpu().float()))

    aggregated_delta = {}
    for name, weighted_list in param_deltas.items():
        total_weight = sum(w for w, _ in weighted_list)
        if total_weight == 0:
            continue
        agg = torch.zeros_like(weighted_list[0][1])
        for w, d in weighted_list:
            agg += w * d
        agg /= total_weight
        aggregated_delta[name] = agg

    new_global = {}
    for name, tensor in global_state.items():
        if name in aggregated_delta:
            new_global[name] = tensor.cpu().float() + server_lr * aggregated_delta[name]
        else:
            new_global[name] = tensor.clone()

    return new_global
