from __future__ import annotations

import torch

def extract_lora_delta(
    local_state: dict[str, torch.Tensor],
    global_state: dict[str, torch.Tensor],
    lease_param_names: list[str] | None = None,
) -> tuple[dict[str, torch.Tensor], int, int]:
    delta_dict = {}
    num_params = 0

    target_names = lease_param_names if lease_param_names else list(local_state.keys())

    for name in target_names:
        if name not in local_state or name not in global_state:
            continue
        local_t = local_state[name].cpu().float()
        global_t = global_state[name].cpu().float()
        delta = local_t - global_t
        delta_dict[name] = delta
        num_params += delta.numel()

    payload_bytes = num_params * 4
    return delta_dict, num_params, payload_bytes

def apply_delta_to_global(
    global_state: dict[str, torch.Tensor],
    aggregated_delta: dict[str, torch.Tensor],
    server_lr: float = 1.0,
) -> dict[str, torch.Tensor]:
    for name, delta in aggregated_delta.items():
        if name in global_state:
            global_state[name] = global_state[name].cpu().float() + server_lr * delta.cpu().float()
    return global_state
