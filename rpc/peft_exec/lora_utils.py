from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

def _is_peft_trainable(name: str) -> bool:
    return "lora_" in name or "modules_to_save" in name

def build_slice_registry(model: nn.Module, bundle_size: int = 1) -> OrderedDict[str, list[str]]:
    layer_params: dict[int, list[str]] = {}
    head_params: list[str] = []

    for name, _ in model.named_parameters():
        if not _is_peft_trainable(name):
            continue
        if "modules_to_save" in name:
            head_params.append(name)
            continue
        layer_idx = _extract_layer_index(name)
        if layer_idx is None:
            layer_idx = 0
        if layer_idx not in layer_params:
            layer_params[layer_idx] = []
        layer_params[layer_idx].append(name)

    sorted_layers = sorted(layer_params.keys())

    registry = OrderedDict()
    bundle_idx = 0
    for i in range(0, len(sorted_layers), bundle_size):
        bundle_layers = sorted_layers[i:i + bundle_size]
        params = []
        for li in bundle_layers:
            params.extend(layer_params[li])
        slice_id = f"slice_{bundle_idx}"
        registry[slice_id] = sorted(params)
        bundle_idx += 1

    if head_params:
        registry["slice_head"] = sorted(head_params)

    return registry

def _extract_layer_index(param_name: str) -> int | None:
    patterns = [
        r"layer\.(\d+)\.",
        r"layers\.(\d+)\.",
        r"transformer\.h\.(\d+)\.",
        r"encoder\.layer\.(\d+)\.",
        r"distilbert\.transformer\.layer\.(\d+)\.",
        r"block\.(\d+)\.",
        r"blocks\.(\d+)\.",
    ]
    for pat in patterns:
        m = re.search(pat, param_name)
        if m:
            return int(m.group(1))
    return None

def get_all_lora_param_names(model: nn.Module) -> list[str]:
    return [n for n, _ in model.named_parameters() if _is_peft_trainable(n)]

def apply_trainable_mask(
    model: nn.Module,
    lease_param_names: list[str] | None,
    mode: str = "leaselora",
):
    lease_set = set(lease_param_names) if lease_param_names else set()

    for name, param in model.named_parameters():
        if not _is_peft_trainable(name):
            param.requires_grad = False
            continue

        if mode == "eval":
            param.requires_grad = False
        elif mode == "leaselora":
            if "modules_to_save" in name:
                param.requires_grad = True
            else:
                param.requires_grad = name in lease_set
        else:
            param.requires_grad = False

def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if _is_peft_trainable(name)
    }

def load_lora_state_dict(model: nn.Module, lora_state: dict[str, torch.Tensor]):
    current_state = model.state_dict()
    for name, tensor in lora_state.items():
        if name in current_state:
            current_state[name] = tensor.to(current_state[name].device)
    model.load_state_dict(current_state, strict=False)

def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
