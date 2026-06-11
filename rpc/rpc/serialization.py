from __future__ import annotations

import io
import time
from typing import Any

import torch

def serialize_delta(delta: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    cpu_delta = {k: v.cpu() for k, v in delta.items()}
    torch.save(cpu_delta, buf)
    return buf.getvalue()

def deserialize_delta(data: bytes) -> dict[str, torch.Tensor]:
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)

def serialize_adapter_state(state: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    cpu_state = {k: v.cpu() for k, v in state.items()}
    torch.save(cpu_state, buf)
    return buf.getvalue()

def deserialize_adapter_state(data: bytes) -> dict[str, torch.Tensor]:
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)

def measure_serialization(delta: dict[str, torch.Tensor]) -> tuple[bytes, float, int]:
    t0 = time.time()
    data = serialize_delta(delta)
    t1 = time.time()
    return data, t1 - t0, len(data)
