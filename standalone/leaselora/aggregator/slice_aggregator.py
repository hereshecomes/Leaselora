from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class SliceUpdate:

    client_id: int
    slice_sid: tuple[int, int]
    delta: np.ndarray
    num_samples: int
    local_steps: int

class SliceAggregator:

    def __init__(self, learning_rate: float = 1.0):
        self.lr = learning_rate

    def aggregate(self, updates: list[SliceUpdate]) -> np.ndarray | None:
        if not updates:
            return None
        total_weight = sum(u.num_samples for u in updates)
        if total_weight == 0:
            return None
        agg = np.zeros_like(updates[0].delta, dtype=np.float64)
        for u in updates:
            agg += u.num_samples * u.delta.astype(np.float64)
        agg /= total_weight
        return agg

    def apply(
        self, current_params: np.ndarray, aggregated_delta: np.ndarray
    ) -> np.ndarray:
        return current_params + self.lr * aggregated_delta
