from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .slice import Slice

@dataclass
class Lease:

    client_id: int
    slices: list[Slice] = field(default_factory=list)
    local_steps: int = 24
    micro_batch: int = 8
    deadline_sec: float = 120.0
    token_budget: int = 0
    quant_mode: str = "fp16"

    completed: bool = False
    wall_time_sec: float = 0.0
    samples_consumed: int = 0
    updates: Optional[dict[tuple[int, int], object]] = None

    @property
    def num_slices(self) -> int:
        return len(self.slices)

    @property
    def total_mem_gb(self) -> float:
        return sum(s.memory_gb for s in self.slices)

    @property
    def total_update_mb(self) -> float:
        return sum(s.update_size_mb for s in self.slices)
