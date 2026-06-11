from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

@dataclass
class VarianceTracker:

    _count: dict[tuple[int, int], int] = field(
        default_factory=lambda: defaultdict(int), repr=False
    )
    _mean: dict[tuple[int, int], float] = field(
        default_factory=lambda: defaultdict(float), repr=False
    )
    _m2: dict[tuple[int, int], float] = field(
        default_factory=lambda: defaultdict(float), repr=False
    )

    def update(self, sid: tuple[int, int], delta_norm: float) -> None:
        self._count[sid] += 1
        n = self._count[sid]
        d = delta_norm - self._mean[sid]
        self._mean[sid] += d / n
        d2 = delta_norm - self._mean[sid]
        self._m2[sid] += d * d2

    def variance(self, sid: tuple[int, int]) -> float:
        n = self._count[sid]
        if n < 2:
            return 1.0
        return self._m2[sid] / (n - 1)

    def std(self, sid: tuple[int, int]) -> float:
        return math.sqrt(self.variance(sid))

    def compute_redundancy(
        self,
        sid: tuple[int, int],
        progress: float,
        r_base: int = 1,
        r_max: int = 2,
        variance_threshold: float = 0.5,
        progress_threshold: float = 0.3,
        coverage_debt: float = 0.0,
        debt_ceiling: float = 3.0,
    ) -> int:
        if r_max <= r_base:
            return r_base
        v = self.variance(sid)
        need = 1.0 - progress
        if v >= variance_threshold and need >= progress_threshold and coverage_debt < debt_ceiling:
            return min(r_base + 1, r_max)
        return r_base

    def predicted_agg_variance(self, sid: tuple[int, int], redundancy: int) -> float:
        if redundancy <= 0:
            return self.variance(sid)
        return self.variance(sid) / redundancy

    def reset(self) -> None:
        self._count.clear()
        self._mean.clear()
        self._m2.clear()
