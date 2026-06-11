from __future__ import annotations

import math
from dataclasses import dataclass

@dataclass
class GranularityIndex:

    alpha1: float = 0.02
    alpha2: float = 0.3
    alpha3: float = 0.5

    fine_centre: float = 3.0
    coarse_centre: float = 1.0

    def compute(
        self,
        num_slices: int,
        per_slice_mem_gb: float,
        feasible_client_dispersion: float = 0.5,
    ) -> float:
        return (
            self.alpha1 * num_slices
            + self.alpha2 / max(per_slice_mem_gb, 0.01)
            + self.alpha3 * feasible_client_dispersion
        )

    @staticmethod
    def _sigmoid(x: float, centre: float, steepness: float = 3.0) -> float:
        return 1.0 / (1.0 + math.exp(-steepness * (x - centre)))

    def fine_weight(self, g: float) -> float:
        return self._sigmoid(g, self.fine_centre)

    def coarse_weight(self, g: float) -> float:
        return 1.0 - self._sigmoid(g, self.coarse_centre)

    def max_local_steps(self, g: float, e_base: int = 24) -> int:
        w_fine = self.fine_weight(g)
        e_max = e_base + int(round(32 * w_fine))
        return max(e_base, e_max)

    def focus_multiplier(self, g: float) -> float:
        w_fine = self.fine_weight(g)
        return 1.0 + 0.10 * w_fine

    def reservation_ratio(self, g: float) -> float:
        w_coarse = self.coarse_weight(g)
        return 0.15 + 0.25 * w_coarse

    def low_resource_bonus_weight(self, g: float) -> float:
        w_coarse = self.coarse_weight(g)
        return 0.12 + 0.20 * w_coarse

    def base_slice_budget(self, g: float, device_large: bool) -> int:
        base = 4 if device_large else 2
        w_fine = self.fine_weight(g)
        cap = max(1, int(round(base * (1.0 - 0.3 * w_fine))))
        return cap
