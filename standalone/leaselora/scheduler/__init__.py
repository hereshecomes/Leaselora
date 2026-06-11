from .base import BaseScheduler
from .lease_lora import LeaseLoRA
from .adaptive_lease_lora import AdaptiveLeaseLoRA

class AdaptiveCore(AdaptiveLeaseLoRA):
    name = "Adaptive-core"

    def __init__(self, slice_set, **kwargs):
        kwargs.setdefault("disable_focus", True)
        kwargs.setdefault("disable_reservation", True)
        kwargs.setdefault("disable_variance_redundancy", True)
        kwargs.setdefault("r_base", 1)
        kwargs.setdefault("r_max", 1)
        super().__init__(slice_set, **kwargs)

STRATEGIES: dict[str, type[BaseScheduler]] = {
    "lease_lora": LeaseLoRA,
    "adaptive_lease_lora": AdaptiveLeaseLoRA,
    "adaptive_core": AdaptiveCore,
}

__all__ = [
    "BaseScheduler",
    "LeaseLoRA",
    "AdaptiveLeaseLoRA",
    "AdaptiveCore",
    "STRATEGIES",
]
