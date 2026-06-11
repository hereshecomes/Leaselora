from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .heartbeat import Heartbeat

class DeviceClass(Enum):
    SMALL = "small"
    LARGE = "large"

@dataclass
class Client:

    cid: int
    device_class: DeviceClass = DeviceClass.SMALL
    heartbeat: Optional[Heartbeat] = None
    affinity: dict[int, float] = field(default_factory=dict)
    local_data_size: int = 500
    rounds_since_last: int = 0
    total_accepted: int = 0

    @property
    def is_low_resource(self) -> bool:
        return self.device_class == DeviceClass.SMALL

    @property
    def freshness(self) -> float:
        return min(1.0, self.rounds_since_last / 10.0)

    def get_affinity(self, group: int) -> float:
        return self.affinity.get(group, 0.5)
