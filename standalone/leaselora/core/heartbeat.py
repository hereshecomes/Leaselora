from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Heartbeat:

    free_mem_gb: float
    upload_bw_mbps: float
    throughput_samples_sec: float
    online_sec: float
    battery_ok: bool = True

    def is_viable(self, min_mem: float = 0.3, min_online: float = 60.0) -> bool:
        return (
            self.battery_ok
            and self.free_mem_gb >= min_mem
            and self.online_sec >= min_online
        )
