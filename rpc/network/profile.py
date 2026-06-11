from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

@dataclass
class NetworkProfile:
    name: str
    rate_mbps: float = 400.0
    rtt_ms: float = 5.0
    jitter_ms: float = 2.0
    loss_percent: float = 0.0
    loss_correlation: float = 0.0
    reorder_percent: float = 0.0
    duplicate_percent: float = 0.0
    queue_limit_packets: int = 1000
    online_prob: float = 1.0

    @property
    def delay_ms(self) -> float:
        return self.rtt_ms / 2.0

    @property
    def rate_bytes_per_sec(self) -> float:
        return self.rate_mbps * 1e6 / 8.0

    def estimate_upload_time(self, payload_bytes: int) -> float:
        if self.rate_bytes_per_sec <= 0:
            return float("inf")
        transfer_time = payload_bytes / self.rate_bytes_per_sec
        latency = self.rtt_ms / 1000.0
        return transfer_time + latency

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rate_mbps": self.rate_mbps,
            "rtt_ms": self.rtt_ms,
            "jitter_ms": self.jitter_ms,
            "loss_percent": self.loss_percent,
            "loss_correlation": self.loss_correlation,
            "reorder_percent": self.reorder_percent,
            "duplicate_percent": self.duplicate_percent,
            "queue_limit_packets": self.queue_limit_packets,
            "online_prob": self.online_prob,
        }

BUILTIN_PROFILES = {
    "weak_lte": NetworkProfile(
        name="weak_lte", rate_mbps=40, rtt_ms=80, jitter_ms=30,
        loss_percent=1.0, loss_correlation=25, reorder_percent=0.1,
        queue_limit_packets=1000, online_prob=0.90,
    ),
    "mid_4g": NetworkProfile(
        name="mid_4g", rate_mbps=80, rtt_ms=50, jitter_ms=15,
        loss_percent=0.5, loss_correlation=15, reorder_percent=0.05,
        queue_limit_packets=1000, online_prob=0.95,
    ),
    "wifi_bursty": NetworkProfile(
        name="wifi_bursty", rate_mbps=160, rtt_ms=20, jitter_ms=20,
        loss_percent=0.2, loss_correlation=40, reorder_percent=0.2,
        queue_limit_packets=1000, online_prob=0.98,
    ),
    "strong_lan": NetworkProfile(
        name="strong_lan", rate_mbps=400, rtt_ms=5, jitter_ms=2,
        loss_percent=0.0, loss_correlation=0, reorder_percent=0.0,
        queue_limit_packets=1000, online_prob=1.0,
    ),
}

def load_profiles_from_yaml(path: str | Path) -> dict[str, NetworkProfile]:
    with open(path) as f:
        data = yaml.safe_load(f)

    profiles = {}
    for name, cfg in data.get("network_profiles", {}).items():
        profiles[name] = NetworkProfile(name=name, **cfg)
    return profiles

def assign_profiles_to_clients(
    num_clients: int,
    profile_names: list[str],
    policy: str = "cycle",
    seed: int = 42,
) -> dict[int, str]:
    import numpy as np
    assignments = {}

    if policy == "cycle":
        for cid in range(num_clients):
            assignments[cid] = profile_names[cid % len(profile_names)]
    elif policy == "random":
        rng = np.random.default_rng(seed)
        for cid in range(num_clients):
            assignments[cid] = rng.choice(profile_names)
    elif policy == "uniform":
        for cid in range(num_clients):
            assignments[cid] = profile_names[0] if profile_names else "strong_lan"

    return assignments
