from __future__ import annotations

from dataclasses import dataclass

from ..core.client import Client
from ..core.slice import Slice
from ..utils.time_model import estimate_lease_time

@dataclass
class DeterministicFeasibility:

    mem_guard_gb: float = 0.5
    time_guard_sec: float = 30.0

    def check_memory(self, client: Client, slices: list[Slice]) -> bool:
        if client.heartbeat is None:
            return False
        total = sum(s.memory_gb for s in slices)
        return total <= client.heartbeat.free_mem_gb - self.mem_guard_gb

    def check_time(
        self, client: Client, slices: list[Slice], local_steps: int, micro_batch: int
    ) -> bool:
        if client.heartbeat is None:
            return False
        est_time = self.estimate_time(client, slices, local_steps, micro_batch)
        return est_time <= client.heartbeat.online_sec - self.time_guard_sec

    def estimate_time(
        self, client: Client, slices: list[Slice], local_steps: int, micro_batch: int
    ) -> float:
        hb = client.heartbeat
        if hb is None or hb.throughput_samples_sec <= 0:
            return float("inf")
        per_slice_mb = slices[0].update_size_mb if slices else 18.0
        te = estimate_lease_time(
            local_steps, micro_batch, len(slices), per_slice_mb,
            hb.throughput_samples_sec, hb.upload_bw_mbps,
        )
        return te.total_sec

    def is_feasible(
        self, client: Client, slices: list[Slice], local_steps: int, micro_batch: int
    ) -> bool:
        return self.check_memory(client, slices) and self.check_time(
            client, slices, local_steps, micro_batch
        )

    def max_feasible_slices(self, client: Client, candidate_slices: list[Slice]) -> int:
        if client.heartbeat is None:
            return 0
        avail = client.heartbeat.free_mem_gb - self.mem_guard_gb
        count = 0
        for s in sorted(candidate_slices, key=lambda x: x.memory_gb):
            if avail >= s.memory_gb:
                avail -= s.memory_gb
                count += 1
            else:
                break
        return count
