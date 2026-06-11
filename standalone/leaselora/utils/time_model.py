from __future__ import annotations

from dataclasses import dataclass

@dataclass
class TimeEstimate:
    compute_sec: float
    upload_sec: float

    @property
    def total_sec(self) -> float:
        return self.compute_sec + self.upload_sec

def estimate_lease_time(
    local_steps: int,
    micro_batch: int,
    num_slices: int,
    per_slice_update_mb: float,
    throughput_samples_sec: float,
    upload_bw_mbps: float,
) -> TimeEstimate:
    samples = local_steps * micro_batch
    compute_sec = samples / max(throughput_samples_sec, 0.01)
    upload_mb = num_slices * per_slice_update_mb
    upload_sec = upload_mb / max(upload_bw_mbps, 0.01)
    return TimeEstimate(compute_sec=compute_sec, upload_sec=upload_sec)
