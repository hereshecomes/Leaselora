from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.client import Client
from ..core.slice import Slice
from ..utils.time_model import estimate_lease_time

DEVICE_CLASS_NAMES = ("SMALL", "MEDIUM", "LARGE")

@dataclass
class ChanceConstrainedFeasibility:

    target_delta: float = 0.05
    min_history: int = 5
    _mem_errors: dict[int, list[float]] = field(default_factory=dict, repr=False)
    _time_errors: dict[int, list[float]] = field(default_factory=dict, repr=False)
    _class_mem_errors: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _class_time_errors: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _client_class: dict[int, str] = field(default_factory=dict, repr=False)

    def register_client_class(self, client_id: int, device_class_name: str) -> None:
        self._client_class[client_id] = device_class_name

    def seed_class_prior(
        self, device_class_name: str, mem_errors: list[float], time_errors: list[float],
    ) -> None:
        self._class_mem_errors.setdefault(device_class_name, []).extend(mem_errors)
        self._class_time_errors.setdefault(device_class_name, []).extend(time_errors)

    def record_outcome(
        self,
        client_id: int,
        predicted_free_mem: float,
        actual_free_mem: float,
        predicted_online: float,
        actual_online: float,
    ) -> None:
        mem_err = predicted_free_mem - actual_free_mem
        time_err = predicted_online - actual_online
        self._mem_errors.setdefault(client_id, []).append(mem_err)
        self._time_errors.setdefault(client_id, []).append(time_err)
        cls = self._client_class.get(client_id)
        if cls:
            self._class_mem_errors.setdefault(cls, []).append(mem_err)
            self._class_time_errors.setdefault(cls, []).append(time_err)

    def _adaptive_guard(self, errors: list[float]) -> float:
        if len(errors) < self.min_history:
            return 0.5
        sorted_err = sorted(errors)
        idx = min(
            int(math.ceil((1.0 - self.target_delta) * len(sorted_err))) - 1,
            len(sorted_err) - 1,
        )
        return max(sorted_err[idx], 0.0)

    def _get_errors(self, client_id: int, per_client: dict, per_class: dict) -> list[float]:
        client_errs = per_client.get(client_id, [])
        if len(client_errs) >= self.min_history:
            return client_errs
        cls = self._client_class.get(client_id)
        if cls:
            class_errs = per_class.get(cls, [])
            if len(class_errs) >= self.min_history:
                return class_errs
        return client_errs

    def get_mem_guard(self, client_id: int) -> float:
        errors = self._get_errors(client_id, self._mem_errors, self._class_mem_errors)
        return self._adaptive_guard(errors)

    def get_time_guard(self, client_id: int) -> float:
        errors = self._get_errors(client_id, self._time_errors, self._class_time_errors)
        return self._adaptive_guard(errors)

    def is_feasible(
        self,
        client: Client,
        slices: list[Slice],
        local_steps: int,
        micro_batch: int,
    ) -> bool:
        if client.heartbeat is None:
            return False
        hb = client.heartbeat
        mem_guard = self.get_mem_guard(client.cid)
        time_guard = self.get_time_guard(client.cid)

        total_mem = sum(s.memory_gb for s in slices)
        if total_mem > hb.free_mem_gb - mem_guard:
            return False

        per_slice_mb = slices[0].update_size_mb if slices else 18.0
        te = estimate_lease_time(
            local_steps, micro_batch, len(slices), per_slice_mb,
            hb.throughput_samples_sec, hb.upload_bw_mbps,
        )
        if te.total_sec > hb.online_sec - time_guard:
            return False

        return True
