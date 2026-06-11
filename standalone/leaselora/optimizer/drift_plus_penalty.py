from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.client import Client
from ..core.lease import Lease
from ..core.slice import Slice, SliceSet
from ..utils.time_model import estimate_lease_time
from .virtual_queue import CoverageQueue, FairnessQueue
from .granularity import GranularityIndex

@dataclass
class DriftPlusPenaltyScheduler:

    V: float = 50.0
    cost_weight: float = 0.05
    coverage_weight: float = 0.08
    fairness_weight: float = 0.06
    utilization_ratio: float = 0.82

    coverage_queue: Optional[CoverageQueue] = None
    fairness_queue: Optional[FairnessQueue] = None
    granularity: GranularityIndex = field(default_factory=GranularityIndex)

    def init_queues(self, slice_set: SliceSet, coverage_target: float = 0.8,
                    participation_target: float = 0.4) -> None:
        sids = [s.sid for s in slice_set]
        self.coverage_queue = CoverageQueue(sids, coverage_target, max_backlog=5.0)
        self.fairness_queue = FairnessQueue(participation_target, max_backlog=5.0)

    def score_assignment(
        self,
        client: Client,
        s: Slice,
        g_index: float,
    ) -> float:
        progress_gain = self._estimate_progress(client, s)
        cost = self._estimate_cost(client, s)

        score = self.V * progress_gain - self.cost_weight * cost

        if self.coverage_queue is not None:
            q_s = self.coverage_queue.get_deficit(s.sid)
            max_q = self.coverage_queue.queues[s.sid].max_backlog
            score += self.coverage_weight * (q_s / max(max_q, 1.0))

        if self.fairness_queue is not None and client.is_low_resource:
            z = self.fairness_queue.deficit
            max_z = self.fairness_queue.max_backlog
            bonus = self.granularity.low_resource_bonus_weight(g_index)
            score += self.fairness_weight * (z / max(max_z, 1.0)) + bonus

        affinity = client.get_affinity(s.group)
        match_score = affinity / max(0.75, s.difficulty)
        score += 0.3 * match_score

        score += 0.2 * client.freshness

        return score

    def _estimate_progress(self, client: Client, s: Slice) -> float:
        base_gain = 0.04
        dr_rate = 5.0

        affinity = client.get_affinity(s.group)
        need = s.need
        diminishing = 1.0 - np.exp(-dr_rate * need)

        throughput = 10.0
        online_sec = 120.0
        if client.heartbeat is not None:
            throughput = max(client.heartbeat.throughput_samples_sec, 0.01)
            online_sec = client.heartbeat.online_sec

        e_max = 56.0
        micro_batch = 8
        available_time = online_sec * self.utilization_ratio
        max_steps = available_time * throughput / micro_batch
        step_ratio = min(max_steps, e_max) / e_max

        effort = base_gain * np.sqrt(step_ratio) * affinity

        k_est = max(self.granularity.base_slice_budget(
            0.5, not client.is_low_resource), 1)
        spread = 1.0 / np.sqrt(max(k_est - 1, 1))

        return max(0.0, effort * diminishing * spread)

    def _estimate_cost(self, client: Client, s: Slice) -> float:
        if client.heartbeat is None:
            return float("inf")
        hb = client.heartbeat
        mem_cost = s.memory_gb / max(hb.free_mem_gb, 0.1)
        comm_cost = s.update_size_mb / max(hb.upload_bw_mbps * 10.0, 0.1)
        return mem_cost + comm_cost

    def select_focus_set(
        self,
        slice_set: SliceSet,
        total_round_budget: int,
        g_index: float,
        min_focus: int = 4,
    ) -> list[Slice]:
        mu = self.granularity.focus_multiplier(g_index)
        focus_size = max(min_focus, int(total_round_budget * mu))
        focus_size = min(focus_size, max(min_focus, int(len(slice_set) * 0.9)))

        scores = []
        for s in slice_set:
            q_s = 0.0
            if self.coverage_queue is not None:
                q_s = self.coverage_queue.get_deficit(s.sid)
            priority = (
                1.2 * (q_s / 10.0)
                + 1.0 * s.need * s.intrinsic_importance
                + 0.25 * s.difficulty
            )
            scores.append((priority, s))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scores[:focus_size]]

    def compute_local_steps(
        self,
        client: Client,
        assigned_slices: list[Slice],
        g_index: float,
        e_base: int = 24,
        step_delta: int = 4,
    ) -> int:
        if client.heartbeat is None:
            return e_base
        e_min = max(step_delta, e_base // 3)
        capability = min(1.0, client.heartbeat.throughput_samples_sec / 500.0)
        e_scaled = max(e_min, int(e_base * capability))
        e_scaled = max(e_min, (e_scaled // step_delta) * step_delta or step_delta)
        return e_scaled

    def compute_slice_budget(self, client: Client, g_index: float) -> int:
        k_base = self.granularity.base_slice_budget(
            g_index, not client.is_low_resource
        )
        if client.heartbeat is None:
            return k_base
        hb = client.heartbeat
        k_mem = int((hb.free_mem_gb - 0.5) / max(0.14, 0.22))
        est_per_slice_time = 40.0
        k_time = int(
            (hb.online_sec * self.utilization_ratio) / max(est_per_slice_time, 1.0)
        )
        return max(1, min(k_base, k_mem, k_time))

    def update_queues(
        self,
        accepted_counts: dict[tuple[int, int], int],
        low_resource_rate: float,
    ) -> None:
        if self.coverage_queue is not None:
            self.coverage_queue.update(accepted_counts)
        if self.fairness_queue is not None:
            self.fairness_queue.update(low_resource_rate)
