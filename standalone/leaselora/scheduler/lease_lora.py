from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from ..core.client import Client
from ..core.lease import Lease
from ..core.slice import Slice, SliceSet
from ..feasibility.deterministic import DeterministicFeasibility
from .base import BaseScheduler

class LeaseLoRA(BaseScheduler):
    name = "LeaseLoRA"

    def __init__(
        self,
        slice_set: SliceSet,
        clients_per_round: int = 16,
        local_steps: int = 24,
        micro_batch: int = 8,
        coverage_window: int = 10,
        coverage_target: int = 3,
        r_base: int = 3,
        r_max: int = 5,
        **kwargs,
    ):
        super().__init__(slice_set, **kwargs)
        self.clients_per_round = clients_per_round
        self.local_steps = local_steps
        self.micro_batch = micro_batch
        self.coverage_window = coverage_window
        self.coverage_target = coverage_target
        self.r_base = r_base
        self.r_max = r_max
        self.feasibility = DeterministicFeasibility()
        self._coverage_history: dict[tuple[int, int], list[int]] = defaultdict(list)

    def _detect_regime(self) -> str:
        n = len(self.slice_set)
        avg_mem = sum(s.memory_gb for s in self.slice_set) / max(n, 1)
        if n >= 40 or avg_mem <= 0.16:
            return "fine"
        if n <= 12 or avg_mem >= 0.30:
            return "coarse"
        return "balanced"

    def _slice_budget(self, client: Client, regime: str) -> int:
        base = 2 if client.is_low_resource else 4
        if regime == "fine":
            base = min(base, 3)
        elif regime == "coarse":
            base = min(base, 2)
        if client.heartbeat is not None:
            k_mem = int((client.heartbeat.free_mem_gb - 0.5) / 0.22)
            base = min(base, max(1, k_mem))
        return max(1, base)

    def _compute_fairness_debt(self, sid: tuple[int, int]) -> float:
        hist = self._coverage_history.get(sid, [])
        recent = hist[-self.coverage_window :]
        count = sum(recent)
        return max(0.0, self.coverage_target - count)

    def _redundancy_cap(self, s: Slice) -> int:
        debt = self._compute_fairness_debt(s.sid)
        r = self.r_base
        if debt >= 1.0:
            r += 1
        if s.need >= 0.55:
            r += 1
        if s.difficulty >= 1.15 and s.need >= 0.35:
            r += 1
        return min(r, self.r_max)

    def _utility(self, client: Client, s: Slice) -> float:
        alpha, beta, gamma, delta = 0.35, 0.25, 0.15, 0.25
        bq = s.blended_quality()
        debt = self._compute_fairness_debt(s.sid)
        fresh = client.freshness
        affinity = client.get_affinity(s.group)
        match = affinity / max(0.75, s.difficulty)

        hb = client.heartbeat
        if hb is None:
            return -float("inf")
        cost = s.memory_gb / max(hb.free_mem_gb, 0.1) + s.update_size_mb / max(
            hb.upload_bw_mbps * 10.0, 0.1
        )
        eps = 0.01
        return (alpha * bq + beta * debt + gamma * fresh + delta * match) / (
            cost + eps
        )

    def schedule(
        self, round_idx: int, candidates: list[Client], rng=None
    ) -> list[Lease]:
        regime = self._detect_regime()
        rng_np = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()

        viable = [
            c
            for c in candidates
            if c.heartbeat is not None and c.heartbeat.is_viable()
        ]
        viable = viable[: self.clients_per_round]

        assignment: dict[int, list[Slice]] = defaultdict(list)
        slice_assign_count: dict[tuple[int, int], int] = defaultdict(int)
        client_budget = {c.cid: self._slice_budget(c, regime) for c in viable}

        pairs = []
        for c in viable:
            for s in self.slice_set:
                if self.feasibility.check_memory(c, assignment[c.cid] + [s]):
                    pairs.append((self._utility(c, s), c, s))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for _, c, s in pairs:
            if len(assignment[c.cid]) >= client_budget[c.cid]:
                continue
            if slice_assign_count[s.sid] >= self._redundancy_cap(s):
                continue
            if not self.feasibility.check_memory(c, assignment[c.cid] + [s]):
                continue
            assignment[c.cid].append(s)
            slice_assign_count[s.sid] += 1

        leases = []
        for c in viable:
            if assignment[c.cid]:
                leases.append(
                    Lease(
                        client_id=c.cid,
                        slices=assignment[c.cid],
                        local_steps=self.local_steps,
                        micro_batch=self.micro_batch,
                        deadline_sec=max(
                            60.0, c.heartbeat.online_sec - 30.0 if c.heartbeat else 120.0
                        ),
                    )
                )
        return leases

    def post_round_update(self, round_idx: int, executed_leases: list[Lease]) -> None:
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for lease in executed_leases:
            if lease.completed:
                for s in lease.slices:
                    counts[s.sid] += 1
        for s in self.slice_set:
            self._coverage_history[s.sid].append(counts.get(s.sid, 0))
