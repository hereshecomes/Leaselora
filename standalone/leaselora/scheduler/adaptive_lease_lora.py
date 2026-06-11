from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from ..core.client import Client
from ..core.lease import Lease
from ..core.slice import Slice, SliceSet
from ..aggregator.variance_tracker import VarianceTracker
from ..feasibility.deterministic import DeterministicFeasibility
from ..feasibility.chance_constrained import ChanceConstrainedFeasibility
from ..optimizer.drift_plus_penalty import DriftPlusPenaltyScheduler
from ..optimizer.granularity import GranularityIndex
from ..optimizer.virtual_queue import CoverageQueue, FairnessQueue
from .base import BaseScheduler

class AdaptiveLeaseLoRA(BaseScheduler):
    name = "Adaptive LeaseLoRA"

    def __init__(
        self,
        slice_set: SliceSet,
        clients_per_round: int = 16,
        local_steps_base: int = 24,
        local_steps_delta: int = 4,
        micro_batch: int = 8,
        coverage_target: float = 0.8,
        coverage_window: int = 10,
        participation_target: float = 0.4,
        V: float = 50.0,
        r_base: int = 1,
        r_max: int = 2,
        use_chance_constraint: bool = True,
        target_delta: float = 0.05,
        disable_dpp: bool = False,
        disable_focus: bool = False,
        disable_variance_redundancy: bool = False,
        disable_reservation: bool = False,
        disable_adaptive_steps: bool = False,
        **kwargs,
    ):
        super().__init__(slice_set, **kwargs)
        self.clients_per_round = clients_per_round
        self.local_steps_base = local_steps_base
        self.local_steps_delta = local_steps_delta
        self.micro_batch = micro_batch
        self.r_base = r_base if r_base is not None else 1
        self.r_max = r_max if r_max is not None else 2

        self.disable_dpp = disable_dpp
        self.disable_focus = disable_focus
        self.disable_variance_redundancy = disable_variance_redundancy
        self.disable_reservation = disable_reservation
        self.disable_adaptive_steps = disable_adaptive_steps
        self.max_slices_per_client: Optional[int] = kwargs.pop("max_slices_per_client", None)

        self.granularity = GranularityIndex()
        self.dpp = DriftPlusPenaltyScheduler(V=V, granularity=self.granularity)
        self.dpp.init_queues(slice_set, coverage_target, participation_target)

        self.variance_tracker = VarianceTracker()

        if use_chance_constraint:
            self.feasibility = ChanceConstrainedFeasibility(target_delta=target_delta)
        else:
            self.feasibility = DeterministicFeasibility()

        self._coverage_window = coverage_window
        self._client_map: dict[int, Client] = {}

    def _compute_granularity(self, candidates: list[Client]) -> float:
        n_slices = len(self.slice_set)
        avg_mem = sum(s.memory_gb for s in self.slice_set) / max(n_slices, 1)
        if candidates:
            mems = [
                c.heartbeat.free_mem_gb
                for c in candidates
                if c.heartbeat is not None
            ]
            dispersion = (max(mems) - min(mems)) / max(max(mems), 0.01) if mems else 0.5
        else:
            dispersion = 0.5
        return self.granularity.compute(n_slices, avg_mem, dispersion)

    def _reservation_pass(
        self,
        low_resource_clients: list[Client],
        candidate_slices: list[Slice],
        g_index: float,
        assignment: dict[int, list[Slice]],
        slice_assign_count: dict[tuple[int, int], int],
        rng_np: np.random.Generator,
    ) -> None:
        if self.disable_reservation:
            return
        lam = self.granularity.reservation_ratio(g_index)
        n_reserve = max(1, int(np.ceil(lam * len(low_resource_clients))))
        reserved = low_resource_clients[:n_reserve]

        for c in reserved:
            best_score = -float("inf")
            best_slice = None
            for s in candidate_slices:
                if not self._check_feasible(c, assignment[c.cid] + [s]):
                    continue
                if self.disable_dpp:
                    score = s.need * c.get_affinity(s.group)
                else:
                    score = self.dpp.score_assignment(c, s, g_index)
                if score > best_score:
                    best_score = score
                    best_slice = s
            if best_slice is not None:
                assignment[c.cid].append(best_slice)
                slice_assign_count[best_slice.sid] += 1

    def _check_feasible(self, client: Client, slices: list[Slice]) -> bool:
        if isinstance(self.feasibility, ChanceConstrainedFeasibility):
            return self.feasibility.is_feasible(
                client, slices, self.local_steps_base, self.micro_batch
            )
        return self.feasibility.is_feasible(
            client, slices, self.local_steps_base, self.micro_batch
        )

    def schedule(
        self, round_idx: int, candidates: list[Client], rng=None
    ) -> list[Lease]:
        rng_np = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()

        viable = [
            c
            for c in candidates
            if c.heartbeat is not None and c.heartbeat.is_viable()
        ][: self.clients_per_round]

        if not viable:
            return []

        for c in viable:
            self._client_map[c.cid] = c

        g_index = self._compute_granularity(viable)

        client_budget = {
            c.cid: self.dpp.compute_slice_budget(c, g_index) for c in viable
        }
        if self.max_slices_per_client is not None:
            client_budget = {
                k: min(v, self.max_slices_per_client) for k, v in client_budget.items()
            }
        total_round_budget = sum(client_budget.values())

        if self.disable_focus:
            candidate_slices = list(self.slice_set)
        else:
            candidate_slices = self.dpp.select_focus_set(
                self.slice_set, total_round_budget, g_index
            )

        assignment: dict[int, list[Slice]] = defaultdict(list)
        slice_assign_count: dict[tuple[int, int], int] = defaultdict(int)

        low_res = [c for c in viable if c.is_low_resource]
        if low_res:
            self._reservation_pass(
                low_res, candidate_slices, g_index, assignment,
                slice_assign_count, rng_np,
            )

        pairs = []
        for c in viable:
            for s in candidate_slices:
                if self.disable_dpp:
                    score = s.need * c.get_affinity(s.group) + 0.2 * c.freshness
                else:
                    score = self.dpp.score_assignment(c, s, g_index)
                pairs.append((score, c, s))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for score, c, s in pairs:
            if len(assignment[c.cid]) >= client_budget[c.cid]:
                continue

            if self.disable_variance_redundancy:
                r_cap = self.r_base
            else:
                cov_debt = 0.0
                if self.dpp.coverage_queue is not None:
                    cov_debt = self.dpp.coverage_queue.get_deficit(s.sid)
                r_cap = self.variance_tracker.compute_redundancy(
                    s.sid, s.progress, self.r_base, self.r_max,
                    coverage_debt=cov_debt,
                )
            if slice_assign_count[s.sid] >= r_cap:
                continue
            if s in assignment[c.cid]:
                continue
            if not self._check_feasible(c, assignment[c.cid] + [s]):
                continue
            assignment[c.cid].append(s)
            slice_assign_count[s.sid] += 1

        leases = []
        for c in viable:
            slices = assignment[c.cid]
            if not slices:
                continue

            if self.disable_adaptive_steps:
                local_steps = self.local_steps_base
            else:
                local_steps = self.dpp.compute_local_steps(
                    c, slices, g_index,
                    e_base=self.local_steps_base,
                    step_delta=self.local_steps_delta,
                )

            deadline = (
                c.heartbeat.online_sec - 10.0 if c.heartbeat else 120.0
            )
            leases.append(
                Lease(
                    client_id=c.cid,
                    slices=slices,
                    local_steps=local_steps,
                    micro_batch=self.micro_batch,
                    deadline_sec=max(30.0, deadline),
                )
            )
        return leases

    def post_round_update(self, round_idx: int, executed_leases: list[Lease]) -> None:
        accepted_counts: dict[tuple[int, int], int] = defaultdict(int)
        low_res_accepted = 0
        low_res_total = 0

        for lease in executed_leases:
            client = self._client_map.get(lease.client_id)
            is_low = client.is_low_resource if client else (lease.num_slices <= 2)
            if is_low:
                low_res_total += 1
            if lease.completed:
                for s in lease.slices:
                    accepted_counts[s.sid] += 1
                if is_low:
                    low_res_accepted += 1
                if lease.updates is not None:
                    for sid, delta in lease.updates.items():
                        norm = (
                            float(np.linalg.norm(delta))
                            if hasattr(delta, "__len__")
                            else float(delta)
                        )
                        self.variance_tracker.update(sid, norm)

        low_rate = low_res_accepted / max(low_res_total, 1)

        if not self.disable_dpp:
            self.dpp.update_queues(accepted_counts, low_rate)
