from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import numpy as np

from .feasibility_guard import FeasibilityGuard, LeaseProposal, GuardResult
from .resource_state import ClientResourceState
from .feedback_state import FeedbackState

logger = logging.getLogger("fl_testbed.scheduler")

class LeasePlanner:

    def __init__(
        self,
        slice_registry: OrderedDict[str, list[str]],
        method: str = "leaselora",
        default_local_steps: int = 24,
        default_micro_batch: int = 16,
        default_deadline_sec: float = 120.0,
        deadline_factor: float = 1.2,
        warmup_full_rounds: int = 5,
        per_slice_memory_gb: float = 0.22,
        per_slice_payload_bytes: int = 500_000,
        per_step_time_sec: float = 0.5,
        guard: FeasibilityGuard | None = None,
    ):
        self.slice_registry = slice_registry
        self.method = method
        self.default_local_steps = default_local_steps
        self.default_micro_batch = default_micro_batch
        self.default_deadline_sec = default_deadline_sec
        self.deadline_factor = deadline_factor
        self.warmup_full_rounds = warmup_full_rounds
        self.per_slice_memory_gb = per_slice_memory_gb
        self.per_slice_payload_bytes = per_slice_payload_bytes
        self.per_step_time_sec = per_step_time_sec
        self.guard = guard or FeasibilityGuard()
        self.all_slice_ids = list(slice_registry.keys())
        self.num_slices = len(self.all_slice_ids)

    def plan_round(
        self,
        round_id: int,
        selected_clients: list[int],
        client_resources: dict[int, ClientResourceState],
        feedback: FeedbackState | None = None,
    ) -> list[dict]:
        is_warmup = round_id < self.warmup_full_rounds
        leases = []

        for cid in selected_clients:
            resource = client_resources.get(cid)
            if resource is None:
                continue

            if is_warmup:
                slice_ids = list(self.all_slice_ids)
            else:
                slice_ids = self._plan_leaselora(cid, resource, feedback)

            param_names = []
            for sid in slice_ids:
                param_names.extend(self.slice_registry.get(sid, []))

            if not is_warmup:
                local_steps = self._compute_local_steps(resource)
            else:
                local_steps = self.default_local_steps

            deadline = self.default_deadline_sec * self.deadline_factor

            proposal = LeaseProposal(
                client_id=cid,
                slice_ids=slice_ids,
                param_names=param_names,
                local_steps=local_steps,
                micro_batch_size=self.default_micro_batch,
                deadline_seconds=deadline,
                per_slice_memory_gb=self.per_slice_memory_gb,
                per_slice_payload_bytes=self.per_slice_payload_bytes,
                per_step_time_sec=self.per_step_time_sec,
            )

            final_proposal, guard_result = self.guard.check(proposal, resource)

            if guard_result.rejected:
                logger.warning(f"Lease rejected for client {cid}: {guard_result.fail_reason}")
                continue

            final_param_names = []
            for sid in final_proposal.slice_ids:
                final_param_names.extend(self.slice_registry.get(sid, []))

            lease = {
                "lease_id": f"r{round_id}_c{cid}",
                "client_id": cid,
                "slice_ids": final_proposal.slice_ids,
                "param_names": final_param_names,
                "local_steps": final_proposal.local_steps,
                "micro_batch_size": final_proposal.micro_batch_size,
                "deadline_seconds": final_proposal.deadline_seconds,
                "budget": resource.budget,
                "guard_pass": guard_result.feasible,
                "guard_fail_reason": guard_result.fail_reason,
                "guard_shrink_count": guard_result.shrink_count,
                "original_num_slices": guard_result.original_num_slices,
                "final_num_slices": guard_result.final_num_slices,
            }
            leases.append(lease)

        return leases

    def _compute_local_steps(self, resource: ClientResourceState) -> int:
        e_base = self.default_local_steps
        capability = min(1.0, resource.tier.throughput / 500.0)
        step_delta = 4
        e_min = max(step_delta, e_base // 3)
        e_scaled = max(e_min, int(e_base * capability))
        e_scaled = max(e_min, (e_scaled // step_delta) * step_delta or step_delta)
        return e_scaled

    def _plan_leaselora(
        self, cid: int, resource: ClientResourceState, feedback: FeedbackState | None
    ) -> list[str]:
        budget = resource.budget
        gated_slices = [s for s in self.all_slice_ids if s != "slice_head"]
        max_slices = max(1, int(len(gated_slices) * budget))

        if feedback is None:
            scores = np.ones(len(gated_slices))
        else:
            scores = feedback.get_slice_scores(gated_slices)

        sorted_indices = np.argsort(-scores)
        selected = [gated_slices[i] for i in sorted_indices[:max_slices]]

        if "slice_head" in self.all_slice_ids:
            selected.append("slice_head")
        return selected

