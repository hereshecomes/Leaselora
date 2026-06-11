from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .resource_state import ClientResourceState

@dataclass
class GuardResult:
    feasible: bool = True
    fail_reason: str | None = None
    original_num_slices: int = 0
    final_num_slices: int = 0
    original_local_steps: int = 0
    final_local_steps: int = 0
    original_micro_batch: int = 0
    final_micro_batch: int = 0
    shrink_count: int = 0
    rejected: bool = False

@dataclass
class LeaseProposal:
    client_id: int
    slice_ids: list[str]
    param_names: list[str]
    local_steps: int = 24
    micro_batch_size: int = 16
    deadline_seconds: float = 120.0
    per_slice_memory_gb: float = 0.22
    per_slice_payload_bytes: int = 500_000
    per_step_time_sec: float = 0.5

class FeasibilityGuard:

    def __init__(
        self,
        shrink_policy: list[str] | None = None,
        min_slices: int = 1,
        min_local_steps: int = 4,
        min_micro_batch: int = 4,
        bandwidth_predictor=None,
        online_threshold: float = 0.5,
        guard_mode: str = "default",
    ):
        self.shrink_policy = shrink_policy or ["slices", "local_steps", "micro_batch"]
        self.min_slices = min_slices
        self.min_local_steps = min_local_steps
        self.min_micro_batch = min_micro_batch
        self.bandwidth_predictor = bandwidth_predictor
        self.online_threshold = online_threshold
        self.guard_mode = guard_mode
        self.current_round: int = 0

    def check(
        self,
        proposal: LeaseProposal,
        resource: ClientResourceState,
    ) -> tuple[LeaseProposal, GuardResult]:
        result = GuardResult(
            original_num_slices=len(proposal.slice_ids),
            original_local_steps=proposal.local_steps,
            original_micro_batch=proposal.micro_batch_size,
        )

        current = LeaseProposal(
            client_id=proposal.client_id,
            slice_ids=list(proposal.slice_ids),
            param_names=list(proposal.param_names),
            local_steps=proposal.local_steps,
            micro_batch_size=proposal.micro_batch_size,
            deadline_seconds=proposal.deadline_seconds,
            per_slice_memory_gb=proposal.per_slice_memory_gb,
            per_slice_payload_bytes=proposal.per_slice_payload_bytes,
            per_step_time_sec=proposal.per_step_time_sec,
        )

        for action in self.shrink_policy:
            if self._is_feasible(current, resource):
                break
            result.shrink_count += 1
            if action == "slices" and len(current.slice_ids) > self.min_slices:
                current.slice_ids = current.slice_ids[:max(len(current.slice_ids) // 2, self.min_slices)]
            elif action == "local_steps" and current.local_steps > self.min_local_steps:
                current.local_steps = max(current.local_steps // 2, self.min_local_steps)
            elif action == "micro_batch" and current.micro_batch_size > self.min_micro_batch:
                current.micro_batch_size = max(current.micro_batch_size // 2, self.min_micro_batch)

        if self._is_feasible(current, resource):
            result.feasible = True
            result.final_num_slices = len(current.slice_ids)
            result.final_local_steps = current.local_steps
            result.final_micro_batch = current.micro_batch_size
        else:
            result.feasible = False
            result.rejected = True
            result.fail_reason = self._get_fail_reason(current, resource)
            result.final_num_slices = len(current.slice_ids)
            result.final_local_steps = current.local_steps
            result.final_micro_batch = current.micro_batch_size

        return current, result

    def _is_feasible(self, proposal: LeaseProposal, resource: ClientResourceState) -> bool:
        if self.guard_mode == "none":
            return True

        if self.guard_mode == "full":
            return (
                self._online_guard(resource)
                and self._memory_guard(proposal, resource)
                and self._upload_guard(proposal, resource)
                and self._time_guard(proposal, resource)
            )

        if self.guard_mode == "oracle":
            return (
                self._memory_guard(proposal, resource)
                and self._oracle_time_guard(proposal, resource)
            )

        return (
            self._online_guard(resource)
            and self._memory_guard(proposal, resource)
        )

    def _memory_guard(self, proposal: LeaseProposal, resource: ClientResourceState) -> bool:
        estimated_mem = len(proposal.slice_ids) * proposal.per_slice_memory_gb
        return estimated_mem <= resource.tier.memory_budget_gb

    def _upload_guard(self, proposal: LeaseProposal, resource: ClientResourceState) -> bool:
        payload_bytes = len(proposal.slice_ids) * proposal.per_slice_payload_bytes
        uplink_mbps = self._get_effective_uplink(resource)
        uplink_bytes_per_sec = uplink_mbps * 1e6 / 8
        upload_time = payload_bytes / max(uplink_bytes_per_sec, 1)
        rtt_sec = self._get_effective_rtt(resource) / 1000.0
        retry_penalty = self._get_retry_penalty(resource)
        total_upload = upload_time + rtt_sec + retry_penalty
        upload_budget = proposal.deadline_seconds * 0.3
        return total_upload <= upload_budget

    def _time_guard(self, proposal: LeaseProposal, resource: ClientResourceState) -> bool:
        compute_time = (
            proposal.local_steps * proposal.per_step_time_sec * resource.tier.compute_slowdown
        )
        payload_bytes = len(proposal.slice_ids) * proposal.per_slice_payload_bytes
        uplink_mbps = self._get_effective_uplink(resource)
        uplink_bytes_per_sec = uplink_mbps * 1e6 / 8
        upload_time = payload_bytes / max(uplink_bytes_per_sec, 1)
        rtt_sec = self._get_effective_rtt(resource) / 1000.0
        retry_penalty = self._get_retry_penalty(resource)
        total_time = compute_time + upload_time + rtt_sec + retry_penalty
        return total_time <= proposal.deadline_seconds

    def _online_guard(self, resource: ClientResourceState) -> bool:
        return True

    def _oracle_time_guard(self, proposal: LeaseProposal, resource: ClientResourceState) -> bool:
        compute_time = (
            proposal.local_steps * proposal.per_step_time_sec * resource.tier.compute_slowdown
        )
        payload_bytes = len(proposal.slice_ids) * proposal.per_slice_payload_bytes
        uplink_bytes_per_sec = resource.tier.uplink_mbps * 1e6 / 8
        upload_time = payload_bytes / max(uplink_bytes_per_sec, 1)
        total_time = compute_time + upload_time
        return total_time <= proposal.deadline_seconds

    def _get_effective_uplink(self, resource: ClientResourceState) -> float:
        if self.bandwidth_predictor:
            prediction = self.bandwidth_predictor.predict(resource.client_id, self.current_round)
            return prediction.predicted_uplink_mbps
        return resource.tier.uplink_mbps

    def _get_effective_rtt(self, resource: ClientResourceState) -> float:
        if self.bandwidth_predictor:
            prediction = self.bandwidth_predictor.predict(resource.client_id, self.current_round)
            return prediction.predicted_rtt_ms
        return 50.0

    def _get_retry_penalty(self, resource: ClientResourceState) -> float:
        if self.bandwidth_predictor:
            return self.bandwidth_predictor.get_retry_penalty_sec(resource.client_id)
        return 0.0

    def _get_fail_reason(self, proposal: LeaseProposal, resource: ClientResourceState) -> str:
        reasons = []
        if not self._online_guard(resource):
            reasons.append("offline_predicted")
        if not self._memory_guard(proposal, resource):
            reasons.append("memory_exceeded")
        if not self._upload_guard(proposal, resource):
            reasons.append("upload_exceeded")
        if not self._time_guard(proposal, resource):
            reasons.append("time_exceeded")
        return "|".join(reasons) if reasons else "unknown"
