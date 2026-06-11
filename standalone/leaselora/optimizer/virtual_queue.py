from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class VirtualQueue:

    target: float = 0.0
    backlog: float = 0.0
    max_backlog: float = 5.0
    decay: float = 0.9

    def update(self, service: float) -> float:
        raw = self.backlog + self.target - service
        self.backlog = min(self.max_backlog, max(0.0, raw) * self.decay)
        return self.backlog

    def drift(self) -> float:
        return self.backlog

    def reset(self) -> None:
        self.backlog = 0.0

class CoverageQueue:

    def __init__(
        self,
        slice_ids: list[tuple[int, int]],
        coverage_target: float = 0.8,
        max_backlog: float = 5.0,
        decay: float = 0.9,
    ):
        self.queues: dict[tuple[int, int], VirtualQueue] = {
            sid: VirtualQueue(
                target=coverage_target, max_backlog=max_backlog, decay=decay,
            )
            for sid in slice_ids
        }

    def update(self, accepted_counts: dict[tuple[int, int], int]) -> None:
        for sid, q in self.queues.items():
            service = float(accepted_counts.get(sid, 0))
            q.update(service)

    def get_deficit(self, sid: tuple[int, int]) -> float:
        return self.queues[sid].backlog

    def total_deficit(self) -> float:
        return sum(q.backlog for q in self.queues.values())

    def max_deficit(self) -> float:
        return max(q.backlog for q in self.queues.values()) if self.queues else 0.0

    def mean_deficit(self) -> float:
        if not self.queues:
            return 0.0
        return self.total_deficit() / len(self.queues)

@dataclass
class FairnessQueue:

    participation_target: float = 0.4
    backlog: float = 0.0
    max_backlog: float = 5.0
    decay: float = 0.9

    def update(self, low_resource_participation_rate: float) -> float:
        raw = self.backlog + self.participation_target - low_resource_participation_rate
        self.backlog = min(self.max_backlog, max(0.0, raw) * self.decay)
        return self.backlog

    @property
    def deficit(self) -> float:
        return self.backlog

    def reset(self) -> None:
        self.backlog = 0.0
