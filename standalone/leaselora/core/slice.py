from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

@dataclass
class Slice:

    group: int
    rank_block: int
    memory_gb: float = 0.22
    update_size_mb: float = 18.0
    intrinsic_importance: float = 1.0
    difficulty: float = 1.0

    quality: float = 0.5
    coverage_count: int = 0
    progress: float = 0.0
    fairness_debt: float = 0.0

    @property
    def sid(self) -> tuple[int, int]:
        return (self.group, self.rank_block)

    @property
    def need(self) -> float:
        return 1.0 - self.progress

    def blended_quality(self) -> float:
        return 0.45 * self.quality + 0.55 * self.need * self.intrinsic_importance

    def __hash__(self) -> int:
        return hash(self.sid)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Slice):
            return NotImplemented
        return self.sid == other.sid

@dataclass
class SliceSet:

    num_groups: int
    num_rank_blocks: int
    per_slice_mem_gb: float = 0.22
    per_slice_update_mb: float = 18.0

    _slices: dict[tuple[int, int], Slice] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._slices = {}
        for g in range(self.num_groups):
            for r in range(self.num_rank_blocks):
                s = Slice(
                    group=g,
                    rank_block=r,
                    memory_gb=self.per_slice_mem_gb,
                    update_size_mb=self.per_slice_update_mb,
                )
                self._slices[(g, r)] = s

    def __len__(self) -> int:
        return len(self._slices)

    def __iter__(self) -> Iterator[Slice]:
        return iter(self._slices.values())

    def __getitem__(self, sid: tuple[int, int]) -> Slice:
        return self._slices[sid]

    def get(self, group: int, rank_block: int) -> Slice:
        return self._slices[(group, rank_block)]

    @property
    def total_memory_gb(self) -> float:
        return sum(s.memory_gb for s in self)

    def reset_round_stats(self) -> None:
        for s in self:
            s.coverage_count = 0
