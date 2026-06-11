from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..core.client import Client
from ..core.lease import Lease
from ..core.slice import SliceSet

class BaseScheduler(ABC):

    name: str = "base"

    def __init__(self, slice_set: SliceSet, **kwargs):
        self.slice_set = slice_set

    @abstractmethod
    def schedule(
        self,
        round_idx: int,
        candidates: list[Client],
        rng: Optional[object] = None,
    ) -> list[Lease]:
        ...

    @abstractmethod
    def post_round_update(
        self,
        round_idx: int,
        executed_leases: list[Lease],
    ) -> None:
        ...
