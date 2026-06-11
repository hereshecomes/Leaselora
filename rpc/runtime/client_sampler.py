from __future__ import annotations

import numpy as np

from ..scheduler.resource_state import ClientResourceState

class ClientSampler:

    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        participation_rate: float | None = None,
        seed: int = 42,
        enable_weak_client_reservation: bool = False,
        weak_client_threshold: float = 0.5,
    ):
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.participation_rate = participation_rate
        self.seed = seed
        self.enable_weak_reservation = enable_weak_client_reservation
        self.weak_threshold = weak_client_threshold
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        round_id: int,
        available_client_ids: list[int] | None = None,
        client_resources: dict[int, ClientResourceState] | None = None,
    ) -> list[int]:
        if available_client_ids is None:
            available_client_ids = list(range(self.num_clients))

        k = self.clients_per_round
        if self.participation_rate is not None:
            k = max(1, int(len(available_client_ids) * self.participation_rate))

        k = min(k, len(available_client_ids))

        if self.enable_weak_reservation and client_resources:
            weak_ids = [
                cid for cid in available_client_ids
                if client_resources.get(cid, ClientResourceState(cid)).budget <= self.weak_threshold
            ]
            strong_ids = [cid for cid in available_client_ids if cid not in weak_ids]

            num_weak = max(1, k // 4) if weak_ids else 0
            num_strong = k - num_weak

            selected_weak = [int(x) for x in self.rng.choice(
                weak_ids, size=min(num_weak, len(weak_ids)), replace=False
            )] if weak_ids else []
            selected_strong = [int(x) for x in self.rng.choice(
                strong_ids, size=min(num_strong, len(strong_ids)), replace=False
            )] if strong_ids else []

            return sorted(selected_weak + selected_strong)

        selected = [int(x) for x in self.rng.choice(available_client_ids, size=k, replace=False)]
        return sorted(selected)
