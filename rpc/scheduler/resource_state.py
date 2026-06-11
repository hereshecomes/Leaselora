from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class ResourceTier:
    budget: float
    compute_slowdown: float = 1.0
    uplink_mbps: float = 50.0
    throughput: float = 500.0
    memory_budget_gb: float = 24.0

DEFAULT_TIERS = {
    0.3: ResourceTier(budget=0.3, compute_slowdown=3.5, uplink_mbps=5, throughput=150, memory_budget_gb=12),
    0.5: ResourceTier(budget=0.5, compute_slowdown=2.0, uplink_mbps=10, throughput=300, memory_budget_gb=16),
    0.7: ResourceTier(budget=0.7, compute_slowdown=1.4, uplink_mbps=20, throughput=450, memory_budget_gb=20),
    1.0: ResourceTier(budget=1.0, compute_slowdown=1.0, uplink_mbps=50, throughput=500, memory_budget_gb=24),
}

@dataclass
class ClientResourceState:
    client_id: int
    budget: float = 1.0
    tier: ResourceTier = field(default_factory=lambda: DEFAULT_TIERS[1.0])
    online: bool = True
    last_train_time_sec: float = 0.0
    last_upload_time_sec: float = 0.0
    last_peak_memory_gb: float = 0.0

def assign_client_budgets(
    num_clients: int,
    budget_dist: list[float] | None = None,
    seed: int = 42,
) -> dict[int, float]:
    import numpy as np
    if budget_dist is None:
        budget_dist = [0.3, 0.5, 0.7, 1.0]

    rng = np.random.default_rng(seed)
    budgets = {}
    for cid in range(num_clients):
        budgets[cid] = budget_dist[cid % len(budget_dist)]
    return budgets

def build_client_resource_states(
    num_clients: int,
    budget_assignments: dict[int, float],
    tiers_config: list[dict] | None = None,
) -> dict[int, ClientResourceState]:
    tiers = {}
    if tiers_config:
        for t in tiers_config:
            tier = ResourceTier(**t)
            tiers[tier.budget] = tier
    else:
        tiers = DEFAULT_TIERS

    states = {}
    for cid in range(num_clients):
        budget = budget_assignments.get(cid, 1.0)
        closest_budget = min(tiers.keys(), key=lambda b: abs(b - budget))
        tier = tiers[closest_budget]
        states[cid] = ClientResourceState(client_id=cid, budget=budget, tier=tier)

    return states
