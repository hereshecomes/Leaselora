# LeaseLoRA: Adaptive Slice-Level Lease Scheduling for Federated PEFT

This repository contains the source code for LeaseLoRA, an adaptive federated parameter-efficient fine-tuning (PEFT) framework that introduces slice-level lease scheduling to handle device heterogeneity in federated learning.

## Project Structure

```
release/
├── standalone/          # Single-machine simulation engine
│   ├── leaselora/       # Core scheduling library
│   │   ├── core/        # Data structures (Client, Heartbeat, Lease, Slice)
│   │   ├── scheduler/   # All scheduling strategies (14 methods)
│   │   ├── optimizer/   # DPP scheduler, virtual queues, granularity index
│   │   ├── aggregator/  # Slice-wise aggregation & variance tracking
│   │   ├── feasibility/ # Deterministic & chance-constrained guards
│   │   └── utils/       # Time model, metrics collector
│   ├── data/            # Dataset loaders (HAR, MovieLens, Amazon, partitioning)
│   └── models/          # Model definitions (HAR Transformer, SASRec, BERT4Rec)
│
├── rpc/                 # Distributed multi-node RPC testbed
│   ├── coordinator.py   # Central coordinator (FL loop, aggregation, evaluation)
│   ├── worker.py        # GPU worker (local PEFT training, delta upload)
│   ├── rpc/             # HTTP communication layer (FastAPI server + client)
│   ├── scheduler/       # Lease planner, feasibility guard, feedback state
│   ├── runtime/         # Round manager, checkpoint, client sampler, evaluator
│   ├── peft_exec/       # Model factory, LoRA utilities, local trainer
│   ├── network/         # Network emulation (tc, bandwidth predictor, traces)
│   ├── data/            # Data partitioning & dataset management
│   └── configs/         # YAML experiment configurations
│
└── requirements.txt     # Python dependencies
```

## Overview

### Standalone (Single-Machine Simulation)

The `standalone/` folder implements the complete LeaseLoRA scheduling algorithm as a reusable Python library. It is designed for large-scale FL simulation experiments where hundreds of heterogeneous clients are modeled within a single process.

**Key components:**

- **AdaptiveLeaseLoRA** (`scheduler/adaptive_lease_lora.py`): The full proposed method integrating Lyapunov drift-plus-penalty scoring, virtual queues for coverage/fairness, continuous granularity index, variance-aware redundancy, focus scheduling, low-resource reservation, and adaptive local step budgets.

- **LeaseLoRA** (`scheduler/lease_lora.py`): The non-adaptive baseline with hand-tuned utility scoring.

- **DriftPlusPenaltyScheduler** (`optimizer/drift_plus_penalty.py`): Theory-grounded online optimization using Lyapunov framework for constrained scheduling.

- **CoverageQueue / FairnessQueue** (`optimizer/virtual_queue.py`): Virtual queues with decay for long-term constraint enforcement.

- **ChanceConstrainedFeasibility** (`feasibility/chance_constrained.py`): Probabilistic feasibility checking with adaptive guards satisfying Pr(violation) ≤ δ.

- **VarianceTracker** (`aggregator/variance_tracker.py`): Welford's online variance estimation for adaptive redundancy control.

- **AdaptiveCore**: Ablation variant with focus/reservation/variance-redundancy disabled.

### RPC (Distributed Multi-Node Testbed)

The `rpc/` folder implements a real coordinator-worker federated PEFT system communicating over HTTP (FastAPI + Requests). It is designed for physical deployment on multiple GPU servers.

**Architecture:**

```
Coordinator (node2:29580)
    ├── Plans lease assignments per round
    ├── Serves global LoRA adapter via HTTP
    ├── Receives partial deltas from workers
    └── Aggregates updates & evaluates

Workers (node0, node1, node2 × 2 GPUs each)
    ├── Poll coordinator for tasks
    ├── Download global adapter
    ├── Execute local PEFT training on leased slices
    └── Upload serialized delta
```

**Core RPC endpoints:**
- `POST /register` — Worker registration
- `POST /heartbeat` — Periodic health reporting
- `GET /task` — Task polling
- `GET /global_adapter` — Download current global adapter state
- `POST /upload_update` — Upload training delta

**Method alignment with standalone:**

| Component | Standalone | RPC |
|-----------|-----------|-----|
| Lease planning | `AdaptiveLeaseLoRA.schedule()` | `LeasePlanner.plan_round()` |
| Feasibility check | `ChanceConstrainedFeasibility.is_feasible()` | `FeasibilityGuard.check()` |
| Local training | Simulated progress model | `local_trainer.local_train()` (real GPU) |
| Aggregation | `SliceAggregator.aggregate()` | `aggregate_partial_deltas()` |
| Feedback/coverage | Virtual queues in DPP scheduler | `FeedbackState.update_after_round()` |
| Adaptive local steps | `DPP.compute_local_steps()` | `LeasePlanner._compute_local_steps()` |

## Running

### Standalone Usage

```python
from standalone.leaselora.core.slice import SliceSet
from standalone.leaselora.scheduler.adaptive_lease_lora import AdaptiveLeaseLoRA

slice_set = SliceSet(num_groups=6, num_rank_blocks=8)
scheduler = AdaptiveLeaseLoRA(slice_set, clients_per_round=16)

leases = scheduler.schedule(round_idx=0, candidates=client_list)
scheduler.post_round_update(round_idx=0, executed_leases=leases)
```

### RPC Testbed

**Start coordinator:**
```bash
python -m rpc --config rpc/configs/debug_local.yaml \
    --host 0.0.0.0 --port 29580
```

**Start worker:**
```bash
python -m rpc.worker --config rpc/configs/debug_local.yaml \
    --worker_id node0_gpu0 \
    --coordinator http://localhost:29580 \
    --cuda_device 0
```

## Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- PEFT ≥ 0.4
- FastAPI + Uvicorn (RPC testbed)
- NumPy, SciPy

## Supported Tasks & Datasets

| Domain | Dataset | Backbone |
|--------|---------|----------|
| NLP | SST-2, AG News | DistilBERT, BERT, RoBERTa |
| CV | CIFAR-100, Tiny-ImageNet | ViT-Base, ViT-Small |
| HAR | WISDM, UCI HAR | Transformer encoder |
| RecSys | MovieLens-1M, Amazon Beauty | SASRec, BERT4Rec |

## Key Design Principles

1. **Slice-level scheduling**: LoRA parameters are partitioned into adapter slices (by transformer layer). Each client trains only a subset based on its resource budget.

2. **Drift-plus-penalty optimization**: Scheduling decisions maximize a Lyapunov objective balancing model progress, communication cost, coverage deficit, and low-resource participation.

3. **Chance-constrained feasibility**: Adaptive guards with probabilistic guarantees replace fixed safety margins, reducing false rejections for well-characterized clients.

4. **Variance-aware redundancy**: Redundancy (number of clients assigned to same slice) adapts based on observed update variance — high-variance slices get replicated for stability.

5. **Continuous granularity index**: Smooth regime transitions replace hard-coded thresholds, eliminating discontinuities in scheduling behavior.
