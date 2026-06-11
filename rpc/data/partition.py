from __future__ import annotations

import json
from pathlib import Path

import numpy as np

def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        class_splits = np.split(class_idx, splits)
        for i, split in enumerate(class_splits):
            client_indices[i].extend(split.tolist())

    for i in range(num_clients):
        rng.shuffle(np.array(client_indices[i]))

    return client_indices

def iid_partition(
    num_samples: int,
    num_clients: int,
    seed: int = 42,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [s.tolist() for s in splits]

def save_partitions(
    client_indices: list[list[int]],
    out_dir: str | Path,
    dataset_name: str = "",
    alpha: float = 0.5,
    seed: int = 42,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": dataset_name,
        "num_clients": len(client_indices),
        "alpha": alpha,
        "seed": seed,
        "client_sizes": [len(idx) for idx in client_indices],
        "total_samples": sum(len(idx) for idx in client_indices),
    }

    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for cid, indices in enumerate(client_indices):
        fname = f"client_{cid:03d}.json"
        with open(out_path / fname, "w") as f:
            json.dump({"client_id": cid, "indices": indices, "num_samples": len(indices)}, f)

def load_partition(partition_dir: str | Path, client_id: int) -> list[int]:
    path = Path(partition_dir) / f"client_{client_id:03d}.json"
    with open(path) as f:
        data = json.load(f)
    return data["indices"]

def load_metadata(partition_dir: str | Path) -> dict:
    path = Path(partition_dir) / "metadata.json"
    with open(path) as f:
        return json.load(f)
