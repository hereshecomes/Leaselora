from __future__ import annotations

import numpy as np

def dirichlet_partition(dataset, num_clients: int, alpha: float = 0.5,
                        label_column: str = "labels", seed: int = 42) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    labels = np.array(dataset[label_column])
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

def iid_partition(dataset, num_clients: int, seed: int = 42) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [s.tolist() for s in splits]
