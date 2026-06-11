from __future__ import annotations

import os
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
CACHE_DIR = Path(os.environ.get("ML_CACHE", Path.home() / ".cache" / "movielens"))

def download_ml100k(cache_dir: Optional[Path] = None) -> Path:
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = cache_dir / "ml-100k"
    if (data_dir / "u.data").exists():
        return data_dir
    zip_path = cache_dir / "ml-100k.zip"
    if not zip_path.exists():
        print(f"Downloading MovieLens-100K to {zip_path}...")
        urlretrieve(ML100K_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(cache_dir)
    return data_dir

def download_ml1m(cache_dir: Optional[Path] = None) -> Path:
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = cache_dir / "ml-1m"
    if (data_dir / "ratings.dat").exists():
        return data_dir
    zip_path = cache_dir / "ml-1m.zip"
    if not zip_path.exists():
        print(f"Downloading MovieLens-1M to {zip_path}...")
        urlretrieve(ML1M_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(cache_dir)
    return data_dir

def load_ml100k_sequences(max_seq_len: int = 50, min_interactions: int = 5,
                          cache_dir: Optional[Path] = None):
    data_dir = download_ml100k(cache_dir)
    interactions = []
    with open(data_dir / "u.data") as f:
        for line in f:
            parts = line.strip().split('\t')
            user, item, rating, ts = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            if rating >= 3:
                interactions.append((user, item, ts))

    interactions.sort(key=lambda x: (x[0], x[2]))

    user_seqs = defaultdict(list)
    items = set()
    for user, item, _ in interactions:
        user_seqs[user].append(item)
        items.add(item)

    item_remap = {old: new for new, old in enumerate(sorted(items), start=1)}
    num_items = len(items)

    user_sequences = {}
    for user, seq in user_seqs.items():
        if len(seq) < min_interactions:
            continue
        remapped = [item_remap[i] for i in seq]
        user_sequences[user] = remapped[-max_seq_len:]

    return user_sequences, num_items

def load_ml1m_sequences(max_seq_len: int = 50, min_interactions: int = 5,
                        cache_dir: Optional[Path] = None):
    data_dir = download_ml1m(cache_dir)
    interactions = []
    with open(data_dir / "ratings.dat", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            user, item, rating, ts = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            if rating >= 3:
                interactions.append((user, item, ts))

    interactions.sort(key=lambda x: (x[0], x[2]))

    user_seqs = defaultdict(list)
    items = set()
    for user, item, _ in interactions:
        user_seqs[user].append(item)
        items.add(item)

    item_remap = {old: new for new, old in enumerate(sorted(items), start=1)}
    num_items = len(items)

    item_counts = defaultdict(int)
    for user, item, _ in interactions:
        item_counts[item_remap.get(item, 0)] += 1

    pop_threshold = np.percentile(list(item_counts.values()), 80)

    user_sequences = {}
    user_meta = {}
    for user, seq in user_seqs.items():
        if len(seq) < min_interactions:
            continue
        remapped = [item_remap[i] for i in seq]
        user_sequences[user] = remapped[-max_seq_len:]

        head_items = sum(1 for i in remapped if item_counts.get(i, 0) >= pop_threshold)
        user_meta[user] = {
            "activity": len(seq),
            "head_item_ratio": head_items / len(remapped) if remapped else 0,
        }

    return user_sequences, num_items, user_meta

def load_recsys_sequences(dataset_name: str, max_seq_len: int = 50,
                          min_interactions: int = 5, cache_dir: Optional[Path] = None):
    if dataset_name == "movielens100k":
        seqs, n_items = load_ml100k_sequences(max_seq_len, min_interactions, cache_dir)
        return seqs, n_items, None
    elif dataset_name in ("ml1m", "movielens1m"):
        return load_ml1m_sequences(max_seq_len, min_interactions, cache_dir)
    elif dataset_name in ("amazon_beauty", "beauty"):
        from src.data.amazon_beauty import load_beauty_sequences
        return load_beauty_sequences(max_seq_len, min_interactions, cache_dir)
    elif dataset_name in ("amazon_toys", "toys"):
        from src.data.amazon_beauty import load_toys_sequences
        return load_toys_sequences(max_seq_len, min_interactions, cache_dir)
    else:
        raise ValueError(f"Unknown recsys dataset: {dataset_name}")

class SeqRecDataset(Dataset):

    def __init__(self, sequences: list[list[int]], max_seq_len: int = 50,
                 num_items: int = 1683, mode: str = "train"):
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.mode = mode
        self.data = []

        for seq in sequences:
            if mode == "train":
                input_seq = seq[:-2]
                target_seq = seq[1:-1]
            elif mode == "val":
                input_seq = seq[:-1]
                target_seq = seq[1:]
            else:
                input_seq = seq[:-1]
                target_seq = seq[1:]

            if len(input_seq) < 2:
                continue

            if len(input_seq) > max_seq_len:
                input_seq = input_seq[-max_seq_len:]
                target_seq = target_seq[-max_seq_len:]

            self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        pad_len = self.max_seq_len - len(input_seq)

        input_ids = [0] * pad_len + input_seq
        labels = [0] * pad_len + target_seq

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def partition_recsys_by_user(user_sequences: dict, num_clients: int,
                             seed: int = 42, user_meta: dict = None,
                             non_iid_strength: str = "medium"):
    rng = np.random.default_rng(seed)
    users = sorted(user_sequences.keys())

    if non_iid_strength == "iid" or user_meta is None:
        rng.shuffle(users)
        clients_users = np.array_split(users, num_clients)
    else:
        activity = {u: user_meta[u]["activity"] if user_meta and u in user_meta
                    else len(user_sequences[u]) for u in users}
        sorted_users = sorted(users, key=lambda u: activity[u])

        if non_iid_strength == "severe":
            chunk_size = max(1, len(sorted_users) // num_clients)
            clients_users = []
            for i in range(num_clients):
                start = i * chunk_size
                end = min(start + chunk_size, len(sorted_users))
                if start < len(sorted_users):
                    clients_users.append(sorted_users[start:end])
            while len(clients_users) < num_clients:
                clients_users.append([sorted_users[rng.integers(len(sorted_users))]])
        elif non_iid_strength == "medium":
            n_groups = min(num_clients, 4)
            group_size = len(sorted_users) // n_groups
            groups = []
            for g in range(n_groups):
                start = g * group_size
                end = start + group_size if g < n_groups - 1 else len(sorted_users)
                groups.append(sorted_users[start:end])
            all_assigned = []
            clients_per_group = num_clients // n_groups
            for g_users in groups:
                rng.shuffle(g_users)
                splits = np.array_split(g_users, clients_per_group)
                all_assigned.extend(splits)
            while len(all_assigned) < num_clients:
                all_assigned.append([sorted_users[rng.integers(len(sorted_users))]])
            clients_users = all_assigned[:num_clients]
        else:
            rng.shuffle(sorted_users)
            clients_users = np.array_split(sorted_users, num_clients)

    client_train_seqs = []
    client_val_items = []
    client_user_groups = []

    for client_users in clients_users:
        train_seqs = []
        val_items = []
        for u in client_users:
            seq = user_sequences[u]
            if len(seq) >= 3:
                train_seqs.append(seq[:-1])
                val_items.append((seq[:-1], seq[-1]))
        client_train_seqs.append(train_seqs)
        client_val_items.append(val_items)

        activities = [len(user_sequences[u]) for u in client_users]
        avg_act = np.mean(activities) if activities else 0
        if avg_act < 10:
            group = "low_activity"
        elif avg_act < 30:
            group = "mid_activity"
        else:
            group = "high_activity"
        client_user_groups.append(group)

    return client_train_seqs, client_val_items, client_user_groups

def partition_ml100k_by_user(user_sequences: dict, num_clients: int,
                             seed: int = 42) -> tuple[list[list[list[int]]], list[list[int]]]:
    train_seqs, val_items, _ = partition_recsys_by_user(
        user_sequences, num_clients, seed)
    return train_seqs, val_items
