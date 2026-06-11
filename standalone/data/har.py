from __future__ import annotations

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_CACHE = Path(__file__).resolve().parent.parent.parent / ".data_cache"
DATA_CACHE.mkdir(parents=True, exist_ok=True)

UCI_HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

UCIHAR_ACTIVITIES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
]

def _download_ucihar() -> Path:
    dest = DATA_CACHE / "UCI HAR Dataset"
    if dest.exists():
        return dest

    zip_path = DATA_CACHE / "ucihar.zip"
    if not zip_path.exists():
        print("Downloading UCI-HAR dataset...")
        urllib.request.urlretrieve(UCI_HAR_URL, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_CACHE)

    return dest

def _load_ucihar_inertial(base_path: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signals_dir = base_path / split / "Inertial Signals"
    signal_files = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    signals = []
    for sf in signal_files:
        filepath = signals_dir / f"{sf}_{split}.txt"
        data = np.loadtxt(filepath)
        signals.append(data)

    X = np.stack(signals, axis=-1)
    y = np.loadtxt(base_path / split / f"y_{split}.txt").astype(int) - 1
    subjects = np.loadtxt(base_path / split / f"subject_{split}.txt").astype(int)

    return X, y, subjects

def load_ucihar(use_raw: bool = True) -> dict:
    base = _download_ucihar()

    train_X, train_y, train_subj = _load_ucihar_inertial(base, "train")
    test_X, test_y, test_subj = _load_ucihar_inertial(base, "test")

    mu = train_X.mean(axis=(0, 1), keepdims=True)
    std = train_X.std(axis=(0, 1), keepdims=True) + 1e-8
    train_X = (train_X - mu) / std
    test_X = (test_X - mu) / std

    return {
        "train_X": train_X.astype(np.float32),
        "train_y": train_y,
        "train_subjects": train_subj,
        "val_X": test_X.astype(np.float32),
        "val_y": test_y,
        "val_subjects": test_subj,
        "num_classes": 6,
        "seq_len": 128,
        "num_channels": 9,
        "num_subjects": 30,
    }

WISDM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip"

WISDM_ACTIVITIES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "O", "P", "Q", "R", "S"
]

def _download_wisdm() -> Path:
    dest = DATA_CACHE / "wisdm-dataset"
    if dest.exists():
        return dest

    zip_path = DATA_CACHE / "wisdm.zip"
    if not zip_path.exists():
        print("Downloading WISDM dataset...")
        urllib.request.urlretrieve(WISDM_URL, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_CACHE)

    return dest

def _parse_wisdm_file(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects, activities, values = [], [], []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';').strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                subj = int(parts[0])
                act = parts[1].strip()
                x = float(parts[3])
                y = float(parts[4])
                z_str = parts[5].rstrip(';').strip()
                z = float(z_str) if z_str else 0.0
                subjects.append(subj)
                activities.append(act)
                values.append([x, y, z])
            except (ValueError, IndexError):
                continue

    return np.array(subjects), np.array(activities), np.array(values, dtype=np.float32)

def load_wisdm(window_size: int = 128, stride: int = 64) -> dict:
    base = _download_wisdm()

    phone_accel_path = base / "raw" / "phone" / "accel"
    phone_gyro_path = base / "raw" / "phone" / "gyro"

    accel_files = sorted(phone_accel_path.glob("*.txt"))
    gyro_files = sorted(phone_gyro_path.glob("*.txt"))

    if not accel_files:
        raise FileNotFoundError(f"No accel data found in {phone_accel_path}")

    gyro_by_subj = {}
    for gf in gyro_files:
        subj_id = int(gf.name.split('_')[1])
        gyro_by_subj[subj_id] = gf

    segments_X = []
    segments_y = []
    segments_subj = []
    all_activities = set()

    subject_data = {}
    for af in accel_files:
        subj_id = int(af.name.split('_')[1])
        subj_a, act_a, vals_a = _parse_wisdm_file(af)
        all_activities.update(act_a)

        gf = gyro_by_subj.get(subj_id)
        if gf is not None:
            _, _, vals_g = _parse_wisdm_file(gf)
        else:
            vals_g = np.zeros((len(vals_a), 3), dtype=np.float32)

        min_len = min(len(vals_a), len(vals_g))
        subject_data[subj_id] = {
            "accel": vals_a[:min_len],
            "gyro": vals_g[:min_len],
            "activities": act_a[:min_len],
        }

    activity_set = sorted(all_activities)
    act_to_idx = {a: i for i, a in enumerate(activity_set)}
    num_classes = len(activity_set)
    all_subjects = sorted(subject_data.keys())
    num_subjects = len(all_subjects)

    for subj_id in all_subjects:
        sd = subject_data[subj_id]
        combined = np.concatenate([sd["accel"], sd["gyro"]], axis=-1)
        acts = sd["activities"]

        for start in range(0, len(combined) - window_size + 1, stride):
            window = combined[start:start + window_size]
            window_acts = acts[start:start + window_size]
            act_counts = {}
            for a in window_acts:
                act_counts[a] = act_counts.get(a, 0) + 1
            majority_act = max(act_counts, key=act_counts.get)

            if majority_act in act_to_idx:
                segments_X.append(window)
                segments_y.append(act_to_idx[majority_act])
                segments_subj.append(subj_id)

    X = np.array(segments_X, dtype=np.float32)
    y = np.array(segments_y, dtype=np.int64)
    subjects = np.array(segments_subj, dtype=np.int64)

    mu = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mu) / std

    rng = np.random.default_rng(42)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split_idx = int(len(X) * 0.8)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    return {
        "train_X": X[train_idx],
        "train_y": y[train_idx],
        "train_subjects": subjects[train_idx],
        "val_X": X[val_idx],
        "val_y": y[val_idx],
        "val_subjects": subjects[val_idx],
        "num_classes": num_classes,
        "seq_len": window_size,
        "num_channels": 6,
        "num_subjects": num_subjects,
    }

class HARDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"input_values": self.X[idx], "labels": self.y[idx]}

def partition_har_by_subject(
    subjects: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    unique_subjects = sorted(set(subjects))
    num_subjects = len(unique_subjects)

    subj_to_indices = {}
    for idx, s in enumerate(subjects):
        if s not in subj_to_indices:
            subj_to_indices[s] = []
        subj_to_indices[s].append(idx)

    client_indices = [[] for _ in range(num_clients)]

    if num_clients <= num_subjects:
        perm = rng.permutation(len(unique_subjects))
        for i, subj_idx in enumerate(perm[:num_clients]):
            subj_id = unique_subjects[subj_idx]
            client_indices[i] = subj_to_indices[subj_id]

        for subj_idx in perm[num_clients:]:
            subj_id = unique_subjects[subj_idx]
            target_client = rng.integers(0, num_clients)
            client_indices[target_client].extend(subj_to_indices[subj_id])
    else:
        subj_list = list(unique_subjects)
        for client_id in range(num_clients):
            subj_id = subj_list[client_id % num_subjects]
            all_idx = subj_to_indices[subj_id]
            chunk_size = max(1, len(all_idx) // (num_clients // num_subjects + 1))
            offset = (client_id // num_subjects) * chunk_size
            end = min(offset + chunk_size, len(all_idx))
            if offset < len(all_idx):
                client_indices[client_id] = all_idx[offset:end]
            else:
                client_indices[client_id] = rng.choice(
                    all_idx, size=min(chunk_size, len(all_idx)), replace=False).tolist()

    for i in range(num_clients):
        if not client_indices[i]:
            subj_id = unique_subjects[rng.integers(0, num_subjects)]
            all_idx = subj_to_indices[subj_id]
            client_indices[i] = rng.choice(
                all_idx, size=max(1, len(all_idx) // 4), replace=False).tolist()

    return client_indices
