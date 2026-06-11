from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoImageProcessor

from .partition import load_partition

DATASET_CONFIGS = {
    "sst2": {
        "hf_name": "glue", "hf_config": "sst2",
        "text_cols": ["sentence"], "label_col": "label",
        "train_split": "train", "val_split": "validation",
        "max_length": 128,
    },
    "agnews": {
        "hf_name": "ag_news", "hf_config": None,
        "text_cols": ["text"], "label_col": "label",
        "train_split": "train", "val_split": "test",
        "max_length": 128,
    },
    "qnli": {
        "hf_name": "glue", "hf_config": "qnli",
        "text_cols": ["question", "sentence"], "label_col": "label",
        "train_split": "train", "val_split": "validation",
        "max_length": 128,
    },
    "mrpc": {
        "hf_name": "glue", "hf_config": "mrpc",
        "text_cols": ["sentence1", "sentence2"], "label_col": "label",
        "train_split": "train", "val_split": "validation",
        "max_length": 128,
    },
    "cifar100": {
        "hf_name": "cifar100", "hf_config": None,
        "task_type": "image_classification",
        "label_col": "fine_label",
        "train_split": "train", "val_split": "test",
    },
}

class TokenizedTextDataset(Dataset):

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class ImageDataset(Dataset):

    def __init__(self, hf_dataset, processor, label_col: str = "fine_label"):
        self.dataset = hf_dataset
        self.processor = processor
        self.label_col = label_col

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item.get("img") or item.get("image")
        if img.mode != "RGB":
            img = img.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label = torch.tensor(item[self.label_col], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": label}

def load_full_dataset(dataset_name: str, split: str = "train", tokenizer=None, processor=None):
    cfg = DATASET_CONFIGS[dataset_name]
    task_type = cfg.get("task_type", "text_classification")

    if cfg["hf_config"]:
        hf_data = load_dataset(cfg["hf_name"], cfg["hf_config"], split=cfg.get(f"{split}_split", split))
    else:
        hf_data = load_dataset(cfg["hf_name"], split=cfg.get(f"{split}_split", split))

    if task_type == "image_classification":
        return ImageDataset(hf_data, processor, label_col=cfg["label_col"])

    text_cols = cfg["text_cols"]
    label_col = cfg["label_col"]
    max_length = cfg.get("max_length", 128)

    if len(text_cols) == 1:
        texts = list(hf_data[text_cols[0]])
        encodings = tokenizer(texts, truncation=True, padding="max_length",
                              max_length=max_length, return_tensors="pt")
    else:
        texts_a = list(hf_data[text_cols[0]])
        texts_b = list(hf_data[text_cols[1]])
        encodings = tokenizer(texts_a, texts_b, truncation=True, padding="max_length",
                              max_length=max_length, return_tensors="pt")

    labels = list(hf_data[label_col])
    return TokenizedTextDataset(encodings, labels)

def get_client_dataloader(
    full_dataset: Dataset,
    partition_dir: str | Path,
    client_id: int,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    indices = load_partition(partition_dir, client_id)
    subset = Subset(full_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def get_val_dataloader(
    dataset_name: str,
    tokenizer=None,
    processor=None,
    batch_size: int = 64,
) -> DataLoader:
    dataset = load_full_dataset(dataset_name, split="val", tokenizer=tokenizer, processor=processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
