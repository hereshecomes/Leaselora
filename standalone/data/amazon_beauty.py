from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np

BEAUTY_URLS = [
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/All_Beauty.json.gz",
    "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/All_Beauty.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
]
TOYS_URLS = [
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Toys_and_Games.json.gz",
    "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Toys_and_Games.json.gz",
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
]
CACHE_DIR = Path(os.environ.get("BEAUTY_CACHE", Path.home() / ".cache" / "amazon_recsys"))

def _download_dataset(urls: list[str], filename: str, dataset_name: str,
                      cache_dir: Optional[Path] = None) -> Path:
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / filename
    if gz_path.exists():
        return gz_path

    for url in urls:
        try:
            print(f"Downloading {dataset_name} from {url}...")
            urlretrieve(url, gz_path)
            if gz_path.stat().st_size > 1000:
                return gz_path
        except Exception as e:
            print(f"  Failed ({e}), trying next URL...")

    raise RuntimeError(
        f"Failed to download {dataset_name}. Please manually download from "
        f"https://jmcauley.ucsd.edu/data/amazon/ and place at {gz_path}"
    )

def _download_beauty(cache_dir: Optional[Path] = None) -> Path:
    return _download_dataset(BEAUTY_URLS, "Beauty_5.json.gz", "Amazon-Beauty", cache_dir)

def _download_toys(cache_dir: Optional[Path] = None) -> Path:
    return _download_dataset(TOYS_URLS, "Toys_5.json.gz", "Amazon-Toys", cache_dir)

def _load_amazon_sequences(gz_path: Path, dataset_label: str,
                           max_seq_len: int = 50, min_interactions: int = 5):
    user_items = defaultdict(list)
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                review = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            user = review.get("reviewerID", "")
            item = review.get("asin", "")
            ts = review.get("unixReviewTime", 0)
            if user and item:
                user_items[user].append((item, ts))

    for u in user_items:
        user_items[u].sort(key=lambda x: x[1])

    all_items = set()
    for u, pairs in user_items.items():
        for item, _ in pairs:
            all_items.add(item)
    item_remap = {old: new for new, old in enumerate(sorted(all_items), start=1)}
    num_items = len(all_items)

    item_counts = defaultdict(int)
    for u, pairs in user_items.items():
        for item, _ in pairs:
            item_counts[item_remap[item]] += 1

    pop_threshold = np.percentile(list(item_counts.values()), 80) if item_counts else 1

    user_sequences = {}
    user_meta = {}
    uid_counter = 0

    for u, pairs in user_items.items():
        if len(pairs) < min_interactions:
            continue

        items = [item_remap[item] for item, _ in pairs]
        items = items[-max_seq_len:]
        user_sequences[uid_counter] = items

        head_items = sum(1 for i in items if item_counts.get(i, 0) >= pop_threshold)
        user_meta[uid_counter] = {
            "activity": len(pairs),
            "head_item_ratio": head_items / len(items) if items else 0,
        }
        uid_counter += 1

    print(f"{dataset_label}: {len(user_sequences)} users, {num_items} items, "
          f"avg seq len {np.mean([len(s) for s in user_sequences.values()]):.1f}")

    return user_sequences, num_items, user_meta

def load_beauty_sequences(max_seq_len: int = 50, min_interactions: int = 5,
                          cache_dir: Optional[Path] = None):
    gz_path = _download_beauty(cache_dir)
    return _load_amazon_sequences(gz_path, "Amazon-Beauty", max_seq_len, min_interactions)

def load_toys_sequences(max_seq_len: int = 50, min_interactions: int = 5,
                        cache_dir: Optional[Path] = None):
    gz_path = _download_toys(cache_dir)
    return _load_amazon_sequences(gz_path, "Amazon-Toys", max_seq_len, min_interactions)
