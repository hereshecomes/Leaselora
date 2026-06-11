from __future__ import annotations

import math

import numpy as np
import torch

def compute_recsys_metrics(model, val_items: list[tuple], max_seq_len: int = 50,
                           num_items: int = 1683, device: str = "cuda:0",
                           k: int = 10) -> dict:
    model.eval()
    torch.cuda.empty_cache()
    hits = 0
    ndcgs = 0.0
    total = 0

    batch_size = 32 if num_items > 5000 else 64
    all_seqs = []
    all_targets = []
    for input_seq, target_item in val_items:
        pad_len = max_seq_len - len(input_seq)
        padded = [0] * pad_len + input_seq
        all_seqs.append(padded)
        all_targets.append(target_item)

    with torch.no_grad():
        for start in range(0, len(all_seqs), batch_size):
            batch_seqs = all_seqs[start:start + batch_size]
            batch_targets = all_targets[start:start + batch_size]

            input_ids = torch.tensor(batch_seqs, dtype=torch.long, device=device)
            out = model(input_ids, eval_last_only=True)
            scores = out.logits[:, -1, :]

            _, topk_indices = scores.topk(k, dim=-1)
            topk_items = (topk_indices + 1).cpu().numpy()

            for i, target_item in enumerate(batch_targets):
                if target_item in topk_items[i]:
                    hits += 1
                    rank = int(np.where(topk_items[i] == target_item)[0][0]) + 1
                    ndcgs += 1.0 / math.log2(rank + 1)
                total += 1

    hr = hits / max(total, 1)
    ndcg = ndcgs / max(total, 1)
    return {"hr@10": hr, "ndcg@10": ndcg, "total_eval": total}
