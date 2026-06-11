from __future__ import annotations

import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

logger = logging.getLogger("fl_testbed.evaluator")

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cuda",
    metric: str = "accuracy",
) -> dict:
    model.to(device).eval()
    t0 = time.time()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                for key in ("labels", "input_ids", "pixel_values"):
                    if key in batch:
                        bs = batch[key].shape[0]
                        break
                else:
                    bs = 1
                total_loss += outputs.loss.item() * bs
                total_samples += bs

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    eval_time = time.time() - t0
    accuracy = accuracy_score(all_labels, all_preds)

    results = {
        "eval_accuracy": accuracy,
        "eval_loss": total_loss / max(total_samples, 1),
        "eval_samples": len(all_labels),
        "evaluation_time_sec": eval_time,
    }

    if metric == "f1" or metric == "macro_f1":
        results["eval_macro_f1"] = f1_score(all_labels, all_preds, average="macro")
    else:
        results["eval_macro_f1"] = None

    logger.info(f"Evaluation: accuracy={accuracy:.4f}, time={eval_time:.1f}s")
    return results
