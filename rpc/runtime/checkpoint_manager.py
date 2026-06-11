from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger("fl_testbed.checkpoint")

class CheckpointManager:

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        round_id: int,
        global_adapter_state: dict[str, torch.Tensor],
        feedback_state: dict | None = None,
        extra: dict | None = None,
    ):
        ckpt_path = self.checkpoint_dir / f"round_{round_id:04d}.pt"
        ckpt = {
            "round_id": round_id,
            "global_adapter_state": {k: v.cpu() for k, v in global_adapter_state.items()},
        }
        if feedback_state:
            ckpt["feedback_state"] = feedback_state
        if extra:
            ckpt["extra"] = extra
        torch.save(ckpt, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(ckpt, latest_path)

    def load_latest(self) -> dict | None:
        latest_path = self.checkpoint_dir / "latest.pt"
        if not latest_path.exists():
            return None
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        logger.info(f"Checkpoint loaded: round={ckpt['round_id']}")
        return ckpt

    def load_round(self, round_id: int) -> dict | None:
        ckpt_path = self.checkpoint_dir / f"round_{round_id:04d}.pt"
        if not ckpt_path.exists():
            return None
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
