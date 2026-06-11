from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

@dataclass
class SliceFeedback:
    slice_id: str
    coverage_count: int = 0
    total_updates: int = 0
    variance_estimate: float = 0.0
    quality_score: float = 0.5
    last_updated_round: int = -1

class FeedbackState:

    def __init__(
        self,
        slice_ids: list[str],
        enable_coverage: bool = True,
        enable_service: bool = True,
        enable_variance: bool = True,
        ema_alpha: float = 0.1,
    ):
        self.enable_coverage = enable_coverage
        self.enable_service = enable_service
        self.enable_variance = enable_variance
        self.ema_alpha = ema_alpha

        self.slices: dict[str, SliceFeedback] = {
            sid: SliceFeedback(slice_id=sid) for sid in slice_ids
        }
        self.round_count: int = 0

    def update_after_round(
        self,
        round_id: int,
        updated_slice_ids: dict[str, list[int]],
        slice_deltas: dict[str, list[np.ndarray]] | None = None,
    ):
        self.round_count = round_id + 1

        for sid, fb in self.slices.items():
            clients = updated_slice_ids.get(sid, [])
            if clients:
                fb.coverage_count += 1
                fb.total_updates += len(clients)
                fb.last_updated_round = round_id
                fb.quality_score = min(1.0, fb.quality_score + self.ema_alpha)
            else:
                fb.quality_score = max(0.0, fb.quality_score - self.ema_alpha * 0.5)

        if self.enable_variance and slice_deltas:
            for sid, deltas in slice_deltas.items():
                if sid in self.slices and len(deltas) > 1:
                    flat = [d.flatten() for d in deltas]
                    stacked = np.stack(flat)
                    var = np.mean(np.var(stacked, axis=0))
                    fb = self.slices[sid]
                    fb.variance_estimate = (
                        (1 - self.ema_alpha) * fb.variance_estimate + self.ema_alpha * float(var)
                    )

    def get_slice_scores(self, slice_ids: list[str]) -> np.ndarray:
        scores = np.zeros(len(slice_ids))
        for i, sid in enumerate(slice_ids):
            fb = self.slices.get(sid)
            if fb is None:
                scores[i] = 1.0
                continue

            coverage_need = 1.0 / max(fb.coverage_count + 1, 1)
            quality_need = 1.0 - fb.quality_score
            recency_need = (self.round_count - fb.last_updated_round) / max(self.round_count, 1)
            variance_bonus = fb.variance_estimate

            score = 0.0
            if self.enable_coverage:
                score += 0.4 * coverage_need
            if self.enable_service:
                score += 0.3 * quality_need
            if self.enable_variance:
                score += 0.2 * variance_bonus
            score += 0.1 * recency_need

            scores[i] = score

        return scores

    def to_dict(self) -> dict:
        return {
            "round_count": self.round_count,
            "slices": {
                sid: {
                    "coverage_count": fb.coverage_count,
                    "total_updates": fb.total_updates,
                    "variance_estimate": fb.variance_estimate,
                    "quality_score": fb.quality_score,
                    "last_updated_round": fb.last_updated_round,
                }
                for sid, fb in self.slices.items()
            },
        }

    def load_dict(self, data: dict):
        self.round_count = data.get("round_count", 0)
        for sid, fb_data in data.get("slices", {}).items():
            if sid in self.slices:
                fb = self.slices[sid]
                fb.coverage_count = fb_data.get("coverage_count", 0)
                fb.total_updates = fb_data.get("total_updates", 0)
                fb.variance_estimate = fb_data.get("variance_estimate", 0.0)
                fb.quality_score = fb_data.get("quality_score", 0.5)
                fb.last_updated_round = fb_data.get("last_updated_round", -1)
