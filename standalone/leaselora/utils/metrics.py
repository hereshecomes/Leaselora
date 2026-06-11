from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

@dataclass
class MetricsCollector:

    target_quality: float = 0.90
    coverage_target: int = 1

    round_qualities: list[float] = field(default_factory=list)
    round_wall_times: list[float] = field(default_factory=list)
    round_low_res_rates: list[float] = field(default_factory=list)
    round_coverage_counts: list[dict] = field(default_factory=list)
    round_violation_rates: list[float] = field(default_factory=list)

    lease_predicted_utilities: list[float] = field(default_factory=list)
    lease_realized_gains: list[float] = field(default_factory=list)

    def record_round(
        self,
        quality: float,
        wall_time_sec: float,
        low_res_rate: float,
        coverage_counts: dict[tuple[int, int], int] | None = None,
        violation_rate: float = 0.0,
    ) -> None:
        self.round_qualities.append(quality)
        self.round_wall_times.append(wall_time_sec)
        self.round_low_res_rates.append(low_res_rate)
        self.round_coverage_counts.append(coverage_counts or {})
        self.round_violation_rates.append(violation_rate)

    def record_lease(self, predicted_utility: float, realized_gain: float) -> None:
        self.lease_predicted_utilities.append(predicted_utility)
        self.lease_realized_gains.append(realized_gain)

    @property
    def final_quality(self) -> float:
        return self.round_qualities[-1] if self.round_qualities else 0.0

    @property
    def reached_target(self) -> bool:
        return any(q >= self.target_quality for q in self.round_qualities)

    @property
    def ttq_round(self) -> Optional[int]:
        for i, q in enumerate(self.round_qualities):
            if q >= self.target_quality:
                return i
        return None

    @property
    def total_runtime_sec(self) -> float:
        return float(sum(self.round_wall_times))

    @property
    def ttq_sec(self) -> Optional[float]:
        cum = 0.0
        for q, wt in zip(self.round_qualities, self.round_wall_times):
            cum += wt
            if q >= self.target_quality:
                return cum
        return None

    @property
    def low_res_participation(self) -> float:
        if not self.round_low_res_rates:
            return 0.0
        return float(np.mean(self.round_low_res_rates))

    @property
    def p95_round_sec(self) -> float:
        if not self.round_wall_times:
            return 0.0
        return float(np.percentile(self.round_wall_times, 95))

    @property
    def starvation_ratio(self) -> float:
        if not self.round_coverage_counts:
            return 0.0
        last = self.round_coverage_counts[-1]
        if not last:
            return 0.0
        starved = sum(1 for v in last.values() if v < self.coverage_target)
        return starved / len(last)

    def effective_coverage(self, slice_set) -> float:
        if slice_set is None or len(slice_set) == 0:
            return 0.0
        n_sufficient = sum(1 for s in slice_set if s.progress >= 0.9)
        return n_sufficient / len(slice_set)

    def undertrained_important_ratio(
        self, slice_set, progress_threshold: float = 0.9, importance_threshold: float = 0.5,
    ) -> float:
        if slice_set is None or len(slice_set) == 0:
            return 0.0
        important = [s for s in slice_set if s.intrinsic_importance >= importance_threshold]
        if not important:
            return 0.0
        undertrained = sum(1 for s in important if s.progress < progress_threshold)
        return undertrained / len(important)

    def starvation_details(self, slice_set=None) -> dict:
        if not self.round_coverage_counts:
            return {}
        last = self.round_coverage_counts[-1]
        details = {}
        for sid, count in last.items():
            entry = {
                "coverage_count": count,
                "starved": count < self.coverage_target,
            }
            if slice_set is not None:
                try:
                    s = slice_set[sid]
                    entry["importance"] = s.intrinsic_importance
                    entry["progress"] = s.progress
                    entry["need"] = s.need
                    entry["fairness_debt"] = s.fairness_debt
                except (KeyError, IndexError):
                    pass
            details[str(sid)] = entry
        return details

    @property
    def mean_violation_rate(self) -> float:
        if not self.round_violation_rates:
            return 0.0
        return float(np.mean(self.round_violation_rates))

    def surrogate_correlation(self) -> tuple[float, float]:
        if len(self.lease_predicted_utilities) < 5:
            return 0.0, 0.0
        from scipy import stats

        pred = np.array(self.lease_predicted_utilities)
        real = np.array(self.lease_realized_gains)
        pearson = float(np.corrcoef(pred, real)[0, 1])
        spearman = float(stats.spearmanr(pred, real).correlation)
        return pearson, spearman

    def quality_at_wallclock(self, budget_sec: float) -> Optional[float]:
        cum = 0.0
        best_q = None
        for q, wt in zip(self.round_qualities, self.round_wall_times):
            cum += wt
            if cum <= budget_sec:
                best_q = q
            else:
                break
        return best_q

    def quality_time_auc(self) -> float:
        if not self.round_qualities:
            return 0.0
        auc = 0.0
        for q, wt in zip(self.round_qualities, self.round_wall_times):
            auc += q * wt
        return auc

    def ttq_at_target(self, target: float) -> Optional[float]:
        cum = 0.0
        for q, wt in zip(self.round_qualities, self.round_wall_times):
            cum += wt
            if q >= target:
                return cum
        return None

    def summary(self, slice_set=None) -> dict:
        wallclock_budgets = [1000, 3000, 5000]
        quality_at = {
            f"quality@{b}s": self.quality_at_wallclock(b) for b in wallclock_budgets
        }
        ttq_targets = [0.15, 0.20, 0.25, 0.30]
        ttq_sweep = {
            f"ttq@{t:.2f}": self.ttq_at_target(t) for t in ttq_targets
        }
        result = {
            "final_quality": self.final_quality,
            "reached_target": self.reached_target,
            "ttq_sec": self.ttq_sec,
            "total_runtime_sec": self.total_runtime_sec,
            "ttq_round": self.ttq_round,
            "low_res_participation": self.low_res_participation,
            "p95_round_sec": self.p95_round_sec,
            "starvation_ratio": self.starvation_ratio,
            "mean_violation_rate": self.mean_violation_rate,
            "quality_time_auc": self.quality_time_auc(),
            **quality_at,
            **ttq_sweep,
        }
        if slice_set is not None:
            result["effective_coverage"] = self.effective_coverage(slice_set)
            result["undertrained_important_ratio"] = self.undertrained_important_ratio(slice_set)
        return result
