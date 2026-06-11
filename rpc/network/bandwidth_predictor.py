from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger("fl_testbed.network.predictor")

@dataclass
class UploadRecord:
    round_id: int
    payload_bytes: int
    upload_time_sec: float
    success: bool
    measured_mbps: float = 0.0

    def __post_init__(self):
        if self.success and self.upload_time_sec > 0:
            self.measured_mbps = (self.payload_bytes * 8) / (self.upload_time_sec * 1e6)

@dataclass
class BandwidthPrediction:
    predicted_uplink_mbps: float
    predicted_rtt_ms: float
    uncertainty: float
    online_probability: float
    staleness: int
    num_samples: int

    def to_dict(self) -> dict:
        return {
            "predicted_uplink_mbps": round(self.predicted_uplink_mbps, 2),
            "predicted_rtt_ms": round(self.predicted_rtt_ms, 2),
            "uncertainty": round(self.uncertainty, 3),
            "online_probability": round(self.online_probability, 3),
            "staleness": self.staleness,
            "num_samples": self.num_samples,
        }

class BandwidthPredictor:

    def __init__(
        self,
        beta: float = 0.3,
        min_samples: int = 3,
        failure_penalty: float = 0.5,
        stale_after_rounds: int = 3,
        max_history: int = 20,
        default_uplink_mbps: float = 80.0,
        default_rtt_ms: float = 50.0,
    ):
        self.beta = beta
        self.min_samples = min_samples
        self.failure_penalty = failure_penalty
        self.stale_after_rounds = stale_after_rounds
        self.max_history = max_history
        self.default_uplink_mbps = default_uplink_mbps
        self.default_rtt_ms = default_rtt_ms

        self.histories: dict[int, deque[UploadRecord]] = {}
        self.bw_ema: dict[int, float] = {}
        self.rtt_ema: dict[int, float] = {}
        self.last_success_round: dict[int, int] = {}
        self.consecutive_failures: dict[int, int] = {}

    def record_upload(self, client_id: int, record: UploadRecord):
        if client_id not in self.histories:
            self.histories[client_id] = deque(maxlen=self.max_history)
            self.bw_ema[client_id] = self.default_uplink_mbps
            self.rtt_ema[client_id] = self.default_rtt_ms
            self.last_success_round[client_id] = -1
            self.consecutive_failures[client_id] = 0

        self.histories[client_id].append(record)

        if record.success and record.measured_mbps > 0:
            self.bw_ema[client_id] = (
                self.beta * record.measured_mbps
                + (1 - self.beta) * self.bw_ema[client_id]
            )
            self.last_success_round[client_id] = record.round_id
            self.consecutive_failures[client_id] = 0
        else:
            self.consecutive_failures[client_id] += 1
            self.bw_ema[client_id] *= self.failure_penalty

    def record_rtt(self, client_id: int, rtt_ms: float):
        if client_id not in self.rtt_ema:
            self.rtt_ema[client_id] = self.default_rtt_ms
        self.rtt_ema[client_id] = (
            self.beta * rtt_ms + (1 - self.beta) * self.rtt_ema[client_id]
        )

    def predict(self, client_id: int, current_round: int) -> BandwidthPrediction:
        if client_id not in self.bw_ema:
            return BandwidthPrediction(
                predicted_uplink_mbps=self.default_uplink_mbps,
                predicted_rtt_ms=self.default_rtt_ms,
                uncertainty=1.0,
                online_probability=0.9,
                staleness=999,
                num_samples=0,
            )

        bw = max(0.1, self.bw_ema[client_id])
        rtt = self.rtt_ema[client_id]
        history = self.histories[client_id]
        num_samples = len(history)

        last_ok = self.last_success_round[client_id]
        staleness = current_round - last_ok if last_ok >= 0 else 999

        if num_samples < self.min_samples:
            uncertainty = 1.0
        else:
            base_uncertainty = 0.1
            stale_factor = min(1.0, staleness / self.stale_after_rounds) * 0.5
            failure_factor = min(1.0, self.consecutive_failures[client_id] / 3) * 0.4
            uncertainty = min(1.0, base_uncertainty + stale_factor + failure_factor)

        if num_samples == 0:
            online_prob = 0.9
        else:
            recent = list(history)[-min(5, len(history)):]
            success_rate = sum(1 for r in recent if r.success) / len(recent)
            online_prob = max(0.1, success_rate)

        if staleness > self.stale_after_rounds:
            bw *= max(0.3, 1.0 - 0.1 * (staleness - self.stale_after_rounds))

        return BandwidthPrediction(
            predicted_uplink_mbps=bw,
            predicted_rtt_ms=rtt,
            uncertainty=uncertainty,
            online_probability=online_prob,
            staleness=staleness,
            num_samples=num_samples,
        )

    def get_retry_penalty_sec(self, client_id: int) -> float:
        failures = self.consecutive_failures.get(client_id, 0)
        if failures == 0:
            return 0.0
        return min(15.0, sum(2**i for i in range(min(failures, 3))))
