from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

from .profile import NetworkProfile
from .tc_manager import TcManager
from .disconnect_controller import DisconnectController

logger = logging.getLogger("fl_testbed.network.trace")

def load_trace(trace_path: str | Path) -> list[dict]:
    rows = []
    with open(trace_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "time_sec": float(row["time_sec"]),
                "rate_mbps": float(row["rate_mbps"]),
                "rtt_ms": float(row["rtt_ms"]),
                "jitter_ms": float(row["jitter_ms"]),
                "loss_percent": float(row["loss_percent"]),
                "online": int(row["online"]),
            })
    return rows

class TraceReplayer:

    def __init__(
        self,
        tc_manager: TcManager,
        disconnect_ctrl: DisconnectController,
        mode: str = "trace",
        trace_path: str | Path | None = None,
        base_profile: NetworkProfile | None = None,
        log_dir: str | Path | None = None,
        worker_id: str = "",
    ):
        self.tc = tc_manager
        self.disconnect = disconnect_ctrl
        self.mode = mode
        self.trace: list[dict] = []
        self.base_profile = base_profile
        self.log_dir = Path(log_dir) if log_dir else None
        self.worker_id = worker_id
        self.current_slot_idx: int = 0
        self.start_time: float = 0.0
        self.current_profile: NetworkProfile | None = None

        if trace_path and mode in ("trace", "repeated"):
            self.trace = load_trace(trace_path)
            logger.info(f"Loaded trace: {trace_path} ({len(self.trace)} slots)")

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        self.start_time = time.time()
        self.current_slot_idx = 0

        if self.mode == "fixed" and self.base_profile:
            self.tc.apply_profile(self.base_profile)
            self.current_profile = self.base_profile
        elif self.mode in ("trace", "repeated") and self.trace:
            self._apply_slot(0)

    def tick(self) -> NetworkProfile | None:
        if self.mode == "fixed":
            return self.current_profile

        elapsed = time.time() - self.start_time

        if self.mode in ("trace", "repeated"):
            return self._tick_trace(elapsed)
        elif self.mode == "random":
            return self._tick_random(elapsed)

        return None

    def _tick_trace(self, elapsed: float) -> NetworkProfile | None:
        if not self.trace:
            return None

        trace_len = self.trace[-1]["time_sec"] if self.trace else 0

        if self.mode == "repeated" and trace_len > 0:
            elapsed = elapsed % (trace_len + 1)

        target_idx = self.current_slot_idx
        for i in range(self.current_slot_idx, len(self.trace)):
            if self.trace[i]["time_sec"] <= elapsed:
                target_idx = i
            else:
                break

        if target_idx != self.current_slot_idx:
            self.current_slot_idx = target_idx
            self._apply_slot(target_idx)
            return self.current_profile

        return None

    def _tick_random(self, elapsed: float) -> NetworkProfile | None:
        import numpy as np
        if self.base_profile is None:
            return None

        rng = np.random.default_rng(int(elapsed))
        p = self.base_profile
        perturbed = NetworkProfile(
            name=f"{p.name}_perturbed",
            rate_mbps=max(1.0, p.rate_mbps * rng.uniform(0.5, 1.5)),
            rtt_ms=max(1.0, p.rtt_ms * rng.uniform(0.7, 1.5)),
            jitter_ms=max(0.0, p.jitter_ms * rng.uniform(0.5, 2.0)),
            loss_percent=min(10.0, max(0.0, p.loss_percent * rng.uniform(0.5, 3.0))),
            loss_correlation=p.loss_correlation,
            reorder_percent=p.reorder_percent,
            online_prob=p.online_prob,
        )
        self.tc.update_netem(perturbed)
        self.current_profile = perturbed
        return perturbed

    def _apply_slot(self, idx: int):
        slot = self.trace[idx]
        online = slot["online"]

        if not online:
            self.disconnect.go_hard_offline()
            self._log_trace_event(slot, "offline")
            self.current_profile = NetworkProfile(
                name="offline", rate_mbps=0, rtt_ms=0, loss_percent=100, online_prob=0,
            )
            return

        if self.disconnect.is_hard_offline:
            self.disconnect.go_hard_online()

        profile = NetworkProfile(
            name=f"trace_slot_{idx}",
            rate_mbps=slot["rate_mbps"],
            rtt_ms=slot["rtt_ms"],
            jitter_ms=slot["jitter_ms"],
            loss_percent=slot["loss_percent"],
            online_prob=1.0,
        )

        if self.tc.state.active:
            self.tc.update_netem(profile)
        else:
            self.tc.apply_profile(profile)

        self.current_profile = profile
        self._log_trace_event(slot, "update")

    def _log_trace_event(self, slot: dict, event: str):
        if not self.log_dir:
            return
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "worker_id": self.worker_id,
            "event": event,
            "elapsed_sec": time.time() - self.start_time,
            "slot": slot,
        }
        log_path = self.log_dir / "network_trace_events.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def stop(self):
        if self.disconnect.is_hard_offline:
            self.disconnect.go_hard_online()
        self.tc.clear()
        logger.info("Trace replayer stopped, tc rules cleared")

    def get_current_state(self) -> dict:
        state = self.tc.get_current_state()
        state["is_online"] = self.disconnect.is_online
        state["trace_mode"] = self.mode
        state["trace_slot_idx"] = self.current_slot_idx
        return state

def main():
    parser = argparse.ArgumentParser(description="Network trace replayer")
    parser.add_argument("--iface", default="eno1")
    parser.add_argument("--coordinator-ip", required=True)
    parser.add_argument("--coordinator-port", type=int, default=29580)
    parser.add_argument("--trace", required=True, help="Path to trace CSV")
    parser.add_argument("--worker-id", default="worker0")
    parser.add_argument("--log-dir", default="logs/testbed/network")
    parser.add_argument("--mode", default="trace", choices=["fixed", "trace", "repeated", "random"])
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tc = TcManager(
        iface=args.iface,
        coordinator_ip=args.coordinator_ip,
        coordinator_port=args.coordinator_port,
        log_dir=args.log_dir,
    )
    dc = DisconnectController(
        coordinator_ip=args.coordinator_ip,
        coordinator_port=args.coordinator_port,
        worker_id=args.worker_id,
        log_dir=args.log_dir,
    )
    replayer = TraceReplayer(
        tc_manager=tc,
        disconnect_ctrl=dc,
        mode=args.mode,
        trace_path=args.trace,
        log_dir=args.log_dir,
        worker_id=args.worker_id,
    )

    replayer.start()
    logger.info("Trace replayer running. Ctrl+C to stop.")

    try:
        while True:
            replayer.tick()
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        replayer.stop()

if __name__ == "__main__":
    main()
