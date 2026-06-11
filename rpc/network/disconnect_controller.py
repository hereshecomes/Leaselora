from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("fl_testbed.network.disconnect")

class DisconnectController:

    def __init__(
        self,
        coordinator_ip: str,
        coordinator_port: int,
        worker_id: str = "",
        log_dir: str | Path | None = None,
        use_sudo: bool = True,
    ):
        self.coordinator_ip = coordinator_ip
        self.coordinator_port = coordinator_port
        self.worker_id = worker_id
        self.use_sudo = use_sudo
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.is_hard_offline: bool = False
        self.is_soft_offline: bool = False
        self.offline_start_time: float = 0.0

    def _run(self, cmd: str) -> int:
        prefix = "sudo " if self.use_sudo else ""
        try:
            result = subprocess.run(
                prefix + cmd, shell=True, capture_output=True, text=True, timeout=5,
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            return -1

    def go_hard_offline(self):
        if self.is_hard_offline:
            return

        cmd = (
            f"iptables -A OUTPUT -p tcp -d {self.coordinator_ip} "
            f"--dport {self.coordinator_port} -j DROP"
        )
        rc = self._run(cmd)
        if rc == 0:
            self.is_hard_offline = True
            self.offline_start_time = time.time()
            logger.info(f"HARD OFFLINE: blocked traffic to {self.coordinator_ip}:{self.coordinator_port}")
            self._log_event("network_offline", "hard")
        else:
            logger.error("Failed to apply iptables DROP rule")

    def go_hard_online(self):
        if not self.is_hard_offline:
            return

        cmd = (
            f"iptables -D OUTPUT -p tcp -d {self.coordinator_ip} "
            f"--dport {self.coordinator_port} -j DROP"
        )
        rc = self._run(cmd)
        duration = time.time() - self.offline_start_time
        self.is_hard_offline = False
        logger.info(f"HARD ONLINE: restored traffic (was offline {duration:.1f}s)")
        self._log_event("network_online", "hard", duration)

    def go_soft_offline(self):
        self.is_soft_offline = True
        self.offline_start_time = time.time()
        logger.info("SOFT OFFLINE")
        self._log_event("network_offline", "soft")

    def go_soft_online(self):
        duration = time.time() - self.offline_start_time if self.is_soft_offline else 0
        self.is_soft_offline = False
        logger.info(f"SOFT ONLINE (was offline {duration:.1f}s)")
        self._log_event("network_online", "soft", duration)

    @property
    def is_online(self) -> bool:
        return not self.is_hard_offline and not self.is_soft_offline

    def _log_event(self, event: str, mode: str, duration: float = 0.0):
        if not self.log_dir:
            return
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "worker_id": self.worker_id,
            "event": event,
            "mode": mode,
            "duration_sec": round(duration, 2),
        }
        log_path = self.log_dir / "disconnect_events.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
