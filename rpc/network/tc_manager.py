from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .profile import NetworkProfile

logger = logging.getLogger("fl_testbed.network.tc")

@dataclass
class TcState:
    iface: str
    coordinator_ip: str
    coordinator_port: int
    profile: NetworkProfile | None = None
    class_id: int = 10
    active: bool = False
    last_applied_at: float = 0.0

class TcManager:

    def __init__(
        self,
        iface: str = "eno1",
        coordinator_ip: str = "127.0.0.1",
        coordinator_port: int = 29580,
        class_id: int = 10,
        log_dir: str | Path | None = None,
        use_sudo: bool = True,
    ):
        self.iface = iface
        self.coordinator_ip = coordinator_ip
        self.coordinator_port = coordinator_port
        self.class_id = class_id
        self.use_sudo = use_sudo
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.state = TcState(
            iface=iface,
            coordinator_ip=coordinator_ip,
            coordinator_port=coordinator_port,
            class_id=class_id,
        )

    def _run(self, cmd: str, check: bool = False) -> tuple[int, str]:
        prefix = "sudo " if self.use_sudo else ""
        full_cmd = prefix + cmd
        try:
            result = subprocess.run(
                full_cmd, shell=True, capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0 and check:
                logger.warning(f"tc command failed: {full_cmd}\n  stderr: {result.stderr.strip()}")
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"tc command timeout: {full_cmd}")
            return -1, "timeout"

    def clear(self):
        self._run(f"tc qdisc del dev {self.iface} root", check=False)
        self.state.active = False
        self.state.profile = None
        logger.info(f"Cleared tc rules on {self.iface}")

    def apply_profile(self, profile: NetworkProfile):
        self.clear()

        iface = self.iface
        cid = self.class_id
        rate = profile.rate_mbps
        delay = profile.delay_ms
        jitter = profile.jitter_ms
        loss = profile.loss_percent
        loss_corr = profile.loss_correlation
        reorder = profile.reorder_percent
        dup = profile.duplicate_percent
        limit = profile.queue_limit_packets

        cmds = [
            f"tc qdisc replace dev {iface} root handle 1: htb default 999",
            f"tc class replace dev {iface} parent 1: classid 1:999 htb rate 10gbit ceil 10gbit",
            f"tc class replace dev {iface} parent 1: classid 1:{cid} htb rate {rate}mbit ceil {rate}mbit burst 64kbit cburst 64kbit",
        ]

        netem_args = f"delay {delay}ms {jitter}ms distribution normal"
        if loss > 0:
            netem_args += f" loss {loss}% {loss_corr}%"
        if reorder > 0:
            netem_args += f" reorder {reorder}%"
        if dup > 0:
            netem_args += f" duplicate {dup}%"
        netem_args += f" limit {limit}"

        cmds.append(
            f"tc qdisc replace dev {iface} parent 1:{cid} handle {cid}0: netem {netem_args}"
        )

        filter_cmd = (
            f"tc filter replace dev {iface} protocol ip parent 1: prio 1 "
            f"flower ip_proto tcp dst_ip {self.coordinator_ip} "
            f"dst_port {self.coordinator_port} classid 1:{cid}"
        )
        cmds.append(filter_cmd)

        for cmd in cmds:
            rc, output = self._run(cmd, check=True)
            if rc != 0 and "flower" in cmd:
                logger.info("flower filter failed, falling back to u32")
                u32_cmd = self._build_u32_filter()
                self._run(u32_cmd, check=True)

        self.state.active = True
        self.state.profile = profile
        self.state.last_applied_at = time.time()

        logger.info(f"Applied profile '{profile.name}' on {iface}: "
                    f"rate={rate}mbit, delay={delay}ms±{jitter}ms, "
                    f"loss={loss}%, reorder={reorder}%")

        self._log_tc_state(profile)

    def _build_u32_filter(self) -> str:
        ip_parts = self.coordinator_ip.split(".")
        ip_hex = "".join(f"{int(p):02x}" for p in ip_parts)
        port_hex = f"{self.coordinator_port:04x}"

        return (
            f"tc filter replace dev {self.iface} protocol ip parent 1: prio 1 u32 "
            f"match ip dst {self.coordinator_ip}/32 "
            f"match ip dport {self.coordinator_port} 0xffff "
            f"classid 1:{self.class_id}"
        )

    def update_netem(self, profile: NetworkProfile):
        cid = self.class_id
        delay = profile.delay_ms
        jitter = profile.jitter_ms
        loss = profile.loss_percent
        loss_corr = profile.loss_correlation
        reorder = profile.reorder_percent
        dup = profile.duplicate_percent
        limit = profile.queue_limit_packets

        netem_args = f"delay {delay}ms {jitter}ms distribution normal"
        if loss > 0:
            netem_args += f" loss {loss}% {loss_corr}%"
        if reorder > 0:
            netem_args += f" reorder {reorder}%"
        if dup > 0:
            netem_args += f" duplicate {dup}%"
        netem_args += f" limit {limit}"

        cmd = f"tc qdisc change dev {self.iface} parent 1:{cid} handle {cid}0: netem {netem_args}"
        rc, _ = self._run(cmd, check=True)

        if rc == 0:
            rate = profile.rate_mbps
            rate_cmd = (
                f"tc class change dev {self.iface} parent 1: classid 1:{cid} "
                f"htb rate {rate}mbit ceil {rate}mbit burst 64kbit cburst 64kbit"
            )
            self._run(rate_cmd, check=True)
            self.state.profile = profile
            self.state.last_applied_at = time.time()
            logger.debug(f"Updated netem: rate={rate}mbit, delay={delay}ms, loss={loss}%")

    def get_current_state(self) -> dict:
        if self.state.profile:
            return {
                "network_profile": self.state.profile.name,
                "rate_mbps": self.state.profile.rate_mbps,
                "rtt_ms": self.state.profile.rtt_ms,
                "jitter_ms": self.state.profile.jitter_ms,
                "loss_percent": self.state.profile.loss_percent,
                "online_prob": self.state.profile.online_prob,
                "tc_active": self.state.active,
            }
        return {"network_profile": "none", "tc_active": False}

    def _log_tc_state(self, profile: NetworkProfile):
        if not self.log_dir:
            return
        _, qdisc_out = self._run(f"tc -s qdisc show dev {self.iface}")
        _, class_out = self._run(f"tc -s class show dev {self.iface}")
        _, filter_out = self._run(f"tc filter show dev {self.iface}")

        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "profile": profile.to_dict(),
            "qdisc": qdisc_out.strip(),
            "class": class_out.strip(),
            "filter": filter_out.strip(),
        }

        log_path = self.log_dir / "tc_state.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
