from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("fl_testbed.round_manager")

class RoundLogger:

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.round_log_path = self.log_dir / "round_metrics.jsonl"
        self.client_log_path = self.log_dir / "client_updates.jsonl"
        self.lease_log_path = self.log_dir / "lease_details.jsonl"

    def log_round(self, metrics: dict):
        with open(self.round_log_path, "a") as f:
            f.write(json.dumps(metrics, default=str) + "\n")

    def log_client_update(self, update_info: dict):
        with open(self.client_log_path, "a") as f:
            f.write(json.dumps(update_info, default=str) + "\n")

    def log_lease(self, lease_info: dict):
        with open(self.lease_log_path, "a") as f:
            f.write(json.dumps(lease_info, default=str) + "\n")

class RoundManager:

    def __init__(
        self,
        round_timeout_sec: float = 300.0,
        discard_late_updates: bool = True,
    ):
        self.round_timeout_sec = round_timeout_sec
        self.discard_late_updates = discard_late_updates

    def wait_for_updates(
        self,
        expected_clients: list[int],
        get_pending_updates_fn,
        timeout: float | None = None,
        poll_interval: float = 1.0,
    ) -> tuple[list[dict], list[int], list[int]]:
        if timeout is None:
            timeout = self.round_timeout_sec

        expected_set = set(expected_clients)
        t0 = time.time()
        completed = set()

        while time.time() - t0 < timeout:
            updates = get_pending_updates_fn()
            for u in updates:
                cid = u.get("client_id")
                if cid in expected_set:
                    completed.add(cid)

            if completed == expected_set:
                break
            time.sleep(poll_interval)

        missed = expected_set - completed
        final_updates = [u for u in get_pending_updates_fn() if u.get("client_id") in expected_set]

        if self.discard_late_updates:
            final_updates = [u for u in final_updates if u.get("client_id") in completed]

        return final_updates, sorted(completed), sorted(missed)
