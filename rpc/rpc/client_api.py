from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from .serialization import serialize_delta, deserialize_adapter_state

logger = logging.getLogger("fl_testbed.worker")

class CoordinatorClient:

    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> dict:
        resp = self.session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def register(
        self,
        worker_id: str,
        node_id: str,
        gpu_id: int,
        gpu_name: str,
        gpu_memory_gb: float,
        logical_client_ids: list[int],
        resource_profile: dict | None = None,
    ) -> dict:
        payload = {
            "worker_id": worker_id,
            "node_id": node_id,
            "gpu_id": gpu_id,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "logical_client_ids": logical_client_ids,
            "resource_profile": resource_profile or {},
        }
        resp = self.session.post(
            f"{self.base_url}/register", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def heartbeat(
        self,
        worker_id: str,
        free_gpu_memory_gb: float = 0.0,
        peak_gpu_memory_gb: float = 0.0,
        throughput_estimate: float = 0.0,
        bandwidth_estimate_mbps: float = 0.0,
        online: bool = True,
    ) -> dict:
        payload = {
            "worker_id": worker_id,
            "free_gpu_memory_gb": free_gpu_memory_gb,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
            "throughput_estimate": throughput_estimate,
            "bandwidth_estimate_mbps": bandwidth_estimate_mbps,
            "online": online,
        }
        resp = self.session.post(
            f"{self.base_url}/heartbeat", json=payload, timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def get_task(self, worker_id: str) -> dict:
        resp = self.session.get(
            f"{self.base_url}/task", params={"worker_id": worker_id}, timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def download_global_adapter(self) -> tuple[dict, float, int]:
        t0 = time.time()
        resp = self.session.get(
            f"{self.base_url}/global_adapter", timeout=self.timeout
        )
        resp.raise_for_status()
        download_time = time.time() - t0
        data = resp.content
        state = deserialize_adapter_state(data)
        return state, download_time, len(data)

    def upload_update(
        self,
        metadata: dict,
        delta: dict,
    ) -> tuple[dict, float, int]:
        delta_bytes = serialize_delta(delta)
        payload_bytes = len(delta_bytes)

        metadata_json = json.dumps(metadata)

        t0 = time.time()
        resp = self.session.post(
            f"{self.base_url}/upload_update",
            data=delta_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Metadata": metadata_json,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        upload_time = time.time() - t0

        return resp.json(), upload_time, payload_bytes

    def poll_task(self, worker_id: str, interval: float = 2.0, max_wait: float = 300.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < max_wait:
            task = self.get_task(worker_id)
            if task.get("task_type") != "idle":
                return task
            time.sleep(interval)
        return {"task_type": "idle", "round": -1}
