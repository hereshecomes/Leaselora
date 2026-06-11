from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from .serialization import deserialize_delta, serialize_adapter_state

logger = logging.getLogger("fl_testbed.coordinator")

app = FastAPI(title="LeaseLoRA FL Coordinator")

class CoordinatorState:

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.workers: dict[str, dict] = {}
        self.heartbeats: dict[str, dict] = {}
        self.current_round: int = 0
        self.round_tasks: dict[str, list] = {}
        self.pending_updates: list[dict] = []
        self.global_adapter_bytes: bytes = b""
        self.global_adapter_state: dict[str, Any] = {}
        self.is_ready: bool = False
        self.stopped: bool = False
        self.round_event_callback = None

state = CoordinatorState()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "registered_workers": len(state.workers),
        "current_round": state.current_round,
        "is_ready": state.is_ready,
    }

@app.post("/register")
async def register_worker(request: Request):
    data = await request.json()
    worker_id = data["worker_id"]
    state.workers[worker_id] = {
        "worker_id": worker_id,
        "node_id": data.get("node_id", "unknown"),
        "gpu_id": data.get("gpu_id", 0),
        "gpu_name": data.get("gpu_name", "unknown"),
        "gpu_memory_gb": data.get("gpu_memory_gb", 0),
        "logical_client_ids": data.get("logical_client_ids", []),
        "resource_profile": data.get("resource_profile", {}),
        "registered_at": time.time(),
        "last_heartbeat": time.time(),
        "online": True,
    }
    logger.info(f"Worker registered: {worker_id} (node={data.get('node_id')}, "
                f"gpu={data.get('gpu_id')}, clients={data.get('logical_client_ids')})")
    return {"status": "ok", "worker_id": worker_id}

@app.post("/heartbeat")
async def heartbeat(request: Request):
    data = await request.json()
    worker_id = data["worker_id"]
    state.heartbeats[worker_id] = {
        **data,
        "timestamp": time.time(),
    }
    if worker_id in state.workers:
        state.workers[worker_id]["last_heartbeat"] = time.time()
        state.workers[worker_id]["online"] = data.get("online", True)
    return {"status": "ok", "current_round": state.current_round}

@app.get("/task")
async def get_task(worker_id: str):
    if state.stopped:
        return {"task_type": "stop", "round": state.current_round}

    with state.lock:
        tasks = state.round_tasks.get(worker_id)
        if not tasks:
            return {"task_type": "idle", "round": state.current_round}

        if isinstance(tasks, list) and len(tasks) > 0:
            task = tasks.pop(0)
            if not tasks:
                del state.round_tasks[worker_id]
            return task

        task = tasks
        del state.round_tasks[worker_id]
        return task

@app.get("/global_adapter")
async def get_global_adapter():
    if not state.global_adapter_bytes:
        raise HTTPException(status_code=404, detail="No global adapter available")
    return Response(
        content=state.global_adapter_bytes,
        media_type="application/octet-stream",
    )

@app.post("/upload_update")
async def upload_update(request: Request):
    content_type = request.headers.get("content-type", "")

    if "multipart" in content_type or "octet-stream" in content_type:
        metadata_json = request.headers.get("x-metadata", "{}")
        metadata = json.loads(metadata_json)
        delta_bytes = await request.body()
    else:
        form = await request.form()
        metadata = json.loads(form["metadata"])
        delta_file = form["delta"]
        delta_bytes = await delta_file.read()

    delta = deserialize_delta(delta_bytes)

    update_record = {
        **metadata,
        "delta": delta,
        "actual_payload_bytes": len(delta_bytes),
        "received_at": time.time(),
    }
    with state.lock:
        existing = {(u.get("round"), u.get("client_id")) for u in state.pending_updates}
        key = (metadata.get("round"), metadata.get("client_id"))
        if key not in existing:
            state.pending_updates.append(update_record)
        else:
            logger.info(f"Duplicate update ignored: round={key[0]}, client={key[1]}")

    logger.info(
        f"Update received: round={metadata.get('round')}, "
        f"client={metadata.get('client_id')}, worker={metadata.get('worker_id')}, "
        f"payload={len(delta_bytes)} bytes"
    )
    return {"status": "ok", "received_bytes": len(delta_bytes)}

@app.post("/finish")
async def finish():
    state.stopped = True
    return {"status": "ok"}

def set_coordinator_state(coordinator_state: CoordinatorState):
    global state
    state = coordinator_state

def run_server(host: str = "0.0.0.0", port: int = 29580):
    uvicorn.run(app, host=host, port=port, log_level="info")
