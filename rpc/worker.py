from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import yaml

from .peft_exec.model_factory import create_model, DATASET_META
from .peft_exec.lora_utils import (
    build_slice_registry, get_lora_state_dict, load_lora_state_dict,
    apply_trainable_mask,
)
from .peft_exec.delta_utils import extract_lora_delta
from .peft_exec.local_trainer import local_train
from .rpc.client_api import CoordinatorClient
from .rpc.serialization import measure_serialization
from .data.client_dataset import load_full_dataset, get_client_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("fl_testbed.worker")

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_nodes_config(nodes_path: str) -> dict:
    with open(nodes_path) as f:
        return yaml.safe_load(f)

def get_worker_info(nodes_cfg: dict, worker_id: str) -> dict:
    for node_id, node_info in nodes_cfg.get("nodes", {}).items():
        for w in node_info.get("workers", []):
            if w["worker_id"] == worker_id:
                return {
                    "node_id": node_id,
                    "host": node_info.get("host", "localhost"),
                    **w,
                }
    return {"node_id": "unknown", "worker_id": worker_id, "cuda_device": 0, "client_ids": []}

def run_worker(
    config_path: str,
    nodes_path: str | None,
    worker_id: str,
    coordinator_url: str,
    cuda_device: int | None = None,
    node_id: str | None = None,
):
    cfg = load_config(config_path)
    nodes_cfg = load_nodes_config(nodes_path) if nodes_path else {}

    worker_info = get_worker_info(nodes_cfg, worker_id)
    if cuda_device is None:
        cuda_device = worker_info.get("cuda_device", 0)
    if node_id is None:
        node_id = worker_info.get("node_id", "unknown")
    client_ids = worker_info.get("client_ids", [])

    if not client_ids:
        num_clients = cfg.get("fl", {}).get("num_clients", 4)
        client_ids = list(range(num_clients))
        logger.info(f"No nodes config for {worker_id}, defaulting to all {num_clients} clients")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    device = torch.device("cuda:0")

    logger.info(f"Worker {worker_id} starting on node={node_id}, cuda={cuda_device}")
    logger.info(f"Assigned clients: {client_ids}")
    logger.info(f"Coordinator: {coordinator_url}")

    model_cfg = cfg.get("model", {})
    fl_cfg = cfg.get("fl", {})
    exp_cfg = cfg.get("experiment", {})
    sched_cfg = cfg.get("scheduler", {})

    method = exp_cfg.get("method", "leaselora")
    dataset_name = model_cfg.get("dataset", exp_cfg.get("name", "sst2").split("_")[0])
    meta = DATASET_META.get(dataset_name, {"num_labels": 2, "task": "text_classification"})
    backbone = model_cfg.get("backbone", "distilbert-base-uncased")
    lora_cfg = model_cfg.get("lora", {})

    logger.info(f"Loading model: {backbone}")
    model, processor = create_model(
        backbone=backbone,
        num_labels=meta["num_labels"],
        task_type=meta["task"],
        lora_rank=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules"),
        precision=model_cfg.get("precision", "bf16"),
    )

    slice_registry = build_slice_registry(model, bundle_size=sched_cfg.get("bundle_size", 1))
    logger.info(f"Model loaded. Slices: {len(slice_registry)}")

    partition_cfg = fl_cfg.get("partition", {})
    partition_dir = partition_cfg.get("dir", f"partitions/{dataset_name}_alpha{partition_cfg.get('alpha', 0.5)}_clients{fl_cfg.get('num_clients', 20)}")

    logger.info(f"Loading dataset: {dataset_name}")
    full_dataset = load_full_dataset(dataset_name, split="train", tokenizer=processor, processor=processor)
    logger.info(f"Dataset loaded: {len(full_dataset)} samples total")

    log_dir = Path(exp_cfg.get("log_dir", "logs/testbed/default"))
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_log_path = log_dir / f"worker_{worker_id}.jsonl"

    def write_worker_log(event: dict):
        event["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        event["worker_id"] = worker_id
        with open(worker_log_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    net_cfg = cfg.get("network_emulation", cfg.get("network", {}))
    tc_manager = None
    trace_replayer = None
    disconnect_ctrl = None

    if net_cfg.get("enabled", False):
        from .network.tc_manager import TcManager
        from .network.disconnect_controller import DisconnectController
        from .network.trace_replayer import TraceReplayer
        from .network.profile import BUILTIN_PROFILES, load_profiles_from_yaml, assign_profiles_to_clients

        net_iface = net_cfg.get("iface", "eno1")
        coord_ip = coordinator_url.split("://")[1].split(":")[0] if "://" in coordinator_url else "127.0.0.1"
        coord_port = int(coordinator_url.split(":")[-1].rstrip("/")) if ":" in coordinator_url else 29580

        tc_manager = TcManager(
            iface=net_iface,
            coordinator_ip=coord_ip,
            coordinator_port=coord_port,
            log_dir=str(log_dir),
            use_sudo=net_cfg.get("use_sudo", True),
        )
        disconnect_ctrl = DisconnectController(
            coordinator_ip=coord_ip,
            coordinator_port=coord_port,
            worker_id=worker_id,
            log_dir=str(log_dir),
        )

        net_mode = net_cfg.get("mode", "fixed")
        trace_path = net_cfg.get("trace_path")

        profiles_cfg_path = net_cfg.get("profiles_config")
        if profiles_cfg_path:
            all_profiles = load_profiles_from_yaml(profiles_cfg_path)
        else:
            all_profiles = BUILTIN_PROFILES

        worker_profile_name = net_cfg.get("worker_profile")
        if worker_profile_name and worker_profile_name in all_profiles:
            base_profile = all_profiles[worker_profile_name]
        elif client_ids:
            profile_names = list(all_profiles.keys())
            assignments = assign_profiles_to_clients(fl_cfg.get("num_clients", 20), profile_names)
            first_client_profile = assignments.get(client_ids[0], "strong_lan")
            base_profile = all_profiles.get(first_client_profile, BUILTIN_PROFILES["strong_lan"])
        else:
            base_profile = BUILTIN_PROFILES["strong_lan"]

        trace_replayer = TraceReplayer(
            tc_manager=tc_manager,
            disconnect_ctrl=disconnect_ctrl,
            mode=net_mode,
            trace_path=trace_path,
            base_profile=base_profile,
            log_dir=str(log_dir),
            worker_id=worker_id,
        )
        trace_replayer.start()
        logger.info(f"Network emulation active: mode={net_mode}, profile={base_profile.name}")

    client = CoordinatorClient(coordinator_url)

    for attempt in range(30):
        try:
            health = client.health_check()
            logger.info(f"Coordinator health: {health}")
            break
        except Exception as e:
            logger.info(f"Waiting for coordinator... ({e})")
            time.sleep(3)
    else:
        logger.error("Could not connect to coordinator")
        return

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0

    client.register(
        worker_id=worker_id,
        node_id=node_id,
        gpu_id=cuda_device,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_mem,
        logical_client_ids=client_ids,
    )
    logger.info(f"Registered with coordinator")
    write_worker_log({"event": "registered", "client_ids": client_ids, "gpu_name": gpu_name})

    local_cfg = fl_cfg.get("local", {})
    lr = local_cfg.get("lr", 3e-4)
    weight_decay = local_cfg.get("weight_decay", 0.01)
    max_grad_norm = local_cfg.get("max_grad_norm", 1.0)

    idle_count = 0
    while True:
        if trace_replayer:
            trace_replayer.tick()

        try:
            free_mem = torch.cuda.mem_get_info(0)[0] / (1024**3) if torch.cuda.is_available() else 0
            is_online = disconnect_ctrl.is_online if disconnect_ctrl else True
            client.heartbeat(
                worker_id=worker_id,
                free_gpu_memory_gb=free_mem,
                online=is_online,
            )
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")

        try:
            task = client.get_task(worker_id)
        except Exception as e:
            logger.warning(f"Failed to get task: {e}")
            time.sleep(5)
            idle_count += 1
            if idle_count > 60:
                logger.info("Lost coordinator connection for too long, exiting")
                write_worker_log({"event": "exit_lost_connection"})
                break
            continue

        task_type = task.get("task_type", "idle")

        if task_type == "stop":
            logger.info("Received stop signal")
            write_worker_log({"event": "stopped"})
            break

        if task_type == "idle":
            idle_count += 1
            if idle_count % 10 == 0:
                logger.debug(f"Idle... (poll #{idle_count})")
            time.sleep(2)
            continue

        idle_count = 0

        if task_type == "train":
            tasks_to_run = task if isinstance(task, list) else [task]
            for single_task in tasks_to_run:
                _execute_train_task(
                    single_task, model, full_dataset, partition_dir, slice_registry,
                    client, device, lr, weight_decay, max_grad_norm,
                    method, worker_id, write_worker_log, cfg,
                )

def _execute_train_task(
    task: dict,
    model,
    full_dataset,
    partition_dir: str,
    slice_registry,
    coordinator_client: CoordinatorClient,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    method: str,
    worker_id: str,
    write_worker_log,
    cfg: dict | None = None,
):
    round_id = task["round"]
    client_id = task["client_id"]
    lease = task.get("lease", {})
    lease_method = lease.get("method", method)

    logger.info(f"Task: round={round_id}, client={client_id}, method={lease_method}")
    write_worker_log({"event": "task_start", "round": round_id, "client_id": client_id})

    t_task_start = time.time()

    try:
        global_state, download_time, download_bytes = coordinator_client.download_global_adapter()
        logger.info(f"Downloaded global adapter: {download_bytes} bytes in {download_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to download global adapter: {e}")
        write_worker_log({"event": "error", "round": round_id, "error": str(e)})
        return

    load_lora_state_dict(model, global_state)

    param_names = lease.get("param_names", [])
    apply_trainable_mask(model, param_names, mode="leaselora")

    local_steps = lease.get("local_steps", 24)
    batch_size = lease.get("micro_batch_size", 16)

    try:
        from .data.partition import load_partition
        indices = load_partition(partition_dir, client_id)
        from torch.utils.data import Subset, DataLoader
        subset = Subset(full_dataset, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)
    except FileNotFoundError:
        logger.warning(f"Partition not found for client {client_id}, using full dataset subset")
        n = len(full_dataset)
        step = n // 20
        indices = list(range(client_id * step, min((client_id + 1) * step, n)))
        from torch.utils.data import Subset, DataLoader
        subset = Subset(full_dataset, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)

    result = local_train(
        model=model,
        dataloader=dataloader,
        local_steps=local_steps,
        lr=lr,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        device=device,
    )

    logger.info(f"Training done: loss={result.local_loss:.4f}, steps={result.actual_steps}, "
                f"time={result.train_time_sec:.2f}s, peak_mem={result.peak_gpu_memory_gb:.2f}GB")

    compute_slowdown = lease.get("compute_slowdown", 1.0)
    simulated_step_time = lease.get("simulated_step_time_sec", 0.0)
    actual_gpu_time = result.train_time_sec

    if simulated_step_time > 0:
        effective_device_time = simulated_step_time * local_steps * compute_slowdown
        result.train_time_sec = effective_device_time
        deadline_sec_check = lease.get("deadline_seconds", 999999)
        if effective_device_time > deadline_sec_check:
            logger.warning(
                f"Virtual deadline MISS: {effective_device_time:.1f}s > "
                f"{deadline_sec_check:.1f}s (slowdown={compute_slowdown:.1f}x, "
                f"steps={local_steps}, step_time={simulated_step_time:.2f}s)")
        else:
            logger.info(
                f"Virtual device time {effective_device_time:.1f}s "
                f"(gpu={actual_gpu_time:.1f}s, slowdown={compute_slowdown:.1f}x)")
    elif compute_slowdown > 1.0:
        result.train_time_sec = actual_gpu_time * compute_slowdown

    local_state = get_lora_state_dict(model)
    delta, num_params, _ = extract_lora_delta(local_state, global_state, param_names)

    serialized_bytes, serialization_time, payload_bytes = measure_serialization(delta)

    deadline_sec = lease.get("deadline_seconds", 999999)
    pre_upload_time = time.time() - t_task_start
    if simulated_step_time > 0 and compute_slowdown > 0:
        effective_total = simulated_step_time * local_steps * compute_slowdown
        if effective_total > pre_upload_time:
            pre_upload_time = effective_total
    pre_upload_miss = pre_upload_time > deadline_sec

    metadata = {
        "round": round_id,
        "worker_id": worker_id,
        "client_id": client_id,
        "method": lease_method,
        "num_examples": result.num_examples,
        "local_loss": result.local_loss,
        "local_steps": result.actual_steps,
        "train_time_sec": result.train_time_sec,
        "actual_gpu_time_sec": actual_gpu_time,
        "compute_slowdown": compute_slowdown,
        "serialization_time_sec": serialization_time,
        "peak_gpu_memory_gb": result.peak_gpu_memory_gb,
        "trained_param_names": list(delta.keys()),
        "trained_slice_ids": lease.get("slice_ids", []),
        "pre_upload_time_sec": pre_upload_time,
        "deadline_seconds": deadline_sec,
        "deadline_miss": pre_upload_miss,
    }

    from .network.measure import measure_upload_with_retry

    def _upload_fn():
        return coordinator_client.upload_update(metadata, delta)

    _cfg = cfg or {}
    upload_cfg = _cfg.get("network_emulation", {}).get("upload_retry", {})
    max_retries = upload_cfg.get("max_retries", 3)
    retry_backoff = upload_cfg.get("retry_backoff_sec", [1.0, 2.0, 4.0])
    upload_timeout = upload_cfg.get("upload_timeout_sec", lease.get("deadline_seconds", 60.0))

    upload_result, measurements = measure_upload_with_retry(
        _upload_fn, max_retries=max_retries,
        retry_backoff_sec=retry_backoff, upload_timeout_sec=upload_timeout,
    )

    for m in measurements:
        write_worker_log({
            "event": "upload_attempt",
            "round": round_id,
            "client_id": client_id,
            "attempt": m.attempt,
            "payload_bytes": m.payload_bytes,
            "upload_time_sec": m.upload_time_sec,
            "success": m.success,
            "error": m.error,
            "measured_throughput_mbps": m.measured_throughput_mbps,
        })

    if upload_result is None:
        logger.error(f"Upload failed after {max_retries} attempts")
        write_worker_log({"event": "upload_failed", "round": round_id, "client_id": client_id})
        torch.cuda.empty_cache()
        return

    successful_m = next((m for m in measurements if m.success), None)
    upload_time = successful_m.upload_time_sec if successful_m else 0
    actual_bytes = successful_m.payload_bytes if successful_m else payload_bytes
    num_attempts = len(measurements)

    logger.info(f"Upload done: {actual_bytes} bytes in {upload_time:.3f}s "
                f"({actual_bytes / max(upload_time, 0.001) / 1e6:.1f} MB/s, "
                f"attempts={num_attempts})")

    total_task_time = time.time() - t_task_start
    deadline_miss = total_task_time > deadline_sec

    write_worker_log({
        "event": "task_done",
        "round": round_id,
        "client_id": client_id,
        "method": lease_method,
        "gpu_id": int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
        "local_steps": result.actual_steps,
        "num_examples": result.num_examples,
        "train_time_sec": result.train_time_sec,
        "actual_gpu_time_sec": actual_gpu_time,
        "compute_slowdown": compute_slowdown,
        "download_time_sec": download_time,
        "serialization_time_sec": serialization_time,
        "upload_time_sec": upload_time,
        "payload_bytes": actual_bytes,
        "peak_gpu_memory_gb": result.peak_gpu_memory_gb,
        "deadline_seconds": deadline_sec,
        "deadline_miss": deadline_miss,
        "total_task_time_sec": total_task_time,
        "local_loss": result.local_loss,
        "trained_param_names": list(delta.keys()),
        "measured_lan_upload_mbps": actual_bytes / max(upload_time, 0.001) / (1024 * 1024),
    })

    torch.cuda.empty_cache()

    if deadline_miss:
        logger.warning(f"DEADLINE MISS: task took {total_task_time:.1f}s > {deadline_sec:.1f}s "
                        f"(slowdown={compute_slowdown:.1f}x, gpu_time={actual_gpu_time:.1f}s)")

def main():
    parser = argparse.ArgumentParser(description="FL Testbed Worker")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--nodes", default=None, help="Nodes config YAML")
    parser.add_argument("--worker_id", required=True, help="Unique worker identifier")
    parser.add_argument("--coordinator", required=True, help="Coordinator URL (http://host:port)")
    parser.add_argument("--cuda_device", type=int, default=None, help="CUDA device index")
    parser.add_argument("--node_id", default=None, help="Node identifier")
    args = parser.parse_args()

    run_worker(
        config_path=args.config,
        nodes_path=args.nodes,
        worker_id=args.worker_id,
        coordinator_url=args.coordinator,
        cuda_device=args.cuda_device,
        node_id=args.node_id,
    )

if __name__ == "__main__":
    main()
