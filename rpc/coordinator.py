from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .peft_exec.model_factory import create_model, DATASET_META
from .peft_exec.lora_utils import (
    build_slice_registry, get_lora_state_dict, load_lora_state_dict,
)
from .peft_exec.aggregation import ClientUpdate, aggregate_partial_deltas
from .rpc.server_api import CoordinatorState, set_coordinator_state, app, run_server
from .rpc.serialization import serialize_adapter_state
from .runtime.client_sampler import ClientSampler
from .runtime.round_manager import RoundManager, RoundLogger
from .runtime.checkpoint_manager import CheckpointManager
from .runtime.evaluator import evaluate_model
from .scheduler.resource_state import (
    assign_client_budgets, build_client_resource_states, ClientResourceState,
)
from .scheduler.lease_planner import LeasePlanner
from .scheduler.feasibility_guard import FeasibilityGuard
from .scheduler.feedback_state import FeedbackState
from .network.bandwidth_predictor import BandwidthPredictor, UploadRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("fl_testbed.coordinator")

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_nodes_config(nodes_path: str) -> dict:
    with open(nodes_path) as f:
        return yaml.safe_load(f)

def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

def get_available_clients_from_workers(state: CoordinatorState) -> list[int]:
    clients = []
    for wid, info in state.workers.items():
        if info.get("online", True):
            clients.extend(info.get("logical_client_ids", []))
    return sorted(int(c) for c in set(clients))

def get_worker_for_client(state: CoordinatorState, client_id: int) -> str | None:
    for wid, info in state.workers.items():
        if client_id in info.get("logical_client_ids", []):
            return wid
    return None

def run_coordinator(config_path: str, nodes_path: str, host: str, port: int, resume: bool = False):
    cfg = load_config(config_path)
    nodes_cfg = load_nodes_config(nodes_path) if nodes_path else {}

    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    fl_cfg = cfg.get("fl", {})
    sched_cfg = cfg.get("scheduler", {})
    runtime_cfg = cfg.get("runtime", {})
    resource_cfg = cfg.get("resources", {})
    log_cfg = cfg.get("logging", {})

    seed = exp_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    method = exp_cfg.get("method", "leaselora")
    num_rounds = exp_cfg.get("num_rounds", 100)
    eval_every = exp_cfg.get("eval_every", 5)
    num_clients = fl_cfg.get("num_clients", 20)
    clients_per_round = fl_cfg.get("clients_per_round", 5)

    log_dir = Path(exp_cfg.get("log_dir", f"logs/testbed/{exp_cfg.get('name', 'default')}"))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(exp_cfg.get("checkpoint_dir", f"checkpoints/testbed/{exp_cfg.get('name', 'default')}"))

    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    with open(log_dir / "git_hash.txt", "w") as f:
        f.write(get_git_hash())

    logger.info(f"Experiment: {exp_cfg.get('name', 'unnamed')}")
    logger.info(f"Method: {method}, Rounds: {num_rounds}, Clients: {num_clients}")
    logger.info(f"Log dir: {log_dir}")

    dataset_name = model_cfg.get("dataset", exp_cfg.get("name", "sst2").split("_")[0])
    meta = DATASET_META.get(dataset_name, {"num_labels": 2, "task": "text_classification"})
    backbone = model_cfg.get("backbone", "distilbert-base-uncased")
    lora_cfg = model_cfg.get("lora", {})

    logger.info(f"Building model: {backbone} for {dataset_name}")
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

    bundle_size = sched_cfg.get("bundle_size", 1)
    slice_registry = build_slice_registry(model, bundle_size=bundle_size)
    logger.info(f"Slice registry: {len(slice_registry)} slices, bundle_size={bundle_size}")
    for sid, params in slice_registry.items():
        logger.info(f"  {sid}: {len(params)} params")

    global_adapter = get_lora_state_dict(model)
    logger.info(f"Global adapter: {len(global_adapter)} parameters, "
                f"{sum(t.numel() for t in global_adapter.values())} scalars")

    budget_dist = [t["budget"] for t in resource_cfg.get("tiers", [{"budget": 1.0}])]
    budget_assignments = assign_client_budgets(num_clients, budget_dist, seed)
    client_resources = build_client_resource_states(
        num_clients, budget_assignments, resource_cfg.get("tiers"),
    )

    net_pred_cfg = cfg.get("network_predictor", cfg.get("network_emulation", {}).get("predictor", {}))
    bw_predictor = BandwidthPredictor(
        beta=net_pred_cfg.get("beta", 0.3),
        min_samples=net_pred_cfg.get("min_samples", 3),
        failure_penalty=net_pred_cfg.get("failure_penalty", 0.5),
        stale_after_rounds=net_pred_cfg.get("stale_after_rounds", 3),
        default_uplink_mbps=net_pred_cfg.get("default_uplink_mbps", 80.0),
        default_rtt_ms=net_pred_cfg.get("default_rtt_ms", 50.0),
    )

    guard = FeasibilityGuard(
        shrink_policy=sched_cfg.get("shrink_policy", ["slices", "local_steps", "micro_batch"]),
        bandwidth_predictor=bw_predictor,
        online_threshold=sched_cfg.get("online_threshold", 0.5),
        guard_mode=sched_cfg.get("guard_mode", "default"),
    )
    feedback = FeedbackState(
        slice_ids=list(slice_registry.keys()),
        enable_coverage=sched_cfg.get("enable_coverage_feedback", True),
        enable_service=sched_cfg.get("enable_service_feedback", True),
        enable_variance=sched_cfg.get("enable_variance_feedback", True),
    )
    planner = LeasePlanner(
        slice_registry=slice_registry,
        method=method,
        default_local_steps=fl_cfg.get("local", {}).get("local_steps", 24),
        default_micro_batch=fl_cfg.get("local", {}).get("batch_size", 16),
        default_deadline_sec=sched_cfg.get("base_deadline_sec", 120.0),
        deadline_factor=sched_cfg.get("deadline_factor", 1.2),
        warmup_full_rounds=sched_cfg.get("warmup_full_rounds", 5),
        guard=guard,
    )
    sampler = ClientSampler(
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        seed=seed,
        enable_weak_client_reservation=sched_cfg.get("enable_weak_client_reservation", False),
        weak_client_threshold=sched_cfg.get("weak_client_threshold", 0.5),
    )
    round_mgr = RoundManager(
        round_timeout_sec=runtime_cfg.get("round_timeout_sec", 300.0),
        discard_late_updates=runtime_cfg.get("discard_late_updates", True),
    )
    round_logger = RoundLogger(log_dir)
    ckpt_mgr = CheckpointManager(checkpoint_dir)

    coord_state = CoordinatorState()
    coord_state.global_adapter_state = global_adapter
    coord_state.global_adapter_bytes = serialize_adapter_state(global_adapter)
    coord_state.is_ready = True
    set_coordinator_state(coord_state)

    server_thread = threading.Thread(
        target=run_server, args=(host, port), daemon=True
    )
    server_thread.start()
    logger.info(f"Coordinator server started on {host}:{port}")

    start_round = 0
    if resume:
        ckpt = ckpt_mgr.load_latest()
        if ckpt:
            start_round = ckpt["round_id"] + 1
            global_adapter = ckpt["global_adapter_state"]
            coord_state.global_adapter_state = global_adapter
            coord_state.global_adapter_bytes = serialize_adapter_state(global_adapter)
            if "feedback_state" in ckpt:
                feedback.load_dict(ckpt["feedback_state"])
            logger.info(f"Resumed from round {start_round}")

    min_workers = runtime_cfg.get("min_workers", 1)
    wait_timeout = runtime_cfg.get("worker_wait_timeout_sec", 120)
    logger.info(f"Waiting for at least {min_workers} worker(s) to register...")
    t_wait = time.time()
    while len(coord_state.workers) < min_workers:
        if time.time() - t_wait > wait_timeout:
            logger.warning(f"Timeout waiting for workers. Registered: {len(coord_state.workers)}")
            break
        time.sleep(2)
    logger.info(f"Registered workers: {list(coord_state.workers.keys())}")

    eval_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from .data.client_dataset import get_val_dataloader
    try:
        val_loader = get_val_dataloader(dataset_name, tokenizer=processor, processor=processor, batch_size=64)
        logger.info(f"Validation loader ready: {len(val_loader)} batches")
    except Exception as e:
        logger.error(f"Failed to create validation loader: {e}")
        val_loader = None

    for round_id in range(start_round, num_rounds):
        t_round_start = time.time()
        coord_state.current_round = round_id
        coord_state.pending_updates = []

        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_id}/{num_rounds}")
        logger.info(f"{'='*60}")

        available_clients = get_available_clients_from_workers(coord_state)
        if not available_clients:
            logger.info("Waiting for workers to become available...")
            wait_start = time.time()
            while not available_clients and time.time() - wait_start < runtime_cfg.get("round_timeout_sec", 300):
                time.sleep(3)
                available_clients = get_available_clients_from_workers(coord_state)
            if not available_clients:
                logger.warning("No available clients after timeout, skipping round")
                continue

        selected = sampler.sample(round_id, available_clients, client_resources)

        drop_weak_threshold = sched_cfg.get("drop_weak_budget_threshold", 0)
        if drop_weak_threshold > 0:
            before = len(selected)
            selected = [
                cid for cid in selected
                if client_resources.get(cid, ClientResourceState(cid)).budget > drop_weak_threshold
            ]
            if len(selected) < before:
                logger.info(f"DropWeak: filtered {before - len(selected)} weak clients "
                            f"(budget <= {drop_weak_threshold})")
            if not selected:
                logger.warning("DropWeak: all clients filtered, skipping round")
                continue

        logger.info(f"Selected clients: {selected}")

        t_plan_start = time.time()
        leases = planner.plan_round(round_id, selected, client_resources, feedback)
        planning_time = time.time() - t_plan_start
        logger.info(f"Planned {len(leases)} leases in {planning_time:.2f}s")

        with coord_state.lock:
            coord_state.round_tasks = {}
        lease_client_map = {}
        new_tasks: dict[str, list] = {}

        for lease in leases:
            cid = lease["client_id"]
            wid = get_worker_for_client(coord_state, cid)
            if wid is None:
                logger.warning(f"No worker for client {cid}")
                continue

            resource = client_resources.get(cid)
            client_slowdown = resource.tier.compute_slowdown if resource else 1.0

            task = {
                "round": int(round_id),
                "task_type": "train",
                "client_id": int(cid),
                "global_adapter_ref": "http",
                "lease": {
                    "lease_id": lease["lease_id"],
                    "slice_ids": lease["slice_ids"],
                    "param_names": lease["param_names"],
                    "local_steps": int(lease["local_steps"]),
                    "micro_batch_size": int(lease["micro_batch_size"]),
                    "deadline_seconds": float(lease["deadline_seconds"]),
                    "compute_slowdown": float(client_slowdown),
                    "simulated_step_time_sec": float(sched_cfg.get("simulated_step_time_sec", 0)),
                    "precision": model_cfg.get("precision", "bf16"),
                    "method": method,
                },
            }

            if wid not in new_tasks:
                new_tasks[wid] = []
            new_tasks[wid].append(task)
            lease_client_map[cid] = lease
            round_logger.log_lease(lease)

        with coord_state.lock:
            coord_state.round_tasks = new_tasks

        expected_clients = [l["client_id"] for l in leases]
        logger.info(f"Waiting for {len(expected_clients)} client updates...")

        def get_pending():
            return list(coord_state.pending_updates)

        received, completed_ids, missed_ids = round_mgr.wait_for_updates(
            expected_clients, get_pending,
            timeout=runtime_cfg.get("round_timeout_sec", 300.0),
        )

        logger.info(f"Received {len(received)} updates. "
                    f"Completed: {completed_ids}, Missed: {missed_ids}")

        guard.current_round = round_id
        for u in received:
            cid = u.get("client_id")
            payload = u.get("actual_payload_bytes", 0)
            upload_t = u.get("upload_time_sec", u.get("train_time_sec", 0))
            bw_predictor.record_upload(cid, UploadRecord(
                round_id=round_id, payload_bytes=payload,
                upload_time_sec=upload_t, success=True,
            ))

        for cid in missed_ids:
            bw_predictor.record_upload(cid, UploadRecord(
                round_id=round_id, payload_bytes=0,
                upload_time_sec=0, success=False,
            ))

        t_agg_start = time.time()
        updates_for_agg = []
        total_upload_bytes = 0
        deadline_miss_count = 0
        for u in received:
            cid = u.get("client_id")
            is_miss = cid in missed_ids

            if not is_miss:
                worker_reported_miss = u.get("deadline_miss", False)
                pre_upload_time = u.get("pre_upload_time_sec", 0)
                client_deadline = lease_client_map.get(cid, {}).get("deadline_seconds", float('inf'))
                if worker_reported_miss or pre_upload_time > client_deadline:
                    is_miss = True
                    deadline_miss_count += 1
                    logger.info(f"Client {cid} deadline miss: "
                                f"task_time={pre_upload_time:.1f}s > deadline={client_deadline:.1f}s "
                                f"(slowdown={u.get('compute_slowdown', 1.0):.1f}x)")

            cu = ClientUpdate(
                round_id=round_id,
                client_id=cid,
                worker_id=u.get("worker_id", ""),
                method=method,
                num_examples=u.get("num_examples", 0),
                local_steps=u.get("local_steps", 0),
                delta=u.get("delta", {}),
                trained_param_names=u.get("trained_param_names", []),
                train_time_sec=u.get("train_time_sec", 0),
                upload_time_sec=u.get("upload_time_sec", 0),
                payload_bytes=u.get("actual_payload_bytes", 0),
                peak_gpu_memory_gb=u.get("peak_gpu_memory_gb", 0),
                deadline_miss=is_miss,
                local_loss=u.get("local_loss", 0),
            )
            updates_for_agg.append(cu)
            total_upload_bytes += cu.payload_bytes

            round_logger.log_client_update({
                "round": round_id,
                "client_id": cid,
                "worker_id": cu.worker_id,
                "budget": lease_client_map.get(cid, {}).get("budget", 1.0),
                "lease_id": lease_client_map.get(cid, {}).get("lease_id", ""),
                "slice_ids": lease_client_map.get(cid, {}).get("slice_ids", []),
                "local_steps": cu.local_steps,
                "micro_batch_size": lease_client_map.get(cid, {}).get("micro_batch_size", 16),
                "num_examples": cu.num_examples,
                "payload_bytes": cu.payload_bytes,
                "train_time_sec": cu.train_time_sec,
                "upload_time_sec": cu.upload_time_sec,
                "peak_gpu_memory_gb": cu.peak_gpu_memory_gb,
                "deadline_seconds": lease_client_map.get(cid, {}).get("deadline_seconds", 0),
                "deadline_miss": is_miss,
                "compute_slowdown": u.get("compute_slowdown", 1.0),
                "pre_upload_time_sec": u.get("pre_upload_time_sec", 0),
                "guard_pass": lease_client_map.get(cid, {}).get("guard_pass", True),
                "guard_fail_reason": lease_client_map.get(cid, {}).get("guard_fail_reason"),
                "local_loss": cu.local_loss,
            })

        if updates_for_agg:
            server_lr = fl_cfg.get("local", {}).get("server_lr", 1.0)
            global_adapter = aggregate_partial_deltas(
                global_adapter, updates_for_agg, server_lr=server_lr,
            )
            coord_state.global_adapter_state = global_adapter
            coord_state.global_adapter_bytes = serialize_adapter_state(global_adapter)

        aggregation_time = time.time() - t_agg_start

        updated_slices: dict[str, list[int]] = defaultdict(list)
        for cu in updates_for_agg:
            if cu.deadline_miss:
                continue
            for pname in cu.trained_param_names:
                for sid, pnames in slice_registry.items():
                    if pname in pnames:
                        updated_slices[sid].append(cu.client_id)
                        break
        feedback.update_after_round(round_id, updated_slices)

        eval_results = {}
        if val_loader and eval_every > 0 and (round_id % eval_every == 0 or round_id == num_rounds - 1):
            try:
                load_lora_state_dict(model, global_adapter)
                eval_results = evaluate_model(model, val_loader, device=eval_device)
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                eval_results = {"eval_accuracy": None, "eval_macro_f1": None, "evaluation_time_sec": 0}

        round_time = time.time() - t_round_start
        client_times = [u.get("train_time_sec", 0) for u in received]
        upload_times = [u.get("upload_time_sec", 0) for u in received]
        serial_times = [u.get("serialization_time_sec", 0) for u in received]
        gpu_mems = [u.get("peak_gpu_memory_gb", 0) for u in received]

        p95_time = float(np.percentile(client_times, 95)) if client_times else 0.0
        p95_upload = float(np.percentile(upload_times, 95)) if upload_times else 0.0

        weak_participated = sum(
            1 for cid in completed_ids
            if client_resources.get(cid, ClientResourceState(cid)).budget <= 0.5
        )
        total_weak = sum(
            1 for cid in selected
            if client_resources.get(cid, ClientResourceState(cid)).budget <= 0.5
        )

        num_leases = len(leases)
        num_rejected = len(selected) - num_leases
        reject_rate = num_rejected / max(len(selected), 1)

        round_metrics = {
            "round": round_id,
            "method": method,
            "selected_clients": selected,
            "completed_clients": completed_ids,
            "missed_clients": missed_ids,
            "num_updates": len([u for u in updates_for_agg if not u.deadline_miss]),
            "eval_accuracy": eval_results.get("eval_accuracy"),
            "eval_macro_f1": eval_results.get("eval_macro_f1"),
            "eval_loss": eval_results.get("eval_loss"),
            "round_time_sec": round_time,
            "p95_client_time_sec": p95_time,
            "avg_client_time_sec": float(np.mean(client_times)) if client_times else 0.0,
            "avg_upload_time_sec": float(np.mean(upload_times)) if upload_times else 0.0,
            "p95_upload_time_sec": p95_upload,
            "avg_serialization_time_sec": float(np.mean(serial_times)) if serial_times else 0.0,
            "avg_peak_gpu_memory_gb": float(np.mean(gpu_mems)) if gpu_mems else 0.0,
            "max_peak_gpu_memory_gb": float(max(gpu_mems)) if gpu_mems else 0.0,
            "total_upload_bytes": total_upload_bytes,
            "avg_upload_bytes_per_client": total_upload_bytes // max(len(completed_ids), 1),
            "deadline_miss_rate": (len(missed_ids) + deadline_miss_count) / max(len(expected_clients), 1),
            "timeout_misses": len(missed_ids),
            "deadline_misses": deadline_miss_count,
            "weak_client_participation": weak_participated / max(total_weak, 1),
            "weak_update_share": weak_participated / max(len(completed_ids), 1),
            "aggregation_time_sec": aggregation_time,
            "planning_time_sec": planning_time,
            "evaluation_time_sec": eval_results.get("evaluation_time_sec", 0),
            "num_leases_planned": num_leases,
            "num_rejected_by_guard": num_rejected,
            "reject_rate": reject_rate,
        }
        round_logger.log_round(round_metrics)

        logger.info(
            f"Round {round_id} done: accuracy={eval_results.get('eval_accuracy', 'N/A')}, "
            f"upload={total_upload_bytes/1e6:.2f}MB, time={round_time:.1f}s"
        )

        if round_id % 5 == 0 or round_id == num_rounds - 1:
            ckpt_mgr.save(round_id, global_adapter, feedback.to_dict())

    coord_state.stopped = True
    logger.info("Training complete!")
    logger.info(f"Logs saved to: {log_dir}")

    logger.info("Keeping server alive for 15s for worker shutdown...")
    time.sleep(15)

def main():
    parser = argparse.ArgumentParser(description="FL Testbed Coordinator")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--nodes", default=None, help="Nodes config YAML")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=29580, help="Server port")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    run_coordinator(args.config, args.nodes, args.host, args.port, args.resume)

if __name__ == "__main__":
    main()
