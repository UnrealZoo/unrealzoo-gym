#!/usr/bin/env python3
"""Measure UE replica scaling and resets in packaged or editor sessions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ue_go1_env import UEGo1Pool, host_resources, state_from, parse_connect
from ue_process_metrics import ExternalUEProcessMonitor, host_cpu_capacity


def save(path: Path, result: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def host_load_snapshot() -> dict:
    """Record shared-host context outside measured sampling intervals."""
    if sys.platform != "linux":
        return {"loadavg": None, "processes": None, "gpu": None,
                "unavailable_reason": "Linux host load/GPU probe is not supported on this platform"}
    result = {"loadavg": Path("/proc/loadavg").read_text().strip()}
    for key, command in (
        ("processes", ["ps", "-eo", "pid,comm,pcpu,rss", "--sort=-pcpu"]),
        ("gpu", ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"]),
    ):
        try:
            text = subprocess.check_output(command, text=True, timeout=5)
            result[key] = "\n".join(text.splitlines()[:15])
        except (OSError, subprocess.SubprocessError) as error:
            result[key] = str(error)
    return result


def isolation_probe(pool: UEGo1Pool) -> dict:
    worker = pool.workers[0]
    if len(worker.robots) < 2:
        return {"status": "requires_two_replicas"}
    first, other = worker.robots[:2]
    before = state_from(worker.request(f"vget /object/{other.name}/mujoco_go1_policy_obs"))
    reply = worker.request(
        f"vset /object/{first.name}/mujoco_go1_policy_step " + " ".join(["0.1"] * 12)
    )
    first.state = state_from(reply, float(first.state["sim_time"]))
    after = state_from(worker.request(f"vget /object/{other.name}/mujoco_go1_policy_obs"))
    unchanged = np.array_equal(before["obs"], after["obs"]) and before["sim_time"] == after["sim_time"]
    if not unchanged:
        raise RuntimeError("Stepping one replica changed another replica's physics")
    worker.reset_one(0)
    after_reset = state_from(worker.request(f"vget /object/{other.name}/mujoco_go1_policy_obs"))
    reset_isolated = np.array_equal(before["obs"], after_reset["obs"]) and before["sim_time"] == after_reset["sim_time"]
    if not reset_isolated:
        raise RuntimeError("Resetting one replica changed another replica's physics")
    locations = np.asarray([robot.location for robot in worker.robots])
    spread = np.ptp(locations, axis=0)
    if np.max(spread) > 0.01:
        raise RuntimeError(f"Co-located robots settled at different locations: {spread}")
    return {
        "status": "passed", "single_step_other_state_unchanged": unchanged,
        "single_reset_other_state_unchanged": reset_isolated,
        "spawn_spread_cm": spread.tolist(),
        "scope": "static scene, no shared dynamic props; first two replicas",
    }


def measure_resets(
    pool: UEGo1Pool, replicas_per_process: int, measured_steps: int,
    sampling_seconds: float,
) -> dict:
    """Time full resets or a declared sample without extrapolating training speed."""
    if replicas_per_process < 0:
        raise ValueError("Reset replicas per process must be non-negative")
    per_worker = pool.agents_per_process
    selected_per_worker = (
        per_worker if replicas_per_process == 0 else min(replicas_per_process, per_worker)
    )
    indices = [
        worker_index * per_worker + local_index
        for worker_index in range(len(pool.workers))
        for local_index in range(selected_per_worker)
    ]
    started = time.perf_counter()
    pool.reset_indices(indices)
    elapsed = time.perf_counter() - started
    # Unselected replicas deliberately retain advanced physics; comparing them
    # against freshly reset states would report a false reset-equivalence error.
    states = pool.states
    reset_obs = np.asarray([states[index]["obs"] for index in indices])
    difference = float(np.max(np.abs(reset_obs - reset_obs[0])))
    if difference > 1e-6:
        raise RuntimeError("Selected replicas did not return to equivalent reset observations")
    full_reset = len(indices) == pool.num_envs
    return {
        "reset_env_count": len(indices),
        "reset_env_indices": indices,
        "reset_replicas_per_process": selected_per_worker,
        "reset_seconds": elapsed,
        "full_reset_seconds": elapsed if full_reset else None,
        "reset_scope": "all_replicas" if full_reset else "first_replicas_per_process",
        "reset_observation_max_difference": difference,
        "reset_observation_comparison_env_count": len(indices),
        "reset_observation_equivalence_checked": len(indices) > 1,
        "sample_plus_reset_steps_per_second": (
            pool.num_envs * measured_steps / (sampling_seconds + elapsed)
            if full_reset else None
        ),
        "reset_throughput_note": (
            "Full-reset synthetic window rate; not measured PPO training throughput"
            if full_reset else
            "Selected reset latency only; no full-reset or training-throughput extrapolation"
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    connection = parser.add_mutually_exclusive_group(required=True)
    connection.add_argument("--ue-binary", type=Path)
    connection.add_argument("--connect", help="Existing local UE PIE at localhost:PORT or 127.0.0.1:PORT")
    parser.add_argument("--ue-pid", type=int, help="Read-only CPU/RSS monitoring for --connect; must match game/context process_id")
    parser.add_argument("--reset-mode", choices=("auto", "rebuild"), default="auto")
    parser.add_argument("--runtime-diagnostics", choices=("enabled", "disabled"), default="disabled",
                        help="Request per-step UE diagnostic writes; old servers report control unavailable")
    parser.add_argument("--step-mode", choices=("auto", "single", "batch_serial", "batch_parallel", "fast"), default="auto",
                        help="auto keeps full-state compatibility; fast requires binary training API and skips per-step Actor pose sync")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--base-port", type=int, default=19100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument(
        "--reset-replicas-per-process", type=int, default=0,
        help="0 resets every replica; K>0 resets only the first min(K, count) per process",
    )
    args = parser.parse_args(argv)
    if args.processes < 1 or min(args.counts) < 1 or args.steps < 1 or args.warmup < 0:
        parser.error("Counts/steps must be positive, warmup non-negative")
    if args.counts != sorted(set(args.counts)):
        parser.error("--counts must be strictly increasing")
    if args.reset_replicas_per_process < 0:
        parser.error("--reset-replicas-per-process must be non-negative")
    if not 1024 <= args.base_port <= 65535 - args.processes + 1:
        parser.error("UE port range must lie between 1024 and 65535")
    if args.connect is not None:
        try:
            parse_connect(args.connect)
        except ValueError as error:
            parser.error(str(error))
        if args.processes != 1:
            parser.error("--connect requires --processes 1")
    if args.ue_pid is not None and (args.connect is None or args.ue_pid <= 0):
        parser.error("--ue-pid requires --connect and a positive process ID")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "backend": "UE external editor" if args.connect else "UE NullRHI", "argv": sys.argv,
        "connection_mode": "attach" if args.connect else "launch", "reset_mode": args.reset_mode,
        "runtime_diagnostics_requested": args.runtime_diagnostics,
        "step_mode_requested": args.step_mode,
        "render_mode": "existing_editor_settings" if args.connect else "NullRHI",
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("benchmark_go1_ue.py", "ue_go1_env.py", "ue_process_metrics.py")
        },
        "policy": "constant zero raw actions; physical capacity only, not PPO training",
        "world_paused": True, "counts_per_process": args.counts,
        "processes": args.processes, "steps_per_case": args.steps, "cases": [],
        "requested_reset_replicas_per_process": args.reset_replicas_per_process,
        "unreset_replicas_continue_across_cases": (
            0 < args.reset_replicas_per_process < max(args.counts)
        ),
        "resource_guards": {"available_ram_min_gib": 8, "free_disk_min_gib": 12},
        "requested_external_ue_pid": args.ue_pid,
        "initial_host": {**host_resources(), **host_cpu_capacity()},
        "initial_load": host_load_snapshot(),
    }
    path = output / "results.json"
    save(path, report)
    pool = None
    try:
        pool = UEGo1Pool(
            args.ue_binary, args.processes, args.counts[0], args.base_port,
            output, args.request_timeout, reset_mode=args.reset_mode, connect=args.connect,
            runtime_diagnostics=args.runtime_diagnostics == "enabled",
            step_mode=args.step_mode,
        )
        process_monitor = ExternalUEProcessMonitor(pool.workers[0], args.ue_pid) if args.ue_pid is not None else None
        for count in args.counts:
            case = {"agents_per_process": count, "num_envs": count * args.processes, "status": "initializing"}
            report["cases"].append(case)
            save(path, report)
            start = time.perf_counter()
            states = pool.reset() if not pool.initialized else pool.grow(count)
            case["incremental_init_seconds"] = time.perf_counter() - start
            case["initial_resources"] = pool.metrics()
            case["isolation"] = isolation_probe(pool)
            case["runtime_environment_geom_counts"] = sorted({
                int(robot.asset_contract["environment_geom_count"])
                for worker in pool.workers for robot in worker.robots
            })
            if count <= 2:
                (output / f"initial-states-{count}.json").write_text(json.dumps(pool.states, indent=2) + "\n")
            actions = np.zeros((pool.num_envs, 12), dtype=np.float32)
            commands = np.zeros((pool.num_envs, 3), dtype=np.float32)
            case.update(status="warming_up", warmup_seconds=0.0,
                        warmup_vector_step_seconds=[])
            save(path, report)
            warmup_started = time.perf_counter()
            try:
                for _ in range(args.warmup):
                    warmup_step_started = time.perf_counter()
                    pool.step(actions, commands)
                    case["warmup_vector_step_seconds"].append(time.perf_counter() - warmup_step_started)
                    case["warmup_seconds"] = time.perf_counter() - warmup_started
                    save(path, report)
            finally:
                case["warmup_seconds"] = time.perf_counter() - warmup_started
            case["sampling_initial_sim_time_range"] = [
                min(float(state["sim_time"]) for state in pool.states),
                max(float(state["sim_time"]) for state in pool.states),
            ]
            case["status"] = "sampling"
            case["load_before_sampling"] = host_load_snapshot()
            save(path, report)
            resources_before = pool.metrics()["processes"]
            cpu_before = sum(p["cpu_seconds"] for p in resources_before) if all(
                p.get("cpu_seconds") is not None for p in resources_before
            ) else None
            samples = []
            upright_min = 1.0
            if process_monitor is not None:
                process_monitor.begin()
            start = time.perf_counter()
            for step in range(args.steps):
                step_start = time.perf_counter()
                states = pool.step(actions, commands)
                samples.append(time.perf_counter() - step_start)
                upright_min = min(upright_min, min(-float(state["obs"][8]) for state in states))
                if step % 16 == 0:
                    pool.check_resources()
                if process_monitor is not None:
                    process_monitor.sample()
            finished = time.perf_counter()
            elapsed = finished - start
            if process_monitor is not None:
                case["external_ue_process_metrics"] = process_monitor.finish(start, finished)
            resources_after = pool.metrics()["processes"]
            cpu_after = sum(p["cpu_seconds"] for p in resources_after) if all(
                p.get("cpu_seconds") is not None for p in resources_after
            ) else None
            case.update(
                step_wall_seconds=elapsed,
                aggregate_steps_per_second=pool.num_envs * args.steps / elapsed,
                per_replica_steps_per_second=args.steps / elapsed,
                vector_step_ms_p50=float(np.median(samples) * 1000),
                vector_step_ms_p95=float(np.percentile(samples, 95) * 1000),
                minimum_upright=upright_min,
                mean_cpu_cores_during_sampling=(cpu_after - cpu_before) / elapsed
                if cpu_before is not None and cpu_after is not None else None,
            )
            if process_monitor is not None:
                case["mean_cpu_cores_during_sampling"] = case["external_ue_process_metrics"]["mean_cpu_cores"]
            for counter in ("wchar", "write_bytes"):
                if all(process.get(counter) is not None for process in resources_before + resources_after):
                    delta = sum(p[counter] for p in resources_after) - sum(p[counter] for p in resources_before)
                    case[f"ue_{counter}_per_second"] = delta / elapsed
                else:
                    case[f"ue_{counter}_per_second"] = None
            # Reset latency is measured outside the pure sampling interval.
            # Sampled reset latency is never represented as full training cost.
            case["status"] = "resetting"
            save(path, report)
            case.update(measure_resets(
                pool, args.reset_replicas_per_process, args.steps, elapsed
            ))
            case["final_resources"] = pool.metrics()
            case["status"] = "completed"
            save(path, report)
            print("CASE|" + json.dumps({key: case[key] for key in (
                "num_envs", "aggregate_steps_per_second", "vector_step_ms_p95",
                "reset_env_count", "reset_seconds", "full_reset_seconds", "minimum_upright",
            )}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="stopped", error=f"{type(error).__name__}: {error}")
        print(report["error"], flush=True)
        if isinstance(error, KeyboardInterrupt):
            raise
        return 1
    finally:
        if pool is not None:
            try:
                report["final_pool"] = pool.metrics()
            except Exception as error:
                report["final_metrics_error"] = f"{type(error).__name__}: {error}"
            finally:
                pool.close()
            report["owned_process_exit_codes"] = [worker.proc.returncode for worker in pool.workers if worker.proc is not None]
            report["cleanup_errors"] = pool.cleanup_errors
        report["final_host"] = {**host_resources(), **host_cpu_capacity()}
        save(path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
