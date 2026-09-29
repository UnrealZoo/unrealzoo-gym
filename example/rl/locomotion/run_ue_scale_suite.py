#!/usr/bin/env python3
"""Run sequential UE capacity and real PPO checks without overlapping UE jobs.

This is a bounded measurement suite, not a locomotion convergence experiment.
Each child owns its UE instances and records a separate manifest/result file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ue-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    entry = Path(__file__).resolve().parent
    jobs = []

    def run(name, script, arguments):
        command = [sys.executable, str(entry / script), "--ue-binary", str(args.ue_binary), *map(str, arguments)]
        print(f"START|{name}", flush=True)
        with (output / f"{name}.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        jobs.append({"name": name, "command": command, "returncode": result.returncode})
        (output / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        print(f"DONE|{name}|returncode={result.returncode}", flush=True)
        return result.returncode

    # Force short timeouts to exercise actual per-replica reset and PPO together.
    if run("ppo-reset-smoke", "train_go1_ue.py", [
        "--num-processes", 1, "--agents-per-process", 2, "--iterations", 2,
        "--episode-seconds", 0.2, "--base-port", 19300,
        "--log-dir", output / "ppo-reset-smoke",
    ]):
        raise RuntimeError("Real UE/PPO smoke failed; inspect its log before scaling")

    measurements = []
    # Fixed replicas per process measures scaling with constant worker load;
    # total replica count grows too, so this is not a fixed-total-N comparison.
    # At most four UE processes are used to limit host CPU pressure. The largest
    # process layout increases replica count until the explicit RAM/disk
    # guard or the configured bound; partial allocation is not called a pass.
    for job_index, processes in enumerate((1, 2, 4), start=1):
        counts = [8, 32, 64, 128, 256] if processes == 4 else [8]
        name = f"capacity-p{processes}"
        code = run(name, "benchmark_go1_ue.py", [
            "--processes", processes, "--counts", *counts, "--steps", 64,
            "--warmup", 8, "--base-port", 19300 + job_index * 100,
            "--reset-replicas-per-process", 1,
            "--output", output / name,
        ])
        result = json.loads((output / name / "results.json").read_text())
        for case in result["cases"]:
            if case["status"] == "completed":
                measurements.append({"processes": processes, **case})
        if code and "Resource guard:" not in result.get("error", ""):
            raise RuntimeError(f"{name} failed unexpectedly; inspect its result")

    if not measurements:
        raise RuntimeError("No completed capacity cases")
    largest = max(measurements, key=lambda case: case["num_envs"])
    efficient = max(
        (case for case in measurements if case["agents_per_process"] == 8),
        key=lambda case: case["aggregate_steps_per_second"],
    )
    (output / "selection.json").write_text(json.dumps({
        "largest_completed_capacity": largest,
        "highest_measured_pure_step_throughput_at_eight_replicas_per_process": efficient,
        "selection_does_not_prove_optimal_training_throughput": True,
    }, indent=2) + "\n")
    # Real updates at the largest fully tested allocation demonstrate training
    # memory capacity. More updates on the faster layout measure reset overhead.
    training_attempts = []
    candidates = sorted(measurements, key=lambda case: case["num_envs"], reverse=True)
    for index, case in enumerate(candidates):
        name = "ppo-largest" if index == 0 else f"ppo-capacity-fallback-{index}"
        code = run(name, "train_go1_ue.py", [
            "--num-processes", case["processes"],
            "--agents-per-process", case["agents_per_process"],
            "--iterations", 2, "--base-port", 19900 + 100 * index,
            "--log-dir", output / name,
        ])
        training_attempts.append({"name": name, "num_envs": case["num_envs"], "returncode": code})
        (output / "ppo_capacity_attempts.json").write_text(json.dumps(training_attempts, indent=2) + "\n")
        if code == 0:
            break
    # A failed large allocation must not hide results for the faster small one.
    # 48 * 24 * 0.02 = 23.04 simulated seconds: surviving replicas cross their
    # 20-second timeout, so this also measures actual automatic reset overhead.
    throughput_code = run("ppo-throughput", "train_go1_ue.py", [
        "--num-processes", efficient["processes"],
        "--agents-per-process", efficient["agents_per_process"],
        "--iterations", 48, "--base-port", 22000,
        "--log-dir", output / "ppo-throughput",
    ])
    if throughput_code or not any(item["returncode"] == 0 for item in training_attempts):
        raise RuntimeError("PPO capacity/throughput checks did not both complete; inspect manifests")


if __name__ == "__main__":
    main()
