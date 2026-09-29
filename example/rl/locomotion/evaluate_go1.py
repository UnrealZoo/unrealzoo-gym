#!/usr/bin/env python3
"""Evaluate a locally trained Go1 checkpoint or a random network without rendering."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import time

from train_go1 import (
    TASK_ID, configure_local_runtime, inspect_source, load_task,
    positive_int, sha256, write_json,
)

# Fixed test cases, never fed back to the learner as an evaluation curriculum.
# "heldout" means held out for evaluation, not excluded from the continuous
# training command distribution. The fast-forward case exceeds its initial range.
COMMANDS = (
    ("stand", (0.0, 0.0, 0.0)),
    ("forward", (0.5, 0.0, 0.0)),
    ("backward", (-0.5, 0.0, 0.0)),
    ("turn", (0.0, 0.0, 0.5)),
    ("heldout_forward_turn", (0.8, 0.25, 0.35)),
    ("heldout_fast_forward", (1.25, 0.0, 0.0)),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=Path,
                        help="Explicit trusted local .pt checkpoint; never defaults to pretrained")
    policy.add_argument("--random-policy", action="store_true",
                        help="Evaluate the deterministic mean of a newly initialized network")
    parser.add_argument("--num-envs", type=positive_int, default=64)
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="Maximum duration of the first episode for each command")
    parser.add_argument("--settle-seconds", type=float, default=1.0,
                        help="Exclude the initial transient from velocity errors, not from falls")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def configure_eval_task(env_cfg, seconds: float) -> None:
    """Freeze the original task's command distribution without replacing rewards."""
    env_cfg.auto_reset = False  # Metrics must see terminal state, not reset state.
    env_cfg.episode_length_s = seconds
    env_cfg.curriculum = {}
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.events.pop("push_robot", None)
    command = env_cfg.commands["twist"]
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.rel_world_envs = 0.0
    command.rel_forward_envs = 0.0
    command.init_velocity_prob = 0.0
    command.resampling_time_range = (seconds + 1.0, seconds + 1.0)


def evaluate_command(env, wrapped, policy, name, command, args, case_index):
    import torch

    term = env.command_manager.get_term("twist")
    for field, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), command):
        setattr(term.cfg.ranges, field, (value, value))
    env.reset(seed=args.seed + case_index)
    obs = wrapped.get_observations()
    target = torch.tensor(command, device=env.device).expand(args.num_envs, -1)
    torch.testing.assert_close(term.command, target, rtol=0, atol=1e-6)
    robot = env.scene["robot"]
    active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
    first_fall = torch.zeros_like(active)
    first_timeout = torch.zeros_like(active)
    first_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    first_returns = torch.zeros(args.num_envs, device=env.device)
    error_sum = torch.zeros(3, device=env.device)
    error_squared_sum = torch.zeros(3, device=env.device)
    measured_samples = torch.zeros((), dtype=torch.long, device=env.device)
    steps = math.ceil(args.seconds / env.step_dt)
    settle_steps = math.ceil(args.settle_seconds / env.step_dt)
    start = time.perf_counter()
    for step in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, _ = wrapped.step(actions)
        if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
            raise RuntimeError(f"Non-finite actions or rewards in {name}, step {step}")
        first_lengths += active.to(torch.long)
        first_returns += rewards * active
        velocity = torch.cat((robot.data.root_link_lin_vel_b[:, :2],
                              robot.data.root_link_ang_vel_b[:, 2:3]), dim=-1)
        if not torch.isfinite(velocity).all():
            raise RuntimeError(f"Non-finite physics state in {name}, step {step}")
        if step >= settle_steps:
            errors = velocity[active] - target[active]
            error_sum += errors.abs().sum(dim=0)
            error_squared_sum += errors.square().sum(dim=0)
            measured_samples += active.sum()
        # These are still the terminal signals because auto_reset is disabled.
        first_fall |= active & env.reset_terminated
        first_timeout |= active & env.reset_time_outs
        active &= ~dones.bool()
        if not bool(active.any()):
            break
        if bool(dones.any()):
            env.reset(env_ids=dones.nonzero(as_tuple=False).flatten())
            obs = wrapped.get_observations()
        torch.testing.assert_close(term.command, target, rtol=0, atol=1e-6)
    torch.cuda.synchronize()
    count = int(measured_samples.item())
    mae = (error_sum / count).cpu().tolist() if count else None
    rmse = torch.sqrt(error_squared_sum / count).cpu().tolist() if count else None
    lengths = first_lengths.float() * env.step_dt
    return {
        "name": name, "command_vx_vy_yaw": list(command),
        "num_first_episodes": args.num_envs, "horizon_seconds": steps * env.step_dt,
        "falls": int(first_fall.sum().item()),
        "timeouts": int(first_timeout.sum().item()),
        "survival_fraction": float((~first_fall).float().mean().item()),
        "first_episode_length_seconds_mean": float(lengths.mean().item()),
        "first_episode_length_seconds_min": float(lengths.min().item()),
        "first_episode_return_mean": float(first_returns.mean().item()),
        "velocity_mae_vx_vy_yaw": mae, "velocity_rmse_vx_vy_yaw": rmse,
        "error_samples": count, "settle_seconds": args.settle_seconds,
        "elapsed_seconds": time.perf_counter() - start,
    }


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if not math.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds < args.seconds:
        parser.error("--settle-seconds must be in [0, seconds)")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite evaluation: {output}")
    configure_local_runtime()
    source = inspect_source()
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("Evaluation requires an available NVIDIA CUDA device")
    torch.set_num_threads(args.threads)
    configure_torch_backends()
    env_cfg, agent_cfg, runner_cls = load_task(args.num_envs, args.seed)
    configure_eval_task(env_cfg, args.seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(output.with_suffix(".env.yaml"), asdict(env_cfg))
    results = {
        "schema_version": 1, "status": "initializing", "backend": "mjlab",
        "ue_actor_backend": False, "task_id": TASK_ID, "source": source,
        "policy": "checkpoint" if args.checkpoint else "random_network_mean",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "seed": args.seed, "num_envs": args.num_envs,
        "evaluation_protocol": {
            "episodes": "first episode per environment and command; stop metrics at first done",
            "errors": "pre-reset terminal state included; initial settling interval excluded",
            "heldout": "fixed evaluation cases, not excluded from continuous training ranges",
            "pushes": False, "actor_observation_noise": False,
            "startup_domain_randomization": "retained from original task",
            "reset_randomization": "retained from original task",
        },
        "commands": [],
    }
    write_json(output, results)
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = runner_cls(wrapped, asdict(agent_cfg), None, args.device)
        if args.checkpoint is not None:
            runner.load(str(args.checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        results.update(status="evaluating", control_dt=env.step_dt)
        for case_index, (name, command) in enumerate(COMMANDS):
            result = evaluate_command(env, wrapped, policy, name, command, args, case_index)
            results["commands"].append(result)
            write_json(output, results)
            print(f"{name}: survival={result['survival_fraction']:.3f} "
                  f"velocity_mae={result['velocity_mae_vx_vy_yaw']}", flush=True)
        results["status"] = "completed"
        write_json(output, results)
        print(f"Evaluation written to {output}")
    except BaseException as error:
        results.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output, results)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
