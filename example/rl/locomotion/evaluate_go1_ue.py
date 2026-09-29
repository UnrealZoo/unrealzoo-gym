#!/usr/bin/env python3
"""Compare explicit Go1 policies in an existing local UE PIE session.

All physics is sampled through UEGo1Pool. Both checkpoint formats receive the
same noise-free Go1Task observation, calibrated from the actual reset joints.
This runner owns its evaluation actor, never the editor process.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from play_go1 import (
    PROFILE, validate_policy_contract, validate_policy_metadata,
    validate_reset_observation_reference,
)
from ue_go1_env import CONTROL_DT, UEGo1Pool, parse_connect
from ue_go1_task import DEFAULT_JOINT_POS, Go1Task, TASK_PROFILE
from ue_training_numerics import V2_PROFILE, applied_actions


CASES = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.5, 0.0, 0.0),
    "backward": (-0.3, 0.0, 0.0),
    "lateral": (0.0, 0.3, 0.0),
    "yaw": (0.0, 0.0, 0.5),
}
FALL_UPRIGHT = math.cos(math.radians(70.0))
FEET = ("FR", "FL", "RR", "RL")
METRICS_SCHEMA_VERSION = 2


def finite_array(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return value


class EvaluationPolicy:
    """A direct actor call, without legacy joint/reference or action adapters."""

    def __init__(self, metadata, session=None, actor=None, device="cpu"):
        if (session is None) == (actor is None):
            raise ValueError("Provide exactly one ONNX session or Torch actor")
        self.metadata = validate_policy_metadata(dict(metadata))
        self.session, self.actor, self.device = session, actor, device
        declared_default = finite_array(
            self.metadata["default_joint_pos"].split(","), (12,), "policy joint reference",
        )
        # Go1Task already subtracts this reference. Refuse a different policy
        # reference rather than silently inserting another adapter.
        if not np.allclose(declared_default, DEFAULT_JOINT_POS, atol=1e-6, rtol=0):
            raise ValueError("Policy reference differs from Go1Task observation coordinates")

    def reset(self):
        if self.actor is not None:
            self.actor.reset()

    def act(self, observation):
        value = finite_array(observation, (1, 48), "actor observation").astype(np.float32)
        if self.session is not None:
            result = self.session.run(
                [self.session.get_outputs()[0].name],
                {self.session.get_inputs()[0].name: value},
            )[0]
        else:
            import torch
            from tensordict import TensorDict

            with torch.inference_mode():
                result = self.actor(
                    TensorDict({"actor": torch.as_tensor(value, device=self.device)}, batch_size=[1]),
                    stochastic_output=False,
                ).detach().cpu().numpy()
        result = finite_array(result, (1, 12), "policy actions").astype(np.float32)
        if self.metadata.get("ue_task_profile") == V2_PROFILE:
            result, _ = applied_actions(result)
        return result


def load_policy(checkpoint: Path, device="cpu", threads=1):
    if checkpoint.suffix.lower() == ".onnx":
        if device != "cpu":
            raise ValueError("ONNX comparison uses CPUExecutionProvider; choose --device cpu")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(checkpoint), sess_options=options, providers=["CPUExecutionProvider"],
        )
        return EvaluationPolicy(validate_policy_contract(session), session=session)
    if checkpoint.suffix.lower() == ".pt":
        from ue_go1_checkpoint import load_ue_checkpoint

        loaded, metadata = load_ue_checkpoint(
            checkpoint, device, PROFILE["default_joint_pos"], threads=threads,
        )
        return EvaluationPolicy(metadata, actor=loaded.actor, device=device)
    raise ValueError("Checkpoint must be an explicit .onnx or UE-trained .pt")


def tracking_metrics(samples, key):
    if not samples:
        return None
    errors = np.array([
        [sample[key][0] - sample["command"][0],
         sample[key][1] - sample["command"][1],
         sample["gyro"][2] - sample["command"][2]]
        for sample in samples
    ])
    return {
        "axes": ["body_vx_m_s", "body_vy_m_s", "body_yaw_rate_rad_s"],
        "mae": np.abs(errors).mean(axis=0).tolist(),
        "rmse": np.sqrt(np.square(errors).mean(axis=0)).tolist(),
        "samples": len(samples),
    }


def optional_telemetry(state, key, width):
    """Missing or malformed diagnostics stay unknown; never manufacture zeros."""
    try:
        values = np.asarray(state.get(key), dtype=float)
    except (TypeError, ValueError):
        return np.full(width, np.nan)
    return values if values.shape == (width,) else np.full(width, np.nan)


def ground_contact_metrics(samples):
    """Diagnostics only; callers provide command samples after warmup."""
    count = len(samples)
    heights = []
    root_probe_count = 0
    foot_probe_count = np.zeros(4, dtype=int)
    contact_valid_count = np.zeros(4, dtype=int)
    contact_count = np.zeros(4, dtype=int)
    contact_speeds = [[] for _ in FEET]
    for sample in samples:
        state = sample.get("state") or {}
        root = optional_telemetry(state, "root_position_ue_cm", 3)
        support = optional_telemetry(
            {"height": [state.get("ground_support_z_ue_cm")]}, "height", 1,
        )[0]
        # Despite the field name, the C++ endpoint reports bLastGroundProbeHit.
        root_valid = state.get("local_ground_patch_active") is True and np.isfinite(support)
        root_probe_count += int(root_valid)
        if root_valid and np.isfinite(root).all():
            heights.append(float(root[2] - support))

        foot_hits = optional_telemetry(state, "foot_ground_probe_hits", 4)
        ground = optional_telemetry(state, "foot_ground_z_ue_cm", 4)
        foot_probe_count += (foot_hits == 1) & np.isfinite(ground)
        contacts = optional_telemetry(state, "foot_contacts", 4)
        valid_contact = np.isin(contacts, (0, 1))
        contact_valid_count += valid_contact
        contact_count += valid_contact & (contacts == 1)
        velocity = optional_telemetry(state, "foot_velocities", 12).reshape(4, 3)
        for foot in range(4):
            if contacts[foot] == 1 and np.isfinite(velocity[foot, :2]).all():
                contact_speeds[foot].append(float(np.linalg.norm(velocity[foot, :2])))

    def fraction(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    def slip_statistics(values):
        return {
            "contact_samples_with_velocity": len(values),
            "mean": float(np.mean(values)) if values else None,
            "rms": float(np.sqrt(np.mean(np.square(values)))) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "max": max(values) if values else None,
        }

    all_speeds = [value for foot_values in contact_speeds for value in foot_values]
    return {
        "scope": "Command steps after warmup; includes the terminal step when present",
        "samples": count,
        "feet": list(FEET),
        "body_height_above_support_cm": {
            "mean": float(np.mean(heights)) if heights else None,
            "min": min(heights) if heights else None,
            "p05": float(np.percentile(heights, 5)) if heights else None,
            "valid_samples": len(heights),
            "valid_fraction": fraction(len(heights), count),
            "definition": "Root origin above the last valid smoothed UE support height, not trunk underside clearance; missing/failed root probes excluded",
        },
        "root_ground_probe_valid_fraction": fraction(root_probe_count, count),
        "foot_ground_probe_valid_fraction": fraction(int(foot_probe_count.sum()), count * 4),
        "foot_ground_probe_valid_fraction_per_foot": [fraction(x, count) for x in foot_probe_count],
        "contact_duty_fraction": {
            "overall": fraction(int(contact_count.sum()), int(contact_valid_count.sum())),
            "per_foot": [fraction(x, y) for x, y in zip(contact_count, contact_valid_count)],
            "valid_sample_fraction": fraction(int(contact_valid_count.sum()), count * 4),
            "valid_sample_fraction_per_foot": [fraction(x, count) for x in contact_valid_count],
        },
        "contact_foot_slip_m_s": {
            **slip_statistics(all_speeds),
            "velocity_valid_fraction_of_contacts": fraction(len(all_speeds), int(contact_count.sum())),
            "per_foot": [slip_statistics(values) for values in contact_speeds],
            "definition": "World-XY foot speed conditioned on reported MuJoCo contact; no-contact samples excluded, no contacts yields null speed",
        },
    }


def summarize(samples, command_steps):
    measured = [sample for sample in samples if sample["phase"] == "command"]
    fallen = next((sample for sample in samples if sample["fallen"]), None)
    actions = np.asarray([sample["action"] for sample in samples], dtype=float)
    return {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "steps_completed": len(samples),
        "command_steps_completed": len(measured),
        "requested_command_steps": command_steps,
        "survived_full_command": fallen is None and len(measured) == command_steps,
        "fallen": fallen is not None,
        "first_fall_step": fallen["step"] if fallen else None,
        "survived_seconds_including_warmup": len(samples) * CONTROL_DT,
        "survived_command_seconds": len(measured) * CONTROL_DT,
        "root_tracking_after_warmup": tracking_metrics(measured, "root_linvel"),
        "imu_tracking_after_warmup": tracking_metrics(measured, "imu_linvel"),
        "ground_contact_after_warmup": ground_contact_metrics(measured),
        "minimum_upright": min((sample["upright"] for sample in samples), default=None),
        "mean_upright": float(np.mean([sample["upright"] for sample in samples])) if samples else None,
        "episode_reward_including_warmup": sum(sample["reward"] for sample in samples),
        "command_reward": sum(sample["reward"] for sample in measured),
        "control_target_clip_count": sum(sample["control_clip_count"] for sample in samples),
        "steps_with_control_target_clipping": sum(sample["control_clip_count"] > 0 for sample in samples),
        "raw_action_abs_max": float(np.max(np.abs(actions))) if samples else None,
        "raw_action_abs_ge_one_fraction": float(np.mean(np.abs(actions) >= 1)) if samples else None,
        "action_range_note": "Raw policy actions are unbounded; abs>=1 is diagnostic, not actuator saturation. UE control_target_clip_count reports actual target clipping.",
    }


def snapshot_runtime_asset(contract, directory):
    """Keep the actual collision/solver XML after the temporary actor is gone."""
    data = Path(contract["runtime_mjcf"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != contract["runtime_mjcf_sha256"]:
        raise RuntimeError("Runtime MJCF changed after the environment verified it")
    target = Path(directory) / f"environment-{digest}.xml"
    target.write_bytes(data)
    return {"path": str(target), "sha256": digest, "bytes": len(data)}


def run_case(pool, policy, name, command, states, command_steps, warmup_steps, case=None,
             environment_profile=TASK_PROFILE, command_schedule=None):
    # A per-step schedule changes commands without resetting the task, policy,
    # or robot at the boundaries. Fixed-command callers keep identical behavior.
    schedule = None if command_schedule is None else finite_array(
        command_schedule, (command_steps, 3), "command schedule",
    )
    # Keep the caller's case object current so interrupted rollouts retain all
    # accepted samples in the final failure report.
    case = {} if case is None else case
    case.update({
        "name": name, "command": list(command), "initial_state": copy.deepcopy(states[0]),
        "samples": [],
    })
    if schedule is not None:
        case["command_schedule"] = schedule.tolist()
    reference_check = validate_reset_observation_reference(states[0], policy.metadata)
    if reference_check["mode"] == "legacy_bridge_default":
        reference_check = {"mode": "policy_has_no_declared_UE_reset_reference"}
    case["checkpoint_reset_reference_validation"] = reference_check
    contract = copy.deepcopy(pool.workers[0].robots[0].asset_contract)
    imu_offset = finite_array(contract["imu_offset_m"], (3,), "runtime IMU offset")
    task = Go1Task(
        1, observation_noise=False,
        episode_seconds=(command_steps + warmup_steps) * CONTROL_DT,
        imu_offset=imu_offset,
        profile=environment_profile,
    )
    task.reset([0], states)
    policy.reset()
    case.update({"runtime_asset_contract": contract, "task_manifest": task.task_manifest})
    started = time.perf_counter()
    try:
        for index in range(warmup_steps + command_steps):
            phase = "warmup" if index < warmup_steps else "command"
            current_command = (command if schedule is None or phase == "warmup"
                               else schedule[index - warmup_steps])
            requested = np.asarray(
                [[0, 0, 0] if phase == "warmup" else current_command], dtype=np.float32,
            )
            # Infinity in set_commands disables resampling inside task.step.
            task.set_commands(requested)
            observation = task.observe(states)["actor"]
            actions = policy.act(observation)
            if environment_profile == V2_PROFILE:
                # This also covers the original ONNX reference under explicitly
                # selected v2 physics: every policy uses the same action boundary.
                actions, _ = applied_actions(actions)
            states = pool.step(actions, requested)
            _, rewards, terminated, truncated, task_metrics = task.step(states, actions)
            state = states[0]
            raw = finite_array(state["obs"], (48,), "UE observation")
            root_linvel = raw[:3] - np.cross(raw[3:6], imu_offset)
            case["samples"].append({
                "step": index + 1, "phase": phase, "sim_time": float(state["sim_time"]),
                "command": requested[0].tolist(), "action": actions[0].tolist(),
                "actor_observation": observation[0].tolist(),
                "root_linvel": root_linvel.tolist(), "imu_linvel": raw[:3].tolist(),
                "gyro": raw[3:6].tolist(), "upright": float(-raw[8]),
                "fallen": bool(terminated[0]), "timed_out": bool(truncated[0]),
                "reward": float(rewards[0]),
                "reward_terms": {key: float(value[0]) for key, value in task_metrics["reward_terms"].items()},
                "control_clip_count": int(state.get("control_clip_count", 0)),
                "state": copy.deepcopy(state),
            })
            if terminated[0] or truncated[0]:
                break
    finally:
        case["wall_seconds"] = time.perf_counter() - started
        case["metrics"] = summarize(case["samples"], command_steps)
    return case


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", required=True, help="Existing local UE PIE HOST:PORT")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON output path")
    parser.add_argument("--episode-seconds", type=float, default=20.0, help="Measured command duration per case, excluding warmup")
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    parser.add_argument("--spawn", type=float, nargs=3, help="Explicit UE cm; otherwise record camera-relative initial spawn")
    parser.add_argument("--spawn-rotation", type=float, nargs=3, help="UE pitch yaw roll in degrees")
    parser.add_argument("--reset-mode", choices=("auto", "rebuild"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--environment-profile", choices=("auto", TASK_PROFILE, V2_PROFILE), default="auto",
                        help="auto follows the checkpoint; explicit profile also permits pretrained source comparison")
    args = parser.parse_args(argv)
    try:
        parse_connect(args.connect)
        for key in ("episode_seconds", "request_timeout"):
            if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        steps = args.episode_seconds / CONTROL_DT
        if not math.isclose(steps, round(steps), abs_tol=1e-7, rel_tol=0) or round(steps) < 1:
            raise ValueError("episode_seconds must contain an integer number of 20 ms steps")
        if args.warmup_steps < 0 or args.threads < 1 or len(set(args.cases)) != len(args.cases):
            raise ValueError("Use nonnegative warmup, positive threads and unique cases")
        for key in ("spawn", "spawn_rotation"):
            if getattr(args, key) is not None:
                finite_array(getattr(args, key), (3,), key)
        args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        if not args.checkpoint.is_file() or args.checkpoint.suffix.lower() not in (".pt", ".onnx"):
            raise ValueError("checkpoint must be an existing .onnx or .pt file")
        args.output = args.output.expanduser().resolve()
        if args.output.suffix.lower() != ".json":
            raise ValueError("output must use a .json suffix")
        if args.output.exists() or args.output.with_suffix(".ue").exists():
            raise ValueError("output JSON and its .ue workspace must not already exist")
        args.command_steps = int(round(steps))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    policy = load_policy(args.checkpoint, args.device, args.threads)
    environment_profile = (policy.metadata.get("ue_task_profile", TASK_PROFILE)
                           if args.environment_profile == "auto" else args.environment_profile)
    if policy.metadata.get("ue_task_profile") == V2_PROFILE and environment_profile != V2_PROFILE:
        raise ValueError("v2 checkpoint evaluation requires v2 environment physics")
    result = {
        "schema_version": 1, "complete": False, "evaluation_profile": "ue_go1_fixed_commands_v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "policy_metadata": policy.metadata, "connect": args.connect, "device": args.device,
        "inference_threads": args.threads, "task_profile": environment_profile,
        "control_dt": CONTROL_DT, "warmup_steps": args.warmup_steps,
        "command_seconds": args.episode_seconds, "fall_upright": FALL_UPRIGHT,
        "scope": "Initial training command range; high-speed and combined commands require separate evaluation",
        "observation_contract": "Go1Task noise-free calibrated joint coordinates; direct ONNX or Torch actor, no legacy bridge correction",
        "physics": "Only explicit synchronous stepping inside the existing UE PIE process",
        "cases": [],
    }
    pool = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        pool = UEGo1Pool(
            ue_binary=None, connect=args.connect, num_processes=1, agents_per_process=1,
            base_port=parse_connect(args.connect)[1], output_dir=args.output.with_suffix(".ue"),
            request_timeout=args.request_timeout, reset_mode=args.reset_mode,
            training_telemetry="keyboard_v2" if environment_profile == V2_PROFILE else "disabled",
        )
        states = pool.reset()
        if args.spawn is not None or args.spawn_rotation is not None:
            states = pool.reset_indices(
                [0], spawn_locations=None if args.spawn is None else [args.spawn],
                spawn_rotations=None if args.spawn_rotation is None else [args.spawn_rotation],
            )
        robot = pool.workers[0].robots[0]
        result["spawn_location"] = robot.location.tolist()
        result["spawn_rotation"] = robot.rotation.tolist()
        result["world_context"] = copy.deepcopy(pool.workers[0].world_context)
        for index, name in enumerate(args.cases):
            if index:
                states = pool.reset_indices([0])
            case = {"name": name}
            result["cases"].append(case)
            case["runtime_asset_snapshot"] = snapshot_runtime_asset(
                pool.workers[0].robots[0].asset_contract, args.output.with_suffix(".ue"),
            )
            run_case(
                pool, policy, name, CASES[name], states,
                args.command_steps, args.warmup_steps, case=case,
                environment_profile=environment_profile,
            )
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(f"EVAL|{name}|{json.dumps(case['metrics'])}", flush=True)
        result["environment"] = pool.metrics()
        result["complete"] = True
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if pool is not None:
            if "environment" not in result:
                try:
                    result["environment"] = pool.metrics()
                except Exception as error:
                    result["environment_metrics_error"] = f"{type(error).__name__}: {error}"
            pool.close()
            result["cleanup_errors"] = pool.cleanup_errors
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"RESULT|{args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
