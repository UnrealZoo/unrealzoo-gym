#!/usr/bin/env python3
"""Evaluate an explicit UE-trained Go1 .pt or legacy Go1 ONNX in UnrealZoo.

Uses the existing keyboard, observation adapter, and synchronous UE step. No
pretrained policy is selected implicitly. With --ue-binary, this script owns a
separate UE process (NullRHI by default); otherwise it attaches to --host/--port.
Linux v3.1.0 needs --compat-v310 because it lacks mujoco_physics_config.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
JOINT_NAMES = tuple(
    "{}_{}_joint".format(leg, joint)
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
OBSERVATION_NAMES = (
    "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos",
    "joint_vel", "actions", "command",
)
PROFILE = {
    "default_joint_pos": [0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2,
    "action_scale": [0.3727530387, 0.3727530387, 0.2485020258] * 4,
    "joint_stiffness": [15.8952426532, 15.8952426532, 35.7642959698] * 4,
    "joint_damping": [1.0119225760, 1.0119225760, 2.2768257960] * 4,
}
CONTROL_PERIOD = 0.02
UE_TASK_PROFILE = "ue_v310_velocity"
UE_TASK_PROFILES = (UE_TASK_PROFILE, "ue_keyboard_flat_v2")


def finite_vector(value, size, name):
    if len(value) != size:
        raise ValueError("{} must have {} values".format(name, size))
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("{} contains non-finite values".format(name))
    return result


def ue_observation_reference(metadata):
    """Validate the explicit UE-trained observation contract, if present."""
    calibration_keys = (
        "ue_bridge_default_joint_pos", "ue_observation_reference", "ue_task_profile",
    )
    ue_backend = metadata.get("unrealzoo_backend") == "ue_mujoco"
    declares_ue_task = metadata.get("task_id") in UE_TASK_PROFILES
    if not (ue_backend or declares_ue_task or any(key in metadata for key in calibration_keys)):
        return None
    for key in calibration_keys:
        if key not in metadata:
            raise ValueError("Required UE policy metadata missing: {}".format(key))
    if metadata["ue_task_profile"] not in UE_TASK_PROFILES:
        raise ValueError("Unsupported UE task profile: {}".format(metadata["ue_task_profile"]))
    if metadata["ue_observation_reference"] != "reset_control_targets":
        raise ValueError("UE policy must declare reset_control_targets observation reference")
    if "task_id" in metadata and metadata["task_id"] != metadata["ue_task_profile"]:
        raise ValueError("UE task_id disagrees with ue_task_profile")
    if ue_backend:
        if "control_dt" not in metadata:
            raise ValueError("Required UE policy metadata missing: control_dt")
        control_dt = float(metadata["control_dt"])
        if not math.isfinite(control_dt) or not math.isclose(
            control_dt, CONTROL_PERIOD, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("UE policy control_dt must be 0.02 seconds")
    return finite_vector(
        metadata["ue_bridge_default_joint_pos"].split(","),
        12,
        "ue_bridge_default_joint_pos",
    )


def prepare_policy_observation(observation, command, metadata, generic_bridge_default):
    """Translate raw UE input into the reference expected by PretrainedPolicy.

    PretrainedPolicy applies ``joint_pos + generic_default - policy_default``.
    New UE policies need ``raw + actual_reset_reference - policy_default``, so
    compensate for the generic default exactly once before calling that adapter.
    Old checkpoints retain the existing generic adapter path.
    """
    result = finite_vector(observation, 48, "UE observation")
    result[45:48] = finite_vector(command, 3, "command")
    reference = ue_observation_reference(metadata)
    if reference is not None:
        generic = finite_vector(generic_bridge_default, 12, "generic bridge default")
        result[9:21] = [
            value + actual - assumed
            for value, actual, assumed in zip(result[9:21], reference, generic)
        ]
    return result


def validate_reset_observation_reference(state, metadata):
    """Reject a changed runtime joint reference before applying a new UE policy."""
    expected = ue_observation_reference(metadata)
    if expected is None:
        return {"mode": "legacy_bridge_default"}
    reset_time = float(state["sim_time"])
    if not math.isfinite(reset_time) or abs(reset_time) > 1e-8:
        raise RuntimeError("UE-trained policy calibration requires reset sim_time=0")
    actual = finite_vector(state.get("control_targets", []), 12, "reset control_targets")
    if any(abs(left - right) > 1e-6 for left, right in zip(actual, expected)):
        raise RuntimeError("UE reset joint reference differs from checkpoint calibration")
    return {"mode": "checkpoint_reset_reference", "joint_reference": actual}


def wait_spawn_camera_ready(env, timeout, proc=None):
    """Let a camera needed for spawning register before pausing the UE world."""
    if env.actor_name or env.spawn_location is not None:
        return None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Camera startup timeout must be positive and finite")
    command = "vget /camera/{}/location".format(env.spawn_camera_id)
    started = time.monotonic()
    deadline = started + timeout
    original_timeout = env.request_timeout
    attempts = 0
    try:
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError("Owned UE exited before the spawn camera registered")
            env.request_timeout = min(original_timeout, max(0.001, deadline - time.monotonic()))
            attempts += 1
            try:
                response = check_response(env.request(command), command)
            except RuntimeError as error:
                if "invalid sensor id" not in str(error).lower():
                    raise
                time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
                continue
            location = finite_vector(response.replace(",", " ").split(), 3, "camera location")
            return {
                "attempts": attempts,
                "wall_seconds": time.monotonic() - started,
                "location_ue_cm": location,
            }
        raise TimeoutError("Spawn camera did not register within {} seconds".format(timeout))
    finally:
        env.request_timeout = original_timeout


def reset_with_policy_calibration(env, metadata, startup_timeout=60.0, proc=None):
    """Reset new UE policies while paused, then restore the original world state."""
    if ue_observation_reference(metadata) is None:
        return env.reset(), {"mode": "legacy_bridge_default"}
    camera_ready = wait_spawn_camera_ready(env, startup_timeout, proc)
    query = "vget /action/game/is_paused"
    original_state = check_response(env.request(query), query).lower()
    if original_state not in ("true", "false"):
        raise RuntimeError("Unexpected UE pause state: {}".format(original_state))
    try:
        env.request("vset /action/game/pause")
        if check_response(env.request(query), query).lower() != "true":
            raise RuntimeError("UE must be paused during observation calibration")
        observation = env.reset()
        calibration = validate_reset_observation_reference(env.state, metadata)
        calibration.update({
            "world_paused_during_reset": True,
            "original_world_paused": original_state == "true",
            "spawn_camera_ready": camera_ready,
        })
        return observation, calibration
    finally:
        active_error = sys.exc_info()[1]
        restore = "pause" if original_state == "true" else "resume"
        try:
            env.request("vset /action/game/{}".format(restore))
            if check_response(env.request(query), query).lower() != original_state:
                raise RuntimeError("UE world pause state did not restore")
        except Exception as restore_error:
            if active_error is not None:
                raise RuntimeError(
                    "{}; restoring UE pause state also failed: {}".format(
                        active_error, restore_error
                    )
                ) from active_error
            raise


def validate_policy_contract(session):
    """Fail before creating a UE actor if shape or control semantics differ."""
    for direction, nodes, width in (
        ("input", session.get_inputs(), 48),
        ("output", session.get_outputs(), 12),
    ):
        if len(nodes) != 1:
            raise ValueError("Expected one ONNX {}".format(direction))
        node = nodes[0]
        if len(node.shape) != 2 or node.shape[1] != width:
            raise ValueError("ONNX {} must have shape [batch, {}]".format(direction, width))
        if isinstance(node.shape[0], int) and node.shape[0] != 1:
            raise ValueError("ONNX {} must support batch size 1".format(direction))
        if node.type != "tensor(float)":
            raise ValueError("ONNX {} must use float32".format(direction))
    metadata = dict(session.get_modelmeta().custom_metadata_map)
    return validate_policy_metadata(metadata)


def validate_policy_metadata(metadata):
    """Validate joint/control semantics for either supported checkpoint format."""
    for key, expected in (
        ("joint_names", JOINT_NAMES),
        ("observation_names", OBSERVATION_NAMES),
        ("command_names", ("twist",)),
    ):
        actual = tuple(item.strip() for item in metadata.get(key, "").split(","))
        if actual != expected:
            raise ValueError("{} does not match the UE Go1 velocity profile".format(key))
    for key, expected in PROFILE.items():
        if key not in metadata:
            raise ValueError("Required policy metadata missing: {}".format(key))
        actual = finite_vector(metadata[key].split(","), 12, key)
        # Original MJLab exporter rounds metadata to three decimal places.
        if any(abs(left - right) > 0.00051 for left, right in zip(actual, expected)):
            raise ValueError("{} differs from the UE Go1 velocity profile".format(key))
    ue_observation_reference(metadata)
    return metadata


def load_policy(checkpoint, device="cpu", threads=1):
    """Load an explicit format without importing ONNX Runtime for Torch playback."""
    checkpoint = Path(checkpoint)
    if checkpoint.suffix.lower() == ".pt":
        from ue_go1_checkpoint import load_ue_checkpoint

        policy, metadata = load_ue_checkpoint(
            checkpoint, device, PROFILE["default_joint_pos"], threads=threads
        )
        return policy, validate_policy_metadata(metadata)
    if checkpoint.suffix.lower() == ".onnx":
        if device != "cpu":
            raise ValueError("Legacy ONNX playback uses CPU; --device applies to .pt checkpoints")
        from common.pretrained_policy import PretrainedPolicy

        policy = PretrainedPolicy("go1", policy_path=str(checkpoint))
        return policy, validate_policy_contract(policy.checkpoint.session)
    raise ValueError("--checkpoint must be an explicit UE .pt or Go1 .onnx file")


def check_response(response, command):
    if response is None:
        raise RuntimeError("No UE response: {}".format(command))
    text = response.decode("utf-8") if isinstance(response, bytes) else str(response)
    text = text.strip()
    if not text or text.lower() in ("none", "null"):
        raise RuntimeError("Empty UE response: {}".format(command))
    if text.lower().startswith(("error", "failed", "unknown command")):
        raise RuntimeError("UE rejected {}: {}".format(command, text[:1000]))
    if text.startswith("{"):
        payload = json.loads(text)
        if payload.get("error") or payload.get("success") is False or payload.get("status") == "error":
            raise RuntimeError("UE error for {}: {}".format(command, text[:1000]))
    return text


def validate_state(state, previous_time=None):
    obs = finite_vector(state["obs"], 48, "UE observation")
    if state.get("policy_profile") != "velocity":
        raise RuntimeError("UE must report policy_profile=velocity")
    if state.get("synchronous") is not True:
        raise RuntimeError("UE synchronous policy control is not active")
    now = float(state["sim_time"])
    if not math.isfinite(now) or now < 0:
        raise RuntimeError("UE returned invalid sim_time")
    if previous_time is not None and not math.isclose(
        now - previous_time, CONTROL_PERIOD, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RuntimeError(
            "UE advanced {:.9f}s; expected 0.020s per control step".format(now - previous_time)
        )
    return obs, now


def environment_class(compat_v310, task_profile=UE_TASK_PROFILE):
    from gym_unrealcv.envs.mujoco import UnrealCvMujocoEnv

    class CheckedGo1Env(UnrealCvMujocoEnv):
        def __init__(self, **options):
            # Prevent settings or a custom config being silently ignored by the
            # legacy path; this entry point deliberately exposes neither.
            if options.get("physics_config") is not None or options.get("setting_file") is not None:
                raise ValueError("Playback accepts only the known Go1 velocity physics profile")
            super().__init__("go1", **options)

        def request(self, command):
            return check_response(super().request(command), command)

        def configure_mujoco_physics(self, actor_name):
            if not compat_v310:
                result = super().configure_mujoco_physics(actor_name)
                if task_profile == "ue_keyboard_flat_v2":
                    # Select physics before the preview model is constructed.
                    # Actor playback needs only the 48 policy observations.
                    self.request(f"vset /object/{actor_name}/mujoco_go1_training_profile keyboard_v2")
                return result
            if task_profile == "ue_keyboard_flat_v2":
                raise ValueError("keyboard v2 playback requires the matching plugin, not v3.1.0 compatibility")
            self.effective_physics_config = {
                "compatibility": "explicit-v3.1.0-go1-velocity",
                "source": "engine_defaults",
                "custom_parameters_applied": False,
                "missing_endpoint": "mujoco_physics_config",
                "expected_timestep_seconds": 0.005,
                "expected_control_decimation": 4,
                "full_runtime_parameters_verified": False,
                "verification": "policy_profile and actual 20ms step checked during rollout; PD/solver not queryable",
            }
            return copy.deepcopy(self.effective_physics_config)

    return CheckedGo1Env


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="UE-trained .pt or legacy Go1 .onnx; no fallback")
    parser.add_argument("--device", default="cpu", help="Torch .pt inference device; ONNX uses CPU")
    parser.add_argument("--threads", type=int, default=1, help="Torch .pt CPU thread count")
    parser.add_argument("--verify-only", action="store_true", help="Validate checkpoint without connecting to UE")
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--steps", type=int, default=1000, help="Control steps; 0 allowed for interactive keyboard")
    parser.add_argument("--warmup-steps", type=int, default=25, help="Zero-command settling steps, excluded from tracking metrics")
    parser.add_argument("--command-vx", type=float, default=0.5)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--keyboard-speed", type=float, default=0.5)
    parser.add_argument("--keyboard-yaw-rate", type=float, default=0.8)
    parser.add_argument("--hz", type=float, default=0.0, help="Wall-clock pacing, 0=unpaced (keyboard defaults to 50Hz)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19025)
    parser.add_argument("--actor", default="", help="Attach to an existing robot instead of spawning")
    parser.add_argument("--spawn", type=float, nargs=3, metavar=("X_CM", "Y_CM", "Z_CM"))
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--keep-actor", action="store_true")
    parser.add_argument("--compat-v310", action="store_true", help="Explicitly accept unqueryable v3.1.0 default physics")
    parser.add_argument("--ue-binary", type=Path, help="Launch and own this packaged Linux executable")
    parser.add_argument("--render", action="store_true", help="Render the owned UE process instead of NullRHI")
    parser.add_argument("--offscreen", action="store_true", help="Use RenderOffScreen with --render")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--fall-upright", type=float, default=math.cos(math.radians(70.0)), help="Fall when -projected_gravity_z is below this threshold; default cos(70deg), matching the source task")
    parser.add_argument("--continue-after-fall", action="store_true")
    parser.add_argument("--output", type=Path, help="JSON metrics destination")
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file() or args.checkpoint.suffix.lower() not in (".pt", ".onnx"):
        parser.error("--checkpoint must name an existing .pt or .onnx file")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.steps < 0 or (args.steps == 0 and not args.keyboard):
        parser.error("--steps must be positive except in keyboard mode")
    if args.warmup_steps < 0 or (args.steps and args.warmup_steps >= args.steps):
        parser.error("--warmup-steps must be non-negative and smaller than --steps")
    if not -1.0 <= args.fall_upright <= 1.0:
        parser.error("--fall-upright must be in [-1, 1]")
    if not math.isfinite(args.hz) or args.hz < 0:
        parser.error("--hz must be finite and non-negative")
    if args.offscreen and not args.render:
        parser.error("--offscreen requires --render")
    if (args.render or args.offscreen) and not args.ue_binary:
        parser.error("Rendering mode is controlled by the existing process; --render needs --ue-binary")
    if args.ue_binary and (not sys.platform.startswith("linux") or args.host not in ("127.0.0.1", "localhost")):
        parser.error("--ue-binary launches only a local Linux process")
    try:
        finite_vector([args.command_vx, args.command_vy, args.command_yaw], 3, "command")
    except ValueError as error:
        parser.error(str(error))
    return args


def launch_owned(args, output_dir):
    binary = args.ue_binary.expanduser().resolve(strict=True)
    # Refuse an occupied port before launching; do not attach to or kill its owner.
    with socket.socket() as listener:
        listener.bind((args.host, args.port))
    run_dir = output_dir / ("ue-{}-{}".format(args.port, os.getpid()))
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(binary), binary.parents[2].name, "-nosound", "-unattended", "-nosplash",
        "-NoVSync", "-NoSaveConfig", "-UnrealCVPort={}".format(args.port),
        "-UserDir={}".format(run_dir / "user"), "-abslog={}".format(run_dir / "ue.log"),
        "-ini:Engine:[/Script/Engine.Engine]:bUseFixedFrameRate=False,[/Script/Engine.Engine]:bSmoothFrameRate=False",
    ]
    if not args.render:
        command.append("-NullRHI")
    elif args.offscreen:
        command.append("-RenderOffScreen")
    with (run_dir / "stdout.log").open("wb") as stream:
        proc = subprocess.Popen(command, cwd=binary.parents[3], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, command


def wait_owned(proc, host, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Owned UE exited with code {}".format(proc.returncode))
        # Verify ownership instead of making a disposable UnrealCV connection.
        listing = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, check=True).stdout
        if any(":{} ".format(port) in line and "pid={},".format(proc.pid) in line for line in listing.splitlines()):
            return
        time.sleep(0.2)
    raise TimeoutError("Owned UE did not listen on {}:{}".format(host, port))


def stop_owned(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=4)


def summarize(samples, warmup_steps, elapsed, fall_upright):
    selected = [sample for sample in samples if sample["step"] > warmup_steps]
    first_fall = next((sample["step"] for sample in samples if sample["upright"] < fall_upright), None)
    tracking = None
    if selected:
        errors = [
            [sample["linvel"][0] - sample["command"][0],
             sample["linvel"][1] - sample["command"][1],
             sample["gyro"][2] - sample["command"][2]]
            for sample in selected
        ]
        tracking = {
            "axes": ["body_vx_m_s", "body_vy_m_s", "body_yaw_rate_rad_s"],
            "rmse": [math.sqrt(statistics.mean(row[i] ** 2 for row in errors)) for i in range(3)],
            "mae": [statistics.mean(abs(row[i]) for row in errors) for i in range(3)],
            "samples": len(selected),
        }
    return {
        "steps_completed": len(samples),
        "rollout_wall_seconds": elapsed,
        "control_steps_per_wall_second": len(samples) / elapsed if elapsed > 0 else None,
        "simulated_seconds_advanced": len(samples) * CONTROL_PERIOD,
        "last_sim_time": samples[-1]["sim_time"] if samples else None,
        "tracking_after_warmup": tracking,
        "minimum_upright": min((sample["upright"] for sample in samples), default=None),
        "mean_upright": statistics.mean(sample["upright"] for sample in samples) if samples else None,
        "fall_detected": first_fall is not None,
        "first_fall_step": first_fall,
        "fall_definition": "-projected_gravity_z < {}; tilt proxy, not contact-based termination".format(fall_upright),
        "control_target_clip_count": sum(sample["control_clip_count"] for sample in samples),
    }


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "example" / "mujoco"))
    import numpy as np
    from common.command_source import FixedCommand, KeyboardCommand

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    policy, metadata = load_policy(checkpoint, args.device, args.threads)
    output = (args.output or ROOT / "artifacts" / "go1_playback" / (
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json"
    )).expanduser().resolve()
    result = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "policy_metadata": metadata, "observation_dim": 48, "action_dim": 12,
        "checkpoint_format": checkpoint.suffix.lower(), "inference_device": args.device,
        "inference_threads": args.threads if checkpoint.suffix.lower() == ".pt" else None,
        "evaluation": "single UE episode; no reward-based training or automatic reset",
        "compat_v310": args.compat_v310, "requested_steps": args.steps,
        "warmup_steps": args.warmup_steps, "complete": False,
        "rendering": ("rendered" if args.render else "NullRHI") if args.ue_binary else "external_process",
    }
    if args.verify_only:
        print(json.dumps({"policy_contract_valid": True, **result}, indent=2))
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    proc, env, samples = None, None, []
    rollout_start = None
    try:
        if args.ue_binary:
            proc, result["ue_launch_command"] = launch_owned(args, output.parent)
            result["owned_ue_pid"] = proc.pid
            wait_owned(proc, args.host, args.port, args.startup_timeout)
        env = environment_class(args.compat_v310, metadata.get("ue_task_profile", UE_TASK_PROFILE))(
            host=args.host, port=args.port, actor_name=args.actor,
            spawn_location=args.spawn, spawn_yaw_offset=args.spawn_yaw,
            keep_actor=args.keep_actor, launch=False, physics_config=None,
            setting_file=None, request_timeout=args.request_timeout,
        )
        initial_command = np.zeros(3, dtype=np.float32)
        env.set_command(initial_command)
        observation, result["observation_calibration"] = reset_with_policy_calibration(
            env, metadata, args.startup_timeout, proc
        )
        if metadata.get("ue_task_profile") == "ue_keyboard_flat_v2":
            from ue_go1_env import validate_training_physics
            validate_training_physics(env.state)
        _, previous_time = validate_state(env.state)
        result["initial_sim_time"] = previous_time
        result["physics_config"] = copy.deepcopy(env.effective_physics_config)
        result["actor"] = env.actor_name
        policy.reset()
        source = KeyboardCommand(args.keyboard_speed, args.keyboard_yaw_rate) if args.keyboard else FixedCommand(
            [args.command_vx, args.command_vy, args.command_yaw]
        )
        hz = args.hz or (50.0 if args.keyboard else 0.0)
        result["wall_clock_pacing_hz"] = hz
        if args.keyboard:
            print("KEYBOARD|I/K=forward/back|J/L=turn|Space=stop|X/Esc=exit", flush=True)
        rollout_start = time.perf_counter()
        with source:
            while not args.steps or len(samples) < args.steps:
                command, exit_requested = source.read()
                if exit_requested:
                    result["stop_reason"] = "keyboard_exit"
                    break
                if len(samples) < args.warmup_steps:
                    command = np.zeros(3, dtype=np.float32)
                command = np.asarray(command, dtype=np.float32)
                env.set_command(command)
                # set_command updates UE, but the cached observation still holds
                # the previous command; make this policy input current.
                policy_observation = prepare_policy_observation(
                    observation, command, metadata, PROFILE["default_joint_pos"]
                )
                action = policy.act(policy_observation, command)
                finite_vector(action, 12, "policy action")
                observation, _, _, info = env.step(action)
                state, previous_time = validate_state(info, previous_time)
                sample = {
                    "step": len(samples) + 1, "sim_time": previous_time,
                    "command": command.tolist(), "linvel": state[0:3], "gyro": state[3:6],
                    "upright": -state[8], "control_clip_count": int(info.get("control_clip_count", 0)),
                }
                samples.append(sample)
                if len(samples) % 100 == 0:
                    print("STEP|{}|sim_time={:.3f}|upright={:.3f}".format(len(samples), previous_time, sample["upright"]), flush=True)
                if sample["upright"] < args.fall_upright and not args.continue_after_fall:
                    result["stop_reason"] = "fall"
                    break
                if hz:
                    delay = rollout_start + len(samples) / hz - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
        result.setdefault("stop_reason", "steps_completed")
        result["complete"] = True
    except KeyboardInterrupt:
        result["stop_reason"] = "keyboard_interrupt"
    except Exception as error:
        result["error"] = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        elapsed = time.perf_counter() - rollout_start if rollout_start is not None else 0.0
        result["metrics"] = summarize(samples, args.warmup_steps, elapsed, args.fall_upright)
        result["samples"] = samples
        if env is not None:
            try:
                env.close()
            except Exception as error:
                result["cleanup_error"] = str(error)
        if proc is not None:
            stop_owned(proc)
            result["owned_ue_exit_code"] = proc.returncode
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("RESULT|{}".format(output), flush=True)
    print(json.dumps(result["metrics"], indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
