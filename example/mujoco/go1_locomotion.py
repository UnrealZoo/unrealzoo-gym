#!/usr/bin/env python3
"""
Run a simple policy-control loop for the UnrealCV MuJoCo Unitree Go1 bridge.

The UE-side bridge owns MuJoCo simulation. This script only:

1. connects to an already running UnrealCV session
2. spawns or reuses the Go1 actor
3. starts the continuous Go1 MuJoCo mode
4. reads the 48D policy observation
5. sends a 12D normalized action back to Unreal

Use --policy path.onnx for an ONNX locomotion model. Use --policy none to verify
the UnrealCV/MuJoCo control plumbing with zero actions.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unrealcv  # noqa: E402


GO1_BP_PATH = "/Game/robot-dog-unitree-go1/BP_UnitreeGo1.BP_UnitreeGo1"
GO1_ACTOR_PREFIX = "BP_UnitreeGo1"


def configure_go1_blueprint(path):
    """Use the Go1 spawn path indexed by the selected environment JSON."""
    global GO1_BP_PATH, GO1_ACTOR_PREFIX
    value = str(path).strip()
    if not value.startswith("/Game/"):
        raise ValueError(f"Invalid Go1 Blueprint path: {path!r}")
    GO1_BP_PATH = value
    object_name = value.rsplit("/", 1)[-1].split(".", 1)[0]
    GO1_ACTOR_PREFIX = object_name.removesuffix("_C")
DEFAULT_SPAWN = (0.0, 0.0, 500.0)
OBS_DIM = 48
ACTION_DIM = 12
JOINT_NAMES = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)
BRIDGE_DEFAULT_JOINT_POS = (
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
)
BRIDGE_ACTION_SCALE = 0.5


def connect_client(host, port):
    client = unrealcv.Client((host, port))
    client.connect()
    if not client.isconnected():
        raise RuntimeError(f"Failed to connect UnrealCV client to {host}:{port}")
    return client


def request(client, command, timeout=None, verbose=True):
    if timeout is None:
        response = client.request(command)
    else:
        response = client.request(command, timeout)
    if verbose:
        print(f"CMD|{command}")
        print(f"RES|{response}")
    return response


def spawn_go1(client, location):
    x, y, z = location
    response = request(
        client,
        f"vset /objects/spawn_from_path {GO1_BP_PATH} {x:.6f} {y:.6f} {z:.6f}",
    )
    actor_name = str(response).strip()
    if not actor_name or actor_name.lower().startswith("error"):
        raise RuntimeError(f"Failed to spawn Go1 actor: {response}")
    if not actor_name.startswith(GO1_ACTOR_PREFIX):
        raise RuntimeError(f"Unexpected Go1 spawn response: {response}")
    time.sleep(1.0)
    return actor_name


def add_demo_spawn_arguments(parser, default_location=None):
    default_location = tuple(default_location) if default_location is not None else (None, None, None)
    if len(default_location) != 3:
        raise ValueError("default_location must contain exactly three values")
    default_description = (
        "the fixed demo start"
        if all(value is not None for value in default_location)
        else "camera-relative spawn"
    )
    parser.add_argument(
        "--spawn-x",
        type=float,
        default=default_location[0],
        help=f"UE world X in cm; defaults to {default_description}",
    )
    parser.add_argument("--spawn-y", type=float, default=default_location[1], help="UE world Y in cm")
    parser.add_argument("--spawn-z", type=float, default=default_location[2], help="UE world Z in cm")
    parser.add_argument("--spawn-camera-id", default="0", help="Camera used by automatic spawn")
    parser.add_argument(
        "--spawn-camera-forward-cm",
        type=float,
        default=300.0,
        help="Automatic spawn distance in front of the camera",
    )
    parser.add_argument(
        "--spawn-camera-right-cm",
        type=float,
        default=0.0,
        help="Automatic spawn offset to the camera's right",
    )
    parser.add_argument(
        "--spawn-camera-up-cm",
        type=float,
        default=100.0,
        help="Temporary height above the camera before ground settling",
    )
    parser.add_argument(
        "--spawn-yaw-offset",
        type=float,
        default=0.0,
        help="Yaw offset from camera yaw, or world yaw for explicit XYZ spawn",
    )
    parser.add_argument(
        "--settle-to-ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Settle the spawned Go1 onto collision geometry before starting MuJoCo",
    )
    parser.add_argument(
        "--settle-mode",
        choices=("simple", "bounds"),
        default="simple",
        help="Go1 fallback-spawn settling mode; simple ignores sensor component bounds",
    )
    parser.add_argument(
        "--settle-height-offset-cm",
        type=float,
        default=35.0,
        help="Root height above the hit surface in simple mode, or clearance in bounds mode",
    )
    parser.add_argument(
        "--settle-trace-start-cm",
        type=float,
        default=100.0,
        help="Start the floor trace this far above the requested spawn point",
    )
    parser.add_argument(
        "--settle-trace-length-cm",
        type=float,
        default=5000.0,
        help="Maximum downward floor-trace distance from the local trace start",
    )
    parser.add_argument(
        "--settle-max-upward-cm",
        type=float,
        default=150.0,
        help="Reject settling that unexpectedly raises Go1 by more than this amount",
    )
    parser.add_argument(
        "--keep-actor",
        action="store_true",
        help="Leave the spawned Go1 in the world when the demo exits",
    )


def validate_demo_spawn_arguments(parser, args):
    explicit_spawn = (args.spawn_x, args.spawn_y, args.spawn_z)
    if any(value is not None for value in explicit_spawn) and not all(
        value is not None for value in explicit_spawn
    ):
        parser.error("--spawn-x, --spawn-y, and --spawn-z must be supplied together")

    numeric_values = (
        *(value for value in explicit_spawn if value is not None),
        args.spawn_camera_forward_cm,
        args.spawn_camera_right_cm,
        args.spawn_camera_up_cm,
        args.spawn_yaw_offset,
        args.settle_height_offset_cm,
        args.settle_trace_start_cm,
        args.settle_trace_length_cm,
        args.settle_max_upward_cm,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        parser.error("spawn and ground-settling values must be finite")
    if args.settle_trace_start_cm < 0.0:
        parser.error("--settle-trace-start-cm must be >= 0")
    if args.settle_trace_length_cm <= 0.0:
        parser.error("--settle-trace-length-cm must be > 0")
    if args.settle_max_upward_cm < 0.0:
        parser.error("--settle-max-upward-cm must be >= 0")


def _parse_vector3_response(response, operation):
    try:
        values = tuple(float(value) for value in str(response).strip().split())
    except ValueError as exc:
        raise RuntimeError(f"Failed to {operation}: {response!r}") from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Failed to {operation}: {response!r}")
    return values


def find_go1_actors(client):
    response = request(client, "vget /objects", verbose=False)
    text = str(response).strip()
    if not text or text.casefold().startswith("error"):
        raise RuntimeError(f"Failed to enumerate actors: {response!r}")
    return sorted(
        name for name in text.split() if name.casefold().startswith(GO1_ACTOR_PREFIX.casefold())
    )


def acquire_go1_for_demo(client, args):
    candidates = find_go1_actors(client)
    stale_candidates = []
    for actor_name in reversed(candidates):
        location_response = request(
            client, f"vget /object/{actor_name}/location", verbose=False
        )
        try:
            location = _parse_vector3_response(
                location_response, "read existing Go1 location"
            )
        except RuntimeError:
            stale_candidates.append(actor_name)
            print(
                f"GO1_STALE|actor={actor_name}|response={str(location_response).strip()}"
            )
            continue
        print(
            f"GO1_REUSE|actor={actor_name}|owned=no|"
            f"location_cm={','.join(f'{value:.3f}' for value in location)}|"
            f"candidate_count={len(candidates)}|candidates={','.join(candidates)}"
        )
        return actor_name, False

    if stale_candidates:
        print(f"GO1_STALE_ONLY|candidates={','.join(stale_candidates)}|fallback=spawn")

    actor_name = spawn_go1_for_demo(client, args)
    print(f"GO1_ACQUIRE|actor={actor_name}|owned=yes|source=spawn")
    return actor_name, True


def demo_start_location(args):
    """Use the old fixed test pose unless the caller supplies all three XYZ values."""
    explicit = (args.spawn_x, args.spawn_y, args.spawn_z)
    if all(value is not None for value in explicit):
        return tuple(float(value) for value in explicit)
    return DEFAULT_SPAWN


def place_go1_at_location(client, actor_name, location=DEFAULT_SPAWN):
    """Move a Go1 through the direct UnrealCV object interface and verify it."""
    target = tuple(float(value) for value in location)
    if len(target) != 3 or not all(math.isfinite(value) for value in target):
        raise ValueError(f"Invalid Go1 demo start location: {location!r}")

    response = request(
        client,
        f"vset /object/{actor_name}/location "
        + " ".join(f"{value:.6f}" for value in target),
        verbose=False,
    )
    if str(response).strip().casefold().startswith("error"):
        raise RuntimeError(f"Failed to move Go1 to the fixed demo start: {response}")
    actual = _parse_vector3_response(
        request(client, f"vget /object/{actor_name}/location", verbose=False),
        "verify fixed Go1 demo location",
    )
    if any(abs(actual[index] - target[index]) > 1.0 for index in range(3)):
        raise RuntimeError(
            "Failed to place Go1 at the fixed demo start: "
            f"requested={target!r}, actual={actual!r}"
        )
    print(
        f"GO1_FIXED_START|actor={actor_name}|"
        f"location_cm={','.join(f'{value:.3f}' for value in actual)}"
    )
    return actual


def place_go1_after_navigation_reset(env, actor_name, location=DEFAULT_SPAWN):
    """Move the acquired Go1 to the deterministic demo start after Gym reset."""
    return place_go1_at_location(env.unwrapped.unrealcv.client, actor_name, location)


def spawn_go1_for_demo(client, args):
    explicit_values = (args.spawn_x, args.spawn_y, args.spawn_z)
    has_explicit_value = tuple(value is not None for value in explicit_values)
    if any(has_explicit_value) and not all(has_explicit_value):
        raise RuntimeError("Explicit spawn requires --spawn-x, --spawn-y, and --spawn-z")

    if all(has_explicit_value):
        requested_location = tuple(float(value) for value in explicit_values)
        spawn_yaw = float(args.spawn_yaw_offset)
        source = "explicit-world"
        camera_location = None
    else:
        camera_id = str(args.spawn_camera_id)
        try:
            camera_location = _parse_vector3_response(
                request(client, f"vget /camera/{camera_id}/location", verbose=False),
                f"read camera {camera_id} location",
            )
            camera_rotation = _parse_vector3_response(
                request(client, f"vget /camera/{camera_id}/rotation", verbose=False),
                f"read camera {camera_id} rotation",
            )
        except RuntimeError as exc:
            camera_location = None
            requested_location = DEFAULT_SPAWN
            spawn_yaw = float(args.spawn_yaw_offset)
            source = f"camera:{camera_id}-unavailable-fallback-world"
            print(
                f"SPAWN_CAMERA_FALLBACK|camera={camera_id}|"
                f"location_cm={','.join(f'{value:.3f}' for value in DEFAULT_SPAWN)}|"
                f"reason={exc}"
            )
        else:
            yaw_radians = math.radians(camera_rotation[1])
            forward = float(args.spawn_camera_forward_cm)
            right = float(args.spawn_camera_right_cm)
            requested_location = (
                camera_location[0]
                + math.cos(yaw_radians) * forward
                - math.sin(yaw_radians) * right,
                camera_location[1]
                + math.sin(yaw_radians) * forward
                + math.cos(yaw_radians) * right,
                camera_location[2] + float(args.spawn_camera_up_cm),
            )
            spawn_yaw = camera_rotation[1] + float(args.spawn_yaw_offset)
            source = f"camera:{camera_id}"

    actor_name = spawn_go1(client, requested_location)
    try:
        rotation_response = request(
            client,
            f"vset /object/{actor_name}/rotation 0 {spawn_yaw:.6f} 0",
            verbose=False,
        )
        if str(rotation_response).strip().casefold().startswith("error"):
            raise RuntimeError(f"Failed to orient spawned Go1: {rotation_response}")
        pre_settle_location = _parse_vector3_response(
            request(client, f"vget /object/{actor_name}/location", verbose=False),
            "read spawned Go1 pre-settle location",
        )
        if args.settle_to_ground:
            settle_response = request(
                client,
                f"vset /object/{actor_name}/settle_to_ground {args.settle_mode} "
                f"{args.settle_trace_start_cm:.6f} {args.settle_trace_length_cm:.6f} "
                f"{args.settle_height_offset_cm:.6f}",
                verbose=False,
            )
            if str(settle_response).strip().casefold().startswith("error"):
                raise RuntimeError(f"Failed to settle spawned Go1: {settle_response}")
        actual_location = _parse_vector3_response(
            request(client, f"vget /object/{actor_name}/location", verbose=False),
            "read spawned Go1 location",
        )
        spawn_delta_z = pre_settle_location[2] - requested_location[2]
        settle_delta_z = actual_location[2] - pre_settle_location[2]
        if args.settle_to_ground and settle_delta_z > args.settle_max_upward_cm:
            raise RuntimeError(
                "Ground settling moved Go1 upward unexpectedly: "
                f"pre_settle_z={pre_settle_location[2]:.3f}cm, "
                f"actual_z={actual_location[2]:.3f}cm, "
                f"delta_z={settle_delta_z:.3f}cm. "
                "The trace probably hit an overhead floor or roof; reduce "
                "--settle-trace-start-cm."
            )
    except Exception:
        try:
            destroy_actor(client, actor_name)
        except Exception:
            pass
        raise

    camera_text = "none" if camera_location is None else ",".join(
        f"{value:.3f}" for value in camera_location
    )
    print(
        f"SPAWN|source={source}|camera_cm={camera_text}|"
        f"requested_cm={','.join(f'{value:.3f}' for value in requested_location)}|"
        f"pre_settle_cm={','.join(f'{value:.3f}' for value in pre_settle_location)}|"
        f"actual_cm={','.join(f'{value:.3f}' for value in actual_location)}|"
        f"spawn_delta_z_cm={spawn_delta_z:.3f}|"
        f"settle_delta_z_cm={settle_delta_z:.3f}|"
        f"yaw={spawn_yaw:.3f}|settled={'yes' if args.settle_to_ground else 'no'}"
    )
    return actor_name


def destroy_actor(client, actor_name):
    response = request(client, f"vset /object/{actor_name}/destroy", verbose=False)
    if str(response).strip().lower().startswith("error"):
        raise RuntimeError(f"Failed to destroy actor '{actor_name}': {response}")
    print(f"ACTOR_DESTROYED|{actor_name}")


def stop_go1_simulation(client, actor_name, *, required=False):
    response = request(
        client,
        f"vset /object/{actor_name}/mujoco_quadruped_pose_preview/stop",
        verbose=False,
    )
    text = str(response).strip()
    if text.casefold().startswith("error"):
        if required:
            raise RuntimeError(f"Failed to stop existing Go1 MuJoCo simulation: {response}")
        print(f"MUJOCO_STOP_WARNING|actor={actor_name}|response={text}")
        return False
    print(f"MUJOCO_STOP|actor={actor_name}|response={text}")
    return True


def parse_start_result(start_result):
    parts = str(start_result).strip().split("|")
    if not parts or not parts[0] or str(parts[0]).lower().startswith("error"):
        raise RuntimeError(f"Unexpected MuJoCo start response: {start_result}")

    trajectory_path = Path(parts[0])
    return {
        "trajectory_path": trajectory_path,
        "state_log_path": Path(parts[1]) if len(parts) > 1 else None,
        "mjcf_path": Path(parts[2]) if len(parts) > 2 else None,
        "initial_snapshot_path": Path(parts[3]) if len(parts) > 3 else None,
        "runtime_snapshot_path": Path(parts[4]) if len(parts) > 4 else None,
        "environment_collision_report_path": Path(parts[5]) if len(parts) > 5 else None,
        "status_path": trajectory_path.with_name(trajectory_path.name.replace("_trajectory.csv", "_status.txt")),
    }


def print_artifact_paths(paths):
    for key, value in paths.items():
        if value is not None:
            print(f"{key.upper()}|{value}")


def parse_vector3(response):
    values = [float(item) for item in str(response).strip().split()]
    if len(values) != 3:
        raise RuntimeError(f"Expected 3D vector response, got: {response}")
    return values


def validate_actor_location(location, origin, max_displacement_cm):
    if not all(math.isfinite(value) for value in location):
        raise RuntimeError(f"Actor location is not finite: location={location}")
    displacement_cm = math.sqrt(
        sum((float(location[i]) - float(origin[i])) ** 2 for i in range(3))
    )
    if displacement_cm > max_displacement_cm:
        raise RuntimeError(
            "Actor appears unstable or out of range: "
            f"location={location}, origin={origin}, "
            f"displacement_cm={displacement_cm:.3f}, "
            f"limit_cm={max_displacement_cm:.3f}"
        )
    return displacement_cm


def parse_policy_observation(response, operation):
    response_text = str(response).strip()
    if not response_text.startswith("{"):
        raise RuntimeError(f"Failed to {operation}: unexpected response: {response_text!r}")
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to {operation}: invalid JSON response: {response_text!r}"
        ) from exc
    if "error" in payload:
        raise RuntimeError(f"Failed to {operation}: {payload['error']}")

    obs = payload.get("obs")
    if not isinstance(obs, list) or len(obs) != OBS_DIM:
        raise RuntimeError(f"Expected {OBS_DIM}D observation, got: {payload}")
    return [float(value) for value in obs], payload


def get_policy_observation(client, actor_name):
    response = request(client, f"vget /object/{actor_name}/mujoco_go1_policy_obs", verbose=False)
    return parse_policy_observation(response, "read Go1 policy obs")


def start_synchronous_policy(client, actor_name):
    response = request(
        client,
        f"vset /object/{actor_name}/mujoco_go1_policy_sync/start",
        verbose=False,
    )
    return parse_policy_observation(response, "start synchronous Go1 policy control")


def step_synchronous_policy(client, actor_name, raw_action):
    if len(raw_action) != ACTION_DIM:
        raise RuntimeError(f"Expected {ACTION_DIM}D raw policy action, got {len(raw_action)}")
    # UnrealCV's [float] command matcher does not accept exponent notation.
    action_text = " ".join(f"{float(value):.9f}" for value in raw_action)
    response = request(
        client,
        f"vset /object/{actor_name}/mujoco_go1_policy_step {action_text}",
        verbose=False,
    )
    return parse_policy_observation(response, "step synchronous Go1 policy control")


def send_policy_command(client, actor_name, command):
    vx, vy, yaw_rate = command
    response = request(
        client,
        f"vset /object/{actor_name}/mujoco_go1_policy_command {vx:.6f} {vy:.6f} {yaw_rate:.6f}",
        verbose=False,
    )
    if str(response).lower().startswith("error"):
        raise RuntimeError(f"Failed to set Go1 policy command: {response}")


def send_policy_action(client, actor_name, action):
    if len(action) != ACTION_DIM:
        raise RuntimeError(f"Expected {ACTION_DIM}D action, got {len(action)}")
    action_text = " ".join(f"{float(value):.6f}" for value in action)
    response = request(
        client,
        f"vset /object/{actor_name}/mujoco_go1_policy_action {action_text}",
        verbose=False,
    )
    if str(response).lower().startswith("error"):
        raise RuntimeError(f"Failed to set Go1 policy action: {response}")


def load_vector(path, expected_dim, name):
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get(name) or payload.get("values")
    if not isinstance(payload, list) or len(payload) != expected_dim:
        raise RuntimeError(f"{name} file must contain a {expected_dim}D list: {path}")
    return [float(value) for value in payload]


class ZeroPolicy:
    def describe(self):
        return "zero-action debug policy"

    def infer(self, obs):
        return [0.0] * ACTION_DIM

    def prepare_obs(self, obs, last_raw_action):
        return obs


class OnnxPolicy:
    def __init__(self, path, obs_mean=None, obs_std=None):
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX policy inference requires onnxruntime and numpy. "
                "Install them into the same Python used to run this script. "
                "For conda, prefer: conda install -c conda-forge onnxruntime numpy. "
                f"Original import error: {exc}"
            ) from exc

        self.np = np
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32) if obs_mean is not None else None
        self.obs_std = np.asarray(obs_std, dtype=np.float32) if obs_std is not None else None
        self.metadata = self.session.get_modelmeta().custom_metadata_map
        self.policy_default = self._parse_metadata_vector("default_joint_pos")
        self.policy_action_scale = self._parse_metadata_vector("action_scale")

    def _parse_metadata_vector(self, key):
        raw_value = self.metadata.get(key)
        if not raw_value:
            return None
        values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
        if len(values) != ACTION_DIM:
            return None
        return self.np.asarray(values, dtype=self.np.float32)

    def describe(self):
        has_action_meta = self.policy_default is not None and self.policy_action_scale is not None
        return (
            f"ONNX input={self.input.name}{self.input.shape} "
            f"output={self.output.name}{self.output.shape} "
            f"action_metadata={'yes' if has_action_meta else 'no'}"
        )

    def infer(self, obs):
        x = self.np.asarray(obs, dtype=self.np.float32)
        if self.obs_mean is not None:
            x = x - self.obs_mean
        if self.obs_std is not None:
            x = x / self.np.maximum(self.obs_std, 1e-6)

        input_shape = self.input.shape
        if len(input_shape) != 1:
            x = x.reshape(1, OBS_DIM)
        result = self.session.run([self.output.name], {self.input.name: x})[0]
        action = self.np.asarray(result, dtype=self.np.float32).reshape(-1)
        if action.size != ACTION_DIM:
            raise RuntimeError(f"Expected ONNX policy to output {ACTION_DIM} actions, got {action.size}")
        return action.tolist()

    def prepare_obs(self, obs, last_raw_action):
        if self.policy_default is None:
            return obs

        policy_obs = list(obs)
        bridge_default = self.np.asarray(BRIDGE_DEFAULT_JOINT_POS, dtype=self.np.float32)
        bridge_joint_delta = self.np.asarray(policy_obs[9:21], dtype=self.np.float32)
        qpos = bridge_joint_delta + bridge_default
        policy_joint_delta = qpos - self.policy_default
        policy_obs[9:21] = policy_joint_delta.tolist()
        policy_obs[33:45] = list(last_raw_action)
        return policy_obs

    def to_bridge_action(self, action):
        if self.policy_default is None or self.policy_action_scale is None:
            return action

        raw = self.np.asarray(action, dtype=self.np.float32)
        target = self.policy_default + raw * self.policy_action_scale
        bridge_default = self.np.asarray(BRIDGE_DEFAULT_JOINT_POS, dtype=self.np.float32)
        bridge_action = (target - bridge_default) / BRIDGE_ACTION_SCALE
        return bridge_action.tolist()


def make_policy(args):
    if args.policy.lower() == "none":
        return ZeroPolicy()

    obs_mean = load_vector(args.obs_mean, OBS_DIM, "obs_mean")
    obs_std = load_vector(args.obs_std, OBS_DIM, "obs_std")
    return OnnxPolicy(Path(args.policy), obs_mean=obs_mean, obs_std=obs_std)


def clamp_action(action, clip, gain):
    clipped = []
    for value in action:
        if not math.isfinite(float(value)):
            raise RuntimeError(f"Policy produced non-finite action: {action}")
        clipped.append(max(-clip, min(clip, float(value) * gain)))
    return clipped


def scale_action(action, gain):
    scaled = []
    for value in action:
        scaled_value = float(value) * gain
        if not math.isfinite(scaled_value):
            raise RuntimeError(f"Policy produced non-finite action: {action}")
        scaled.append(scaled_value)
    return scaled


def clamp_command(command, clip):
    if clip <= 0.0:
        return tuple(float(value) for value in command), False
    clipped = tuple(max(-clip, min(clip, float(value))) for value in command)
    return clipped, clipped != tuple(float(value) for value in command)


def ramp_command(command, elapsed, warmup, ramp_seconds):
    if elapsed < warmup:
        scale = 0.0
    elif ramp_seconds <= 0.0:
        scale = 1.0
    else:
        scale = min(1.0, max(0.0, (elapsed - warmup) / ramp_seconds))
    return tuple(float(value) * scale for value in command), scale


def vector_rms(values):
    if not values:
        return 0.0
    return math.sqrt(sum(float(value) * float(value) for value in values) / len(values))


def upright_score_from_gravity(gravity):
    # In the Go1 policy observation convention, a stable upright base is close to gravity.z = -1.
    return -float(gravity[2])


def format_vector(values, precision=3):
    return "[" + ",".join(f"{float(value):.{precision}f}" for value in values) + "]"


def format_named_vector(names, values, precision=3):
    return "[" + ",".join(f"{name}:{float(value):.{precision}f}" for name, value in zip(names, values)) + "]"


def to_bridge_action(policy, action):
    if hasattr(policy, "to_bridge_action"):
        return policy.to_bridge_action(action)
    return action


def prepare_policy_obs(policy, obs, last_raw_action):
    if hasattr(policy, "prepare_obs"):
        return policy.prepare_obs(obs, last_raw_action)
    return obs


def run_control_loop(client, actor_name, policy, args):
    interval = 1.0 / args.policy_hz
    synchronous = args.control_mode == "sync"
    if synchronous and not math.isclose(args.policy_hz, 50.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError("Synchronous Go1 policy control requires --policy-hz 50")

    requested_command = (args.vx, args.vy, args.yaw_rate)
    target_command, command_was_clipped = clamp_command(requested_command, args.command_clip)
    print(f"CONTROL|mode={args.control_mode}|policy_hz={args.policy_hz:.3f}|sim_dt={interval:.6f}")
    print(
        "COMMAND|"
        f"requested=({requested_command[0]:.3f},{requested_command[1]:.3f},{requested_command[2]:.3f})|"
        f"target=({target_command[0]:.3f},{target_command[1]:.3f},{target_command[2]:.3f})|"
        f"clip={args.command_clip:.3f}|clipped={'yes' if command_was_clipped else 'no'}|"
        f"ramp={args.command_ramp:.3f}"
    )
    last_raw_action = [0.0] * ACTION_DIM
    last_command = None
    if synchronous:
        obs, obs_payload = start_synchronous_policy(client, actor_name)
    else:
        obs = None
        obs_payload = None

    location_origin = parse_vector3(
        request(client, f"vget /object/{actor_name}/location", verbose=False)
    )
    validate_actor_location(location_origin, location_origin, args.max_displacement_cm)
    print(
        "LOCATION_GUARD|"
        f"origin=({location_origin[0]:.3f},{location_origin[1]:.3f},{location_origin[2]:.3f})|"
        f"max_displacement_cm={args.max_displacement_cm:.3f}"
    )

    start_time = time.perf_counter()
    next_tick = start_time
    step = 0
    foot_contact_samples = [0] * 4
    foot_slip_sums = [0.0] * 4
    foot_slip_maxima = [0.0] * 4
    loaded_contact_samples = [0] * 4
    loaded_slip_sums = [0.0] * 4
    loaded_slip_maxima = [0.0] * 4
    calf_error_maxima = [0.0] * 4
    while True:
        iteration_start = time.perf_counter()
        if synchronous:
            elapsed = step * interval
        else:
            elapsed = iteration_start - start_time
        if elapsed >= args.duration:
            break
        if not synchronous and iteration_start < next_tick:
            time.sleep(next_tick - iteration_start)

        active_command, command_scale = ramp_command(target_command, elapsed, args.warmup, args.command_ramp)
        if last_command is None or any(abs(active_command[i] - last_command[i]) > 1e-6 for i in range(3)):
            send_policy_command(client, actor_name, active_command)
            last_command = active_command

        if not synchronous:
            obs, obs_payload = get_policy_observation(client, actor_name)
        policy_obs = prepare_policy_obs(policy, obs, last_raw_action)
        phase = "warmup" if elapsed < args.warmup else "policy"
        if elapsed < args.warmup:
            network_action = [0.0] * ACTION_DIM
        else:
            network_action = scale_action(policy.infer(policy_obs), args.action_gain)

        if synchronous:
            raw_action = clamp_action(network_action, args.raw_action_clip, 1.0)
            bridge_action = to_bridge_action(policy, raw_action)
            next_obs, next_payload = step_synchronous_policy(client, actor_name, raw_action)
            obs = next_obs
            obs_payload = next_payload
            action = bridge_action
        else:
            raw_action = clamp_action(network_action, args.raw_action_clip, 1.0)
            bridge_action = to_bridge_action(policy, raw_action)
            action = clamp_action(bridge_action, args.bridge_action_clip, 1.0)
            send_policy_action(client, actor_name, action)
        last_raw_action = list(raw_action)

        foot_contacts = obs_payload.get("foot_contacts", []) if obs_payload else []
        foot_slip_speeds = obs_payload.get("foot_slip_speeds", []) if obs_payload else []
        foot_normal_forces = obs_payload.get("foot_normal_forces", []) if obs_payload else []
        calf_body_errors = obs_payload.get("calf_body_errors_cm", []) if obs_payload else []
        if len(foot_contacts) == 4 and len(foot_slip_speeds) == 4:
            for foot_index in range(4):
                if bool(foot_contacts[foot_index]):
                    slip_speed = float(foot_slip_speeds[foot_index])
                    foot_contact_samples[foot_index] += 1
                    foot_slip_sums[foot_index] += slip_speed
                    foot_slip_maxima[foot_index] = max(foot_slip_maxima[foot_index], slip_speed)
                    if (
                        len(foot_normal_forces) == 4
                        and float(foot_normal_forces[foot_index]) >= args.loaded_contact_force
                    ):
                        loaded_contact_samples[foot_index] += 1
                        loaded_slip_sums[foot_index] += slip_speed
                        loaded_slip_maxima[foot_index] = max(
                            loaded_slip_maxima[foot_index], slip_speed
                        )
        if len(calf_body_errors) == 4:
            for foot_index in range(4):
                calf_error_maxima[foot_index] = max(
                    calf_error_maxima[foot_index], float(calf_body_errors[foot_index])
                )

        if step % args.print_every == 0:
            location = parse_vector3(request(client, f"vget /object/{actor_name}/location", verbose=False))
            rotation = parse_vector3(request(client, f"vget /object/{actor_name}/rotation", verbose=False))
            displacement_cm = validate_actor_location(
                location, location_origin, args.max_displacement_cm
            )
            linvel = obs_payload.get("linvel", [0.0, 0.0, 0.0])
            gravity = obs_payload.get("gravity", [0.0, 0.0, -1.0])
            joint_delta = obs_payload.get("joint_angles", [0.0] * ACTION_DIM)
            joint_qpos = [BRIDGE_DEFAULT_JOINT_POS[i] + float(joint_delta[i]) for i in range(ACTION_DIM)]
            joint_qvel = obs_payload.get("joint_velocities", [0.0] * ACTION_DIM)
            obs_command = obs_payload.get("command", list(active_command))
            raw_over_limit = sum(1 for value in network_action if abs(float(value)) >= args.raw_action_clip - 1e-6)
            bridge_over_limit = sum(1 for value in bridge_action if abs(float(value)) >= args.bridge_action_clip - 1e-6)
            control_clip_count = int(obs_payload.get("control_clip_count", 0))
            planar_speed = math.sqrt(float(linvel[0]) * float(linvel[0]) + float(linvel[1]) * float(linvel[1]))
            upright_score = upright_score_from_gravity(gravity)
            has_rolled_over = float(gravity[2]) >= args.fall_gravity_z_threshold
            has_lost_upright = args.min_upright_gravity_z > 0.0 and upright_score < args.min_upright_gravity_z
            has_fallen = has_rolled_over or has_lost_upright
            print(
                "STEP|"
                f"{step}|t={elapsed:.3f}|"
                f"phase={phase}|"
                f"cmd=({obs_command[0]:.3f},{obs_command[1]:.3f},{obs_command[2]:.3f})|cmd_scale={command_scale:.3f}|"
                f"loc=({location[0]:.2f},{location[1]:.2f},{location[2]:.2f})|"
                f"displacement_cm={displacement_cm:.2f}|"
                f"rot=({rotation[0]:.2f},{rotation[1]:.2f},{rotation[2]:.2f})|"
                f"linvel=({linvel[0]:.3f},{linvel[1]:.3f},{linvel[2]:.3f})|"
                f"speed={planar_speed:.3f}|"
                f"gravity=({gravity[0]:.3f},{gravity[1]:.3f},{gravity[2]:.3f})|"
                f"upright={upright_score:.3f}|"
                f"raw_rms={vector_rms(raw_action):.3f}|bridge_rms={vector_rms(action):.3f}|"
                f"raw_over={raw_over_limit}/{ACTION_DIM}|bridge_over={bridge_over_limit}/{ACTION_DIM}|"
                f"control_clip={control_clip_count}/{ACTION_DIM}"
            )
            if has_fallen:
                if has_rolled_over:
                    fall_reason = "rolled_over"
                elif has_lost_upright:
                    fall_reason = "lost_upright"
                else:
                    fall_reason = "unknown"
                print(
                    "FALL|"
                    f"step={step}|t={elapsed:.3f}|reason={fall_reason}|"
                    f"gravity_z={float(gravity[2]):.3f}|upright={upright_score:.3f}|"
                    f"loc=({location[0]:.2f},{location[1]:.2f},{location[2]:.2f})|"
                    f"rot=({rotation[0]:.2f},{rotation[1]:.2f},{rotation[2]:.2f})"
                )
                if args.stop_on_fall:
                    break
            if args.debug_state:
                print(
                    "STATE|"
                    f"{step}|"
                    f"obs_last_action={format_named_vector(JOINT_NAMES, policy_obs[33:45])}|"
                    f"qpos={format_named_vector(JOINT_NAMES, joint_qpos)}|"
                    f"qvel={format_named_vector(JOINT_NAMES, joint_qvel)}|"
                    f"raw_action={format_named_vector(JOINT_NAMES, raw_action)}|"
                    f"bridge_action={format_named_vector(JOINT_NAMES, action)}|"
                    f"control_target={format_named_vector(JOINT_NAMES, obs_payload.get('control_targets', []))}|"
                    f"policy_joint_delta={format_named_vector(JOINT_NAMES, policy_obs[9:21])}"
                )
            if args.debug_feet and len(foot_contacts) == 4:
                foot_positions = obs_payload.get("foot_positions", [])
                foot_z = [foot_positions[index * 3 + 2] for index in range(4)] if len(foot_positions) == 12 else []
                print(
                    "FEET|"
                    f"{step}|names=FR,FL,RR,RL|"
                    f"contact={''.join('1' if value else '0' for value in foot_contacts)}|"
                    f"normal_n={format_vector(foot_normal_forces)}|"
                    f"slip_mps={format_vector(foot_slip_speeds)}|"
                    f"z_m={format_vector(foot_z)}|"
                    f"calf_error_cm={format_vector(calf_body_errors)}"
                )

        step += 1
        if synchronous:
            remaining = interval - (time.perf_counter() - iteration_start)
            if remaining > 0.0:
                time.sleep(remaining)
        else:
            next_tick += interval

    if any(foot_contact_samples):
        mean_slip = [
            foot_slip_sums[index] / foot_contact_samples[index]
            if foot_contact_samples[index]
            else 0.0
            for index in range(4)
        ]
        loaded_mean_slip = [
            loaded_slip_sums[index] / loaded_contact_samples[index]
            if loaded_contact_samples[index]
            else 0.0
            for index in range(4)
        ]
        print(
            "FOOT_SUMMARY|names=FR,FL,RR,RL|"
            f"contact_samples={foot_contact_samples}|"
            f"mean_slip_mps={format_vector(mean_slip)}|"
            f"max_slip_mps={format_vector(foot_slip_maxima)}|"
            f"loaded_force_threshold_n={args.loaded_contact_force:.1f}|"
            f"loaded_samples={loaded_contact_samples}|"
            f"loaded_mean_slip_mps={format_vector(loaded_mean_slip)}|"
            f"loaded_max_slip_mps={format_vector(loaded_slip_maxima)}|"
            f"max_calf_error_cm={format_vector(calf_error_maxima)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run Go1 locomotion policy through UnrealCV MuJoCo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    add_demo_spawn_arguments(parser)
    parser.add_argument(
        "--skip-spawn",
        action="store_true",
        help="Compatibility option: require and use the actor named by --actor",
    )
    parser.add_argument(
        "--actor",
        default="",
        help="Explicit existing Go1 actor; otherwise reuse a scene Go1 or spawn from camera 0",
    )
    parser.add_argument("--policy", default="none", help="Path to an ONNX policy, or 'none' for zero-action debug")
    parser.add_argument("--obs-mean", default="", help="Optional JSON list/dict for observation mean")
    parser.add_argument("--obs-std", default="", help="Optional JSON list/dict for observation std")
    parser.add_argument("--vx", type=float, default=0.2, help="Forward velocity command in m/s")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command in m/s")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="Yaw-rate command in rad/s")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=1.0, help="Seconds to hold zero action before policy control")
    parser.add_argument("--policy-hz", type=float, default=50.0)
    parser.add_argument(
        "--control-mode",
        choices=("sync", "async"),
        default="sync",
        help="Use fixed-step request/response control or the legacy UE Tick-driven loop",
    )
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument(
        "--action-clip",
        type=float,
        default=None,
        help="Legacy alias that sets both --raw-action-clip and --bridge-action-clip",
    )
    parser.add_argument(
        "--raw-action-clip",
        type=float,
        default=1.0,
        help="Legacy async clip; sync mode only reports how many raw actions exceed this threshold",
    )
    parser.add_argument(
        "--bridge-action-clip",
        type=float,
        default=1.0,
        help="Legacy async bridge clip; sync mode only reports how many converted actions exceed this threshold",
    )
    parser.add_argument(
        "--command-clip",
        type=float,
        default=1.0,
        help="Symmetric clip for policy command values; use <=0 to disable",
    )
    parser.add_argument(
        "--command-ramp",
        type=float,
        default=1.0,
        help="Seconds used to ramp command from zero after warmup; use 0 for an immediate step command",
    )
    parser.add_argument(
        "--fall-gravity-z-threshold",
        type=float,
        default=0.3,
        help="Print FALL when projected gravity z is above this value, indicating the base has rolled over",
    )
    parser.add_argument(
        "--min-upright-gravity-z",
        type=float,
        default=0.7,
        help="Print FALL when -projected_gravity_z drops below this upright score; set <=0 to disable side-fall detection",
    )
    parser.add_argument("--stop-on-fall", action="store_true", help="Stop the control loop after FALL is detected")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--debug-state", action="store_true", help="Print joint qpos/qvel and raw/bridge action details")
    parser.add_argument(
        "--debug-feet",
        action="store_true",
        help="Print per-foot contact, slip speed, normal force, and UE/MuJoCo calf mapping error",
    )
    parser.add_argument(
        "--loaded-contact-force",
        type=float,
        default=20.0,
        help="Normal-force threshold in newtons used for loaded-foot slip summary metrics",
    )
    parser.add_argument(
        "--max-displacement-cm",
        "--max-abs-location-cm",
        dest="max_displacement_cm",
        type=float,
        default=100000.0,
        help=(
            "Maximum Euclidean displacement from the actor's control-loop start location; "
            "--max-abs-location-cm is retained as a compatibility alias"
        ),
    )
    args = parser.parse_args()
    validate_demo_spawn_arguments(parser, args)
    if not math.isfinite(args.max_displacement_cm) or args.max_displacement_cm <= 0.0:
        parser.error("--max-displacement-cm must be a positive finite number")
    if args.action_clip is not None:
        args.raw_action_clip = args.action_clip
        args.bridge_action_clip = args.action_clip

    policy = make_policy(args)
    print(f"POLICY|{policy.describe()}")

    client = connect_client(args.host, args.port)
    actor_name = None
    owns_actor = False
    try:
        level_name = request(client, "vget /level/name")
        print(f"LEVEL|{level_name}")

        if args.actor:
            actor_name = args.actor
            print(f"GO1_REUSE|actor={actor_name}|owned=no|source=explicit-actor")
        elif args.skip_spawn:
            raise RuntimeError("--skip-spawn requires --actor")
        else:
            actor_name, owns_actor = acquire_go1_for_demo(client, args)

        print(f"ACTOR|{actor_name}")
        print(f"ACTOR_LOCATION|{request(client, f'vget /object/{actor_name}/location')}")
        stop_go1_simulation(client, actor_name, required=False)
        start_result = request(client, f"vset /object/{actor_name}/mujoco_quadruped_pose_preview/start go1")
        paths = parse_start_result(start_result)
        print_artifact_paths(paths)

        run_control_loop(client, actor_name, policy, args)
        print(f"ACTOR_LOCATION_AFTER|{request(client, f'vget /object/{actor_name}/location')}")
        print(f"ACTOR_ROTATION_AFTER|{request(client, f'vget /object/{actor_name}/rotation')}")
        print("DONE|locomotion_loop_finished")
    finally:
        if actor_name:
            try:
                send_policy_command(client, actor_name, (0.0, 0.0, 0.0))
                send_policy_action(client, actor_name, [0.0] * ACTION_DIM)
            except Exception:
                pass
            try:
                stop_go1_simulation(client, actor_name, required=False)
            except Exception:
                pass
            if owns_actor and not args.keep_actor:
                try:
                    destroy_actor(client, actor_name)
                except Exception as exc:
                    print(f"CLEANUP_WARNING|actor={actor_name}|error={exc}")
        try:
            client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
