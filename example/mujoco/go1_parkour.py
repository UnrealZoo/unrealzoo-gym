#!/usr/bin/env python3
"""Run Robot Parkour Learning in an already running UnrealZoo session.

The demo combines MuJoCo proprioception with depth from the Go1 FusionCamSensor
and shows the observations sent to the visual recurrent policy. An existing
Go1 is controlled directly through UnrealCV without creating a Gym environment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
EXAMPLE_DIR = SCRIPT_DIR.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import go1_locomotion as locomotion  # noqa: E402
from go1_keyboard_control import (  # noqa: E402
    KeyboardController as HoldKeyboardController,
    ijkl_command,
    interpolate_command,
)
from go1_observation_visualizer import Go1ObservationVisualizer  # noqa: E402
from parkour_policy import ACTION_DIM, RobotParkourPolicy  # noqa: E402
from unrealcv.util import read_npy  # noqa: E402


def parse_vector3(response, operation):
    try:
        values = [float(value) for value in str(response).strip().split()]
    except ValueError as exc:
        raise RuntimeError(f"Failed to {operation}: {response!r}") from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Failed to {operation}: expected three finite values, got {response!r}")
    return values


def rotation_matrix(pitch_degrees, yaw_degrees, roll_degrees):
    """Return Unreal's FRotationMatrix basis (X forward, Y right, Z up)."""
    pitch, yaw, roll = map(
        math.radians, (pitch_degrees, yaw_degrees, roll_degrees)
    )
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    sr, cr = math.sin(roll), math.cos(roll)
    return (
        (cp * cy, sr * sp * cy - cr * sy, -(cr * sp * cy + sr * sy)),
        (cp * sy, sr * sp * sy + cr * cy, cy * sr - cr * sp * sy),
        (sp, -sr * cp, cr * cp),
    )


def matrix_multiply(left, right):
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def matrix_vector(matrix, vector):
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))


def matrix_to_rotator(matrix):
    pitch = math.asin(max(-1.0, min(1.0, matrix[2][0])))
    yaw = math.atan2(matrix[1][0], matrix[0][0])
    roll = math.atan2(-matrix[2][1], matrix[2][2])
    return tuple(map(math.degrees, (pitch, yaw, roll)))


class UnrealDepthCamera:
    """Own or temporarily configure a UE scene-capture camera that follows Go1."""

    def __init__(self, client, actor_name, policy, args):
        self.client = client
        self.actor_name = actor_name
        self.policy = policy
        self.args = args
        self.camera_id = None
        self.spawned_actor = None
        self.attached_to_go1 = False
        self.original = {}

    def _camera_mappings(self):
        response = locomotion.request(self.client, "vget /cameras/ids", verbose=False)
        try:
            mappings = json.loads(str(response))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid camera mapping response: {response!r}") from exc
        if not isinstance(mappings, list):
            raise RuntimeError(f"Expected a camera mapping list, got: {response!r}")
        return mappings

    @staticmethod
    def _mapping_id(mapping):
        for key in ("camera_id", "id", "cid", "CID"):
            if key in mapping:
                return str(mapping[key])
        return ""

    @staticmethod
    def _mapping_actor(mapping):
        for key in ("actor_name", "name", "actor", "object_name"):
            if key in mapping:
                return str(mapping[key])
        return ""

    def _owned_by_go1(self, mapping):
        if self._mapping_actor(mapping) == self.actor_name:
            return True
        return self._mapping_id(mapping).startswith(f"CID-{self.actor_name}-")

    def open(self):
        requested = str(self.args.camera_id).strip()
        if requested.casefold() == "auto":
            before = self._camera_mappings()
            attached = [item for item in before if self._owned_by_go1(item)]
            if attached:
                self.camera_id = self._mapping_id(attached[0])
                self.attached_to_go1 = True
                print(
                    f"CAMERA_BIND|source=go1-fusion-component|id={self.camera_id}|"
                    f"sensor={attached[0].get('sensor_name', 'unknown')}"
                )
            else:
                before_ids = {self._mapping_id(item) for item in before}
                response = locomotion.request(
                    self.client, "vset /cameras/spawn", verbose=False
                )
                if str(response).strip().casefold().startswith("error"):
                    raise RuntimeError(f"Failed to spawn depth camera: {response}")
                self.spawned_actor = str(response).strip()
                time.sleep(0.15)
                after = self._camera_mappings()
                candidates = [
                    item for item in after if self._mapping_id(item) not in before_ids
                ]
                if not candidates and self.spawned_actor:
                    candidates = [
                        item
                        for item in after
                        if self._mapping_actor(item) == self.spawned_actor
                    ]
                if not candidates:
                    raise RuntimeError(
                        f"Camera was spawned as {self.spawned_actor!r}, but no new camera ID was registered"
                    )
                self.camera_id = self._mapping_id(candidates[-1])
                print(
                    f"CAMERA_BIND|source=temporary-fusion-actor|id={self.camera_id}"
                )
        else:
            self.camera_id = requested
            mappings = self._camera_mappings()
            selected = next(
                (item for item in mappings if self._mapping_id(item) == self.camera_id),
                None,
            )
            self.attached_to_go1 = bool(selected and self._owned_by_go1(selected))
            self.original = {
                name: locomotion.request(
                    self.client, f"vget /camera/{self.camera_id}/{name}", verbose=False
                )
                for name in ("location", "rotation", "fov", "size")
            }

        if not self.camera_id:
            raise RuntimeError("The selected camera has no usable camera ID")
        if self.spawned_actor:
            self._set("projection_type", "perspective")
        self._set("size", f"{self.policy.capture_width} {self.policy.capture_height}")
        fov = self.policy.horizontal_fov_degrees if self.args.camera_fov is None else self.args.camera_fov
        self._set("fov", f"{fov:.6f}")
        if self.attached_to_go1:
            if not self.args.preserve_attached_camera_pose:
                # Configure the component once in world space. Because it is
                # attached to Go1, Unreal keeps the resulting relative pose
                # and follows subsequent MuJoCo actor updates automatically.
                self.update_pose(force=True)
        else:
            self.update_pose()
        print(
            f"CAMERA_POSE|id={self.camera_id}|"
            f"location_cm={str(locomotion.request(self.client, f'vget /camera/{self.camera_id}/location', verbose=False)).strip()}|"
            f"rotation_deg={str(locomotion.request(self.client, f'vget /camera/{self.camera_id}/rotation', verbose=False)).strip()}|"
            f"configured={'no' if self.attached_to_go1 and self.args.preserve_attached_camera_pose else 'yes'}"
        )
        return self.camera_id

    def _set(self, property_name, value):
        response = locomotion.request(
            self.client,
            f"vset /camera/{self.camera_id}/{property_name} {value}",
            verbose=False,
        )
        if str(response).strip().casefold().startswith("error"):
            raise RuntimeError(
                f"Failed to set camera {self.camera_id} {property_name}: {response}"
            )

    def update_pose(self, force=False):
        if self.attached_to_go1 and not force:
            return
        location = parse_vector3(
            locomotion.request(
                self.client, f"vget /object/{self.actor_name}/location", verbose=False
            ),
            "read Go1 location",
        )
        rotation = parse_vector3(
            locomotion.request(
                self.client, f"vget /object/{self.actor_name}/rotation", verbose=False
            ),
            "read Go1 rotation",
        )
        mount = (
            self.policy.camera_position_m[0] if self.args.camera_mount_x is None else self.args.camera_mount_x,
            self.policy.camera_position_m[1] if self.args.camera_mount_y is None else self.args.camera_mount_y,
            self.policy.camera_position_m[2] if self.args.camera_mount_z is None else self.args.camera_mount_z,
        )
        # The upstream robot frame is X-forward/Y-left/Z-up; Unreal uses Y-right.
        local_offset_cm = (mount[0] * 100.0, -mount[1] * 100.0, mount[2] * 100.0)
        actor_matrix = rotation_matrix(*rotation)
        world_offset = matrix_vector(actor_matrix, local_offset_cm)
        camera_location = tuple(location[index] + world_offset[index] for index in range(3))
        camera_pitch = (
            self.policy.camera_pitch_degrees
            if self.args.camera_pitch is None
            else self.args.camera_pitch
        )
        camera_matrix = matrix_multiply(actor_matrix, rotation_matrix(camera_pitch, 0.0, 0.0))
        camera_rotation = matrix_to_rotator(camera_matrix)
        self._set("location", " ".join(f"{value:.6f}" for value in camera_location))
        self._set("rotation", " ".join(f"{value:.6f}" for value in camera_rotation))

    def capture_depth_m(self):
        response = locomotion.request(
            self.client, f"vget /camera/{self.camera_id}/depth npy", verbose=False
        )
        if not isinstance(response, (bytes, bytearray, memoryview)):
            raise RuntimeError(f"Depth camera returned a non-binary response: {response!r}")
        depth = read_npy(response)
        if depth is None or depth.ndim not in (2, 3):
            raise RuntimeError("Failed to decode the UnrealCV depth NPY response")
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise RuntimeError(f"Expected HxW depth, got {depth.shape}")
        return depth.astype("float32", copy=False) * float(self.args.depth_scale)

    def close(self):
        if self.attached_to_go1:
            return
        if self.spawned_actor:
            try:
                locomotion.request(
                    self.client,
                    f"vset /object/{self.spawned_actor}/destroy",
                    verbose=False,
                )
            except Exception:
                pass
            return
        if self.camera_id and self.original:
            for property_name in ("size", "fov", "rotation", "location"):
                try:
                    self._set(property_name, str(self.original[property_name]).strip())
                except Exception:
                    pass


class DepthDelayBuffer:
    def __init__(self, latency_seconds):
        self.latency_seconds = max(0.0, float(latency_seconds))
        self.frames = deque()

    def reset(self):
        self.frames.clear()

    def push(self, timestamp, frame):
        self.frames.append((float(timestamp), frame))
        while len(self.frames) > 32:
            self.frames.popleft()

    def delayed(self, timestamp):
        if not self.frames:
            raise RuntimeError("No depth frame is available")
        target = float(timestamp) - self.latency_seconds
        selected = self.frames[0]
        for item in self.frames:
            if item[0] <= target:
                selected = item
            else:
                break
        return selected


class ParkourKeyboardController:
    """Windows hold input with the legacy latch mode still available."""

    WINDOWS_KEYS = (
        ("X", ord("X")),
        ("Esc", 0x1B),
        ("Space", 0x20),
        ("R", ord("R")),
        ("I", ord("I")),
        ("K", ord("K")),
        ("J", ord("J")),
        ("L", ord("L")),
    )

    def __init__(self, mode):
        self.mode = mode
        self.hold_controller = HoldKeyboardController()
        self.previous_key_state = {name: False for name, _ in self.WINDOWS_KEYS}

    def __enter__(self):
        self.hold_controller.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.hold_controller.__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def _print_event(key, command):
        print(
            f"KEY_EVENT|key={key}|target=({command[0]:.3f},"
            f"{command[1]:.3f},{command[2]:.3f})"
        )

    def read(self, current_command, vx, yaw_rate):
        if self.mode == "hold" and os.name == "nt":
            import ctypes

            get_key_state = ctypes.windll.user32.GetAsyncKeyState

            def pressed(virtual_key):
                return bool(get_key_state(virtual_key) & 0x8000)

            exit_requested = pressed(ord("X")) or pressed(0x1B)
            hard_stop = exit_requested or pressed(0x20)
            command = ijkl_command(
                pressed(ord("I")),
                pressed(ord("K")),
                pressed(ord("J")),
                pressed(ord("L")),
                vx,
                yaw_rate,
            )
            if hard_stop:
                command = (0.0, 0.0, 0.0)
            reset_requested = bool(get_key_state(ord("R")) & 0x0001)
            if command != current_command or exit_requested or hard_stop:
                self._print_event("hold", command)
            return command, exit_requested, hard_stop, reset_requested

        if os.name != "nt":
            command, exit_requested, hard_stop = self.hold_controller.read(
                current_command, vx, yaw_rate
            )
            reset_requested = False
            if command != current_command or exit_requested or hard_stop:
                self._print_event("hold", command)
            return command, exit_requested, hard_stop, reset_requested

        import ctypes

        get_key_state = ctypes.windll.user32.GetAsyncKeyState
        pressed_edges = []
        for name, virtual_key in self.WINDOWS_KEYS:
            is_down = bool(get_key_state(virtual_key) & 0x8000)
            if is_down and not self.previous_key_state[name]:
                pressed_edges.append(name)
            self.previous_key_state[name] = is_down

        # Some IDE and redirected PowerShell terminals deliver key presses to
        # the console without exposing a reliable asynchronous key-down state.
        # Poll the console as well so latch mode works in both terminal types.
        import msvcrt

        console_key_names = {
            "i": "I",
            "k": "K",
            "j": "J",
            "l": "L",
            "r": "R",
            "x": "X",
            " ": "Space",
            "\x1b": "Esc",
        }
        while msvcrt.kbhit():
            character = msvcrt.getwch()
            if character in ("\x00", "\xe0"):
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            key_name = console_key_names.get(character.lower())
            if key_name is not None and key_name not in pressed_edges:
                pressed_edges.append(key_name)

        command = current_command
        exit_requested = False
        hard_stop = False
        reset_requested = False
        commands = {
            "I": (vx, 0.0, 0.0),
            "K": (-vx, 0.0, 0.0),
            "J": (0.0, 0.0, yaw_rate),
            "L": (0.0, 0.0, -yaw_rate),
        }
        for key in pressed_edges:
            if key in ("X", "Esc"):
                command = (0.0, 0.0, 0.0)
                exit_requested = True
                hard_stop = True
            elif key == "Space":
                command = (0.0, 0.0, 0.0)
                hard_stop = True
            elif key == "R":
                reset_requested = True
            elif key in commands:
                command = commands[key]
            if command != current_command or exit_requested or hard_stop or reset_requested:
                self._print_event(key, command)
        return command, exit_requested, hard_stop, reset_requested


def reset_recurrent(policy, depth_buffer, reason, step):
    policy.reset()
    depth_buffer.reset()
    print(f"RESET|step={step}|reason={reason}")
    return [0.0] * ACTION_DIM


def self_test(policy):
    np = policy.np
    raw_depth = np.full(
        (policy.capture_height, policy.capture_width), policy.depth_max, dtype=np.float32
    )
    processed = policy.preprocess_depth(raw_depth)
    bridge_obs = [0.0] * 48
    bridge_obs[8] = -1.0
    proprio = policy.build_proprioception(bridge_obs, (0.0, 0.0, 0.0), [0.0] * ACTION_DIM)
    action = policy.infer_preprocessed(proprio, processed)
    bridge_action, clipped = policy.to_bridge_sync_action(action)
    policy.reset()
    print(
        f"SELF_TEST|depth={tuple(processed.shape)}|action_dim={len(action)}|"
        f"policy_rms={locomotion.vector_rms(clipped):.6f}|"
        f"bridge_rms={locomotion.vector_rms(bridge_action):.6f}|PASS"
    )


def run_control_loop(client, actor_name, policy, camera, args):
    interval = 1.0 / args.policy_hz
    depth_interval = policy.depth_refresh_seconds if args.depth_hz <= 0.0 else 1.0 / args.depth_hz
    depth_latency = policy.depth_latency_seconds if args.depth_latency is None else args.depth_latency
    depth_buffer = DepthDelayBuffer(depth_latency)
    actor_location_before_start = parse_vector3(
        locomotion.request(
            client, f"vget /object/{actor_name}/location", verbose=False
        ),
        "read Go1 location before synchronous start",
    )
    start_response = locomotion.request(
        client,
        f"vset /object/{actor_name}/mujoco_go1_policy_sync/start parkour",
        verbose=False,
    )
    try:
        obs, payload = locomotion.parse_policy_observation(
            start_response, "start synchronous Go1 parkour control"
        )
    except RuntimeError as profile_error:
        # A completed demo intentionally leaves MuJoCo attached to the current
        # Go1 pose. Reattach to that same parkour instance on a subsequent run;
        # never accept a velocity-profile instance through this fallback.
        if "Cannot change the Go1 policy profile while MuJoCo is running" not in str(
            start_response
        ):
            raise
        resume_response = locomotion.request(
            client,
            f"vset /object/{actor_name}/mujoco_go1_policy_sync/start",
            verbose=False,
        )
        obs, payload = locomotion.parse_policy_observation(
            resume_response, "resume synchronous Go1 parkour control"
        )
        if payload.get("policy_profile") != "parkour":
            raise profile_error
    actor_location_after_start = parse_vector3(
        locomotion.request(
            client, f"vget /object/{actor_name}/location", verbose=False
        ),
        "read Go1 location after synchronous start",
    )
    startup_shift_cm = math.sqrt(
        sum(
            (actor_location_after_start[index] - actor_location_before_start[index]) ** 2
            for index in range(3)
        )
    )
    print(
        "START_BIND|before_cm="
        + ",".join(f"{value:.3f}" for value in actor_location_before_start)
        + "|after_cm="
        + ",".join(f"{value:.3f}" for value in actor_location_after_start)
        + f"|shift_cm={startup_shift_cm:.6f}"
    )
    if startup_shift_cm > 0.1:
        raise RuntimeError(
            "Starting MuJoCo moved the spawned Go1 before any policy step; "
            f"shift_cm={startup_shift_cm:.6f}"
        )
    print(
        f"ENVIRONMENT|geom_count={payload.get('environment_geom_count', 'unknown')}|"
        f"profile={payload.get('policy_profile', 'unknown')}|"
        f"local_ground_patch={payload.get('local_ground_patch_enabled', 'unknown')}|"
        f"local_ground_heightfield={payload.get('local_ground_heightfield_enabled', 'unknown')}|"
        f"ground_probe_active={payload.get('local_ground_patch_active', 'unknown')}|"
        f"ground_probe_component={payload.get('ground_probe_component', 'unknown')}|"
        f"collision_report={payload.get('environment_collision_report_path', 'unknown')}"
    )
    print(
        "BRIDGE_CONFIG|observation_defaults="
        + ",".join(
            f"{value:.6f}" for value in policy.bridge_observation_default_joint_pos
        )
        + "|action_defaults="
        + ",".join(f"{value:.6f}" for value in policy.bridge_action_default_joint_pos)
        + "|scales="
        + ",".join(f"{value:.6f}" for value in policy.bridge_action_scale)
        + "|startup_steps=0"
    )
    initial_level = str(locomotion.request(client, "vget /level/name", verbose=False)).strip()

    print(
        "KEYBOARD|I/K=forward/back|J/L=turn|Space=stop|"
        f"R=reset-GRU|X/Esc=exit|mode={args.keyboard_mode}"
    )
    print(
        f"CONTROL|policy_hz={args.policy_hz:.1f}|depth_hz={1.0 / depth_interval:.1f}|"
        f"depth_latency={depth_latency:.3f}|vx={args.vx:.3f}|vy={args.vy:.3f}|"
        f"yaw_rate={args.yaw_rate:.3f}"
    )

    start_time = time.perf_counter()
    next_depth_time = 0.0
    target_command = (0.0, 0.0, 0.0)
    active_command = (0.0, 0.0, 0.0)
    ramp_start_command = active_command
    ramp_start_time = 0.0
    last_sent_command = None
    last_policy_action = [0.0] * ACTION_DIM
    bridge_home_action = policy.bridge_home_action()
    previous_sim_time = payload.get("sim_time")
    command_was_active = False
    policy_running = args.command_mode != "keyboard"
    stop_blend_start_time = None
    stop_blend_start_action = list(bridge_home_action)
    last_bridge_action = list(bridge_home_action)
    depth_frame = 0
    step = 0
    fall_active = False

    visualizer = Go1ObservationVisualizer(
        policy.np,
        enabled=args.visualize_observation,
        dump_dir=args.observation_dump_dir,
        dump_every=args.observation_dump_every,
        depth_max=policy.depth_max,
        expected_raw_shape=(policy.capture_height, policy.capture_width),
        expected_processed_shape=(policy.input_height, policy.input_width),
    )
    with ParkourKeyboardController(args.keyboard_mode) as keyboard, visualizer:
        while True:
            iteration_start = time.perf_counter()
            elapsed = step * interval
            if args.duration > 0.0 and elapsed >= args.duration:
                break

            if elapsed + 1e-9 >= next_depth_time or not depth_buffer.frames:
                camera.update_pose()
                depth_m = camera.capture_depth_m()
                processed_depth = policy.preprocess_depth(depth_m, args.crop_far)
                depth_frame += 1
                depth_buffer.push(elapsed, (depth_frame, depth_m, processed_depth))
                next_depth_time = elapsed + depth_interval

            if args.command_mode == "keyboard":
                new_target, exit_requested, hard_stop, manual_reset = keyboard.read(
                    target_command, args.vx, args.yaw_rate
                )
            else:
                new_target = (args.vx, args.vy, args.yaw_rate)
                exit_requested = False
                hard_stop = False
                manual_reset = False
            new_target, _ = locomotion.clamp_command(new_target, args.command_clip)
            if exit_requested:
                locomotion.send_policy_command(client, actor_name, (0.0, 0.0, 0.0))
                break
            if new_target != target_command:
                ramp_start_command = active_command
                ramp_start_time = elapsed
                target_command = new_target

            if elapsed < args.warmup or hard_stop or target_command == (0.0, 0.0, 0.0):
                active_command = (0.0, 0.0, 0.0)
            else:
                active_command = interpolate_command(
                    ramp_start_command,
                    target_command,
                    elapsed - max(ramp_start_time, args.warmup),
                    args.command_ramp,
                )
            if last_sent_command != active_command:
                locomotion.send_policy_command(client, actor_name, active_command)
                last_sent_command = active_command

            if manual_reset:
                last_policy_action = reset_recurrent(
                    policy, depth_buffer, "manual-key", step
                )
                next_depth_time = elapsed

            sim_time = payload.get("sim_time")
            if (
                previous_sim_time is not None
                and sim_time is not None
                and float(sim_time) + 1e-6 < float(previous_sim_time)
            ):
                last_policy_action = reset_recurrent(
                    policy, depth_buffer, "simulation-time-rollback", step
                )
                next_depth_time = elapsed
            previous_sim_time = sim_time

            gravity = payload.get("gravity", obs[6:9])
            is_fallen = len(gravity) == 3 and float(gravity[2]) > args.reset_gravity_z
            if is_fallen and not fall_active:
                fall_active = True
                visualizer.save_last("fall")
                print(
                    f"FALL_DETECTED|step={step}|gravity_z={float(gravity[2]):.3f}|"
                    "simulation=continuing"
                )
            elif not is_fallen and fall_active:
                fall_active = False
                print(
                    f"RECOVERY_DETECTED|step={step}|gravity_z={float(gravity[2]):.3f}|"
                    "simulation=continuing"
                )

            if args.level_check_every > 0 and step > 0 and step % args.level_check_every == 0:
                level = str(locomotion.request(client, "vget /level/name", verbose=False)).strip()
                if level != initial_level:
                    policy.reset()
                    raise RuntimeError(
                        f"Level changed from {initial_level!r} to {level!r}; rerun the demo to rebind Go1 and camera"
                    )
                objects = str(locomotion.request(client, "vget /objects", verbose=False)).split()
                if actor_name not in objects:
                    policy.reset()
                    raise RuntimeError(f"Go1 actor {actor_name!r} was removed or respawned; rerun the demo")

            # A reset intentionally discards delayed visual history. Seed the
            # new history immediately so this same control tick remains valid.
            if not depth_buffer.frames:
                camera.update_pose()
                depth_m = camera.capture_depth_m()
                processed_depth = policy.preprocess_depth(depth_m, args.crop_far)
                depth_frame += 1
                depth_buffer.push(elapsed, (depth_frame, depth_m, processed_depth))
                next_depth_time = elapsed + depth_interval

            depth_timestamp, delayed_visual_observation = depth_buffer.delayed(elapsed)
            delayed_depth_frame, delayed_raw_depth, delayed_depth = delayed_visual_observation
            inference_start = time.perf_counter()
            command_active = any(abs(value) > 1e-6 for value in active_command)
            if command_active and not command_was_active:
                policy.reset()
                last_policy_action = [0.0] * ACTION_DIM
                policy_running = True
                stop_blend_start_time = None
                print(f"POLICY_MODE|step={step}|running=yes|reason=movement-key")
                print(f"COMMAND_MODE|step={step}|moving=yes")
            elif not command_active and command_was_active:
                # Robot Parkour's distilled actor does not reliably stop for a
                # zero command.  Stop invoking it on key-up, then ease the last
                # dynamic joint target back to the stable bridge home stance.
                policy_running = False
                stop_blend_start_time = elapsed
                stop_blend_start_action = list(last_bridge_action)
                policy.reset()
                last_policy_action = [0.0] * ACTION_DIM
                print(
                    f"POLICY_MODE|step={step}|running=no|"
                    f"stand_blend={args.stop_blend:.3f}s"
                )
                print(f"COMMAND_MODE|step={step}|moving=no|velocity=zero")
            command_was_active = command_active

            proprio = policy.build_proprioception(obs, active_command, last_policy_action)
            if elapsed < args.warmup or not policy_running:
                policy_action = [0.0] * ACTION_DIM
            else:
                policy_action = policy.infer_preprocessed(proprio, delayed_depth)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            bridge_action, last_policy_action = policy.to_bridge_sync_action(policy_action)
            if not policy_running and elapsed >= args.warmup:
                if stop_blend_start_time is None:
                    bridge_action = list(bridge_home_action)
                else:
                    stop_alpha = min(
                        1.0,
                        (elapsed - stop_blend_start_time) / max(args.stop_blend, 1e-6),
                    )
                    stop_alpha = stop_alpha * stop_alpha * (3.0 - 2.0 * stop_alpha)
                    bridge_action = [
                        stop_blend_start_action[index]
                        + stop_alpha
                        * (bridge_home_action[index] - stop_blend_start_action[index])
                        for index in range(ACTION_DIM)
                    ]
                    if stop_alpha >= 1.0:
                        stop_blend_start_time = None
                last_policy_action = [0.0] * ACTION_DIM
            if elapsed < args.warmup:
                # Start at the bridge's current/default stance, then blend into
                # the official parkour stance instead of applying a joint step
                # on the first simulation tick.
                warmup_alpha = min(1.0, elapsed / max(args.warmup, 1e-6))
                warmup_alpha = warmup_alpha * warmup_alpha * (3.0 - 2.0 * warmup_alpha)
                bridge_action = [
                    bridge_home_action[index]
                    + warmup_alpha * (bridge_action[index] - bridge_home_action[index])
                    for index in range(ACTION_DIM)
                ]
            last_bridge_action = list(bridge_action)
            obs, payload = locomotion.step_synchronous_policy(client, actor_name, bridge_action)
            next_sim_time = payload.get("sim_time")
            if (
                sim_time is not None
                and next_sim_time is not None
                and float(next_sim_time) <= float(sim_time) + 1e-9
            ):
                raise RuntimeError(
                    "MuJoCo simulation did not advance; check the generated ground "
                    "plane/collision report and rebuild the UnrealCV C++ plugin"
                )

            visualizer.update(
                step=step,
                sim_time=next_sim_time,
                depth_frame=delayed_depth_frame,
                raw_depth=delayed_raw_depth,
                processed_depth=delayed_depth,
                proprio=proprio,
                command=active_command,
                action=last_policy_action,
                payload=payload,
                depth_age=elapsed - depth_timestamp,
                camera_id=camera.camera_id,
                camera_source=(
                    "go1-fusion-component"
                    if camera.attached_to_go1
                    else "temporary-fusion-actor"
                ),
            )

            if step % args.print_every == 0:
                linvel = payload.get("linvel", [0.0, 0.0, 0.0])
                gravity = payload.get("gravity", obs[6:9])
                root_position = payload.get("root_position_ue_cm", [0.0, 0.0, 0.0])
                print(
                    f"STEP|{step}|t={elapsed:.3f}|cmd=({active_command[0]:.3f},"
                    f"{active_command[1]:.3f},{active_command[2]:.3f})|"
                    f"root_cm=({root_position[0]:.1f},{root_position[1]:.1f},{root_position[2]:.1f})|"
                    f"linvel=({linvel[0]:.3f},{linvel[1]:.3f},{linvel[2]:.3f})|"
                    f"gravity=({gravity[0]:.3f},{gravity[1]:.3f},{gravity[2]:.3f})|"
                    f"ground_probe={payload.get('local_ground_patch_active', 'unknown')}|"
                    f"ground_z_cm={payload.get('ground_probe_z_ue_cm', 'unknown')}|"
                    f"feet={payload.get('foot_contacts', 'unknown')}|"
                    f"foot_ground_z_cm={payload.get('foot_ground_z_ue_cm', 'unknown')}|"
                    f"foot_positions_cm={payload.get('foot_positions_ue_cm', 'unknown')}|"
                    f"foot_ground_components={payload.get('foot_ground_probe_components', 'unknown')}|"
                    f"foot_surface_clearance_cm={payload.get('foot_surface_clearance_ue_cm', 'unknown')}|"
                    f"foot_hfield_vertices={payload.get('foot_heightfield_corrected_vertex_count', 0)}|"
                    f"foot_hfield_max_correction_cm={payload.get('foot_heightfield_max_correction_cm', 0.0)}|"
                    f"foot_penetration_cm={payload.get('foot_contact_penetration_cm', 'unknown')}|"
                    f"normal_forces={payload.get('foot_normal_forces', 'unknown')}|"
                    f"depth_age={elapsed - depth_timestamp:.3f}|infer_ms={inference_ms:.2f}|"
                    f"policy_rms={locomotion.vector_rms(last_policy_action):.3f}|"
                    f"bridge_rms={locomotion.vector_rms(bridge_action):.3f}"
                )
                if root_position[0] >= 1550.0:
                    print(
                        f"GROUND_COMPONENT|{step}|"
                        f"{payload.get('ground_probe_component', 'unknown')}"
                    )
                if args.debug_state:
                    print(f"PROPRIO|{step}|{proprio}")
                    print(f"POLICY_ACTION|{step}|{last_policy_action}")
                    print(f"BRIDGE_ACTION|{step}|{bridge_action}")

            step += 1
            remaining = interval - (time.perf_counter() - iteration_start)
            if remaining > 0.0:
                time.sleep(remaining)

    print(f"DONE|steps={step}")


def parse_args() -> argparse.Namespace:
    default_policy_dir = SCRIPT_DIR / "policies" / "robot_parkour_go1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--actor", default="", help="Explicit existing Go1 actor name")
    locomotion.add_demo_spawn_arguments(parser)
    parser.add_argument("--policy-dir", default=str(default_policy_dir))
    parser.add_argument("--checkpoint", default="", help="Checkpoint path; latest model_*.pt when omitted")
    parser.add_argument("--runtime-dir", default="", help="Official rsl_rl root; bundled runtime when omitted")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0")
    parser.add_argument(
        "--camera-id",
        default="auto",
        help="Camera ID, or auto to prefer the Go1 FusionCamSensor and use a temporary fallback",
    )
    parser.add_argument("--camera-fov", type=float, default=None)
    parser.add_argument("--camera-pitch", type=float, default=None, help="Relative pitch in degrees")
    parser.add_argument("--camera-mount-x", type=float, default=None, help="Forward mount offset in metres")
    parser.add_argument("--camera-mount-y", type=float, default=None, help="Left mount offset in metres")
    parser.add_argument("--camera-mount-z", type=float, default=None, help="Up mount offset in metres")
    parser.add_argument(
        "--preserve-attached-camera-pose",
        action="store_true",
        help="Do not apply checkpoint camera extrinsics to a Go1-attached FusionCamSensor",
    )
    parser.add_argument("--depth-scale", type=float, default=0.01, help="UE depth units to metres")
    parser.add_argument("--depth-hz", type=float, default=0.0, help="0 uses checkpoint config (10 Hz)")
    parser.add_argument("--depth-latency", type=float, default=None, help="Seconds; config mean when omitted")
    parser.add_argument("--crop-far", type=float, default=None, help="Treat farther depth as max range")
    parser.add_argument(
        "--visualize-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show raw depth, policy depth, proprioception, and action live",
    )
    parser.add_argument(
        "--observation-dump-dir",
        default=str(SCRIPT_DIR / "observation_debug"),
        help="Directory for automatic anomaly/fall observation snapshots",
    )
    parser.add_argument(
        "--observation-dump-every",
        type=int,
        default=0,
        help="Save every N new depth frames; 0 saves anomalies and falls only",
    )
    parser.add_argument("--vx", type=float, default=1.0, help="I/K speed in m/s")
    parser.add_argument("--vy", type=float, default=0.4, help="Fixed-command lateral speed in m/s")
    parser.add_argument("--yaw-rate", type=float, default=0.8, help="J/L yaw rate in rad/s")
    parser.add_argument("--command-mode", choices=("keyboard", "fixed"), default="keyboard")
    parser.add_argument(
        "--keyboard-mode",
        choices=("latch", "hold"),
        default="hold",
        help="hold moves only while a key is held; latch keeps the last tapped command",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until X/Esc/Ctrl+C")
    parser.add_argument("--warmup", type=float, default=0.8)
    parser.add_argument("--command-ramp", type=float, default=0.0)
    parser.add_argument(
        "--stop-blend",
        type=float,
        default=0.6,
        help="Seconds to blend from the last parkour action back to the standing pose",
    )
    parser.add_argument("--command-clip", type=float, default=1.5)
    parser.add_argument("--policy-hz", type=float, default=50.0)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument(
        "--reset-gravity-z",
        type=float,
        default=-0.5,
        help="Projected-gravity z threshold used only to log fall/recovery events",
    )
    parser.add_argument("--level-check-every", type=int, default=50)
    parser.add_argument("--debug-state", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Load the policy and run one offline inference")
    args = parser.parse_args()
    if args.duration < 0.0 or args.warmup < 0.0 or args.stop_blend <= 0.0:
        parser.error("--duration/--warmup must be >= 0 and --stop-blend must be > 0")
    if not math.isclose(args.policy_hz, 50.0, abs_tol=1e-6):
        parser.error("Robot Parkour synchronous control requires --policy-hz 50")
    if args.depth_hz < 0.0 or (args.depth_latency is not None and args.depth_latency < 0.0):
        parser.error("--depth-hz and --depth-latency must be >= 0")
    if (
        args.depth_scale <= 0.0
        or args.print_every <= 0
        or args.observation_dump_every < 0
    ):
        parser.error(
            "--depth-scale/--print-every must be > 0 and "
            "--observation-dump-every must be >= 0"
        )
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in the range 1..65535")
    locomotion.validate_demo_spawn_arguments(parser, args)
    return args


def run_online(args: argparse.Namespace, policy: RobotParkourPolicy, client) -> int:
    actor_name = None
    owns_actor = False
    camera = None
    try:
        print(f"LEVEL|{locomotion.request(client, 'vget /level/name')}")
        if args.actor:
            actor_name = args.actor
            print(f"GO1_REUSE|actor={actor_name}|owned=no|source=explicit-actor")
        else:
            actor_name, owns_actor = locomotion.acquire_go1_for_demo(client, args)
        print(f"ACTOR|{actor_name}")
        locomotion.stop_go1_simulation(client, actor_name, required=False)
        print(
            "ACTOR_LOCATION|"
            + str(
                locomotion.request(
                    client, f"vget /object/{actor_name}/location", verbose=False
                )
            ).strip()
        )
        camera = UnrealDepthCamera(client, actor_name, policy, args)
        print(
            f"CAMERA|id={camera.open()}|"
            f"spawned_actor={camera.spawned_actor or 'no'}|"
            f"attached_to_go1={'yes' if camera.attached_to_go1 else 'no'}"
        )
        run_control_loop(client, actor_name, policy, camera, args)
    finally:
        policy.reset()
        if camera is not None:
            camera.close()
        if actor_name:
            try:
                locomotion.send_policy_command(client, actor_name, (0.0, 0.0, 0.0))
            except Exception:
                pass
            try:
                locomotion.stop_go1_simulation(client, actor_name, required=False)
            except Exception:
                pass
            if owns_actor and not args.keep_actor:
                try:
                    locomotion.destroy_actor(client, actor_name)
                except Exception as exc:
                    print(f"CLEANUP_WARNING|actor={actor_name}|error={exc}")
    return 0


def main() -> int:
    args = parse_args()
    policy = RobotParkourPolicy(
        args.policy_dir,
        checkpoint=args.checkpoint or None,
        runtime_dir=args.runtime_dir or None,
        device=args.device,
    )
    print(f"POLICY|{policy.describe()}")
    if args.self_test:
        self_test(policy)
        return 0
    client = locomotion.connect_client(args.host, args.port)
    try:
        return run_online(args, policy, client)
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
