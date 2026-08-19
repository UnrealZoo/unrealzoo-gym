#!/usr/bin/env python3
"""Keyboard-control a Go1 in an already running UnrealZoo session.

The script connects directly through UnrealCV. It reuses an existing Go1 or
spawns one relative to Camera 0, without creating a Gym environment.
"""
from __future__ import annotations

import argparse
import math
import os
import select
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import go1_locomotion as locomotion  # noqa: E402


def ijkl_command(i_pressed, k_pressed, j_pressed, l_pressed, vx, yaw_rate):
    """Build the shared I/K forward/back and J/L turn command."""
    return (
        vx * (int(bool(i_pressed)) - int(bool(k_pressed))),
        0.0,
        yaw_rate * (int(bool(j_pressed)) - int(bool(l_pressed))),
    )


class KeyboardController:
    def __init__(self):
        self._fd = None
        self._old_terminal_settings = None

    def __enter__(self):
        if os.name != "nt":
            if not sys.stdin.isatty():
                raise RuntimeError("Keyboard control requires an interactive terminal")
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old_terminal_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._old_terminal_settings is not None and self._fd is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_terminal_settings)

    def read(self, current_command, vx, yaw_rate):
        if os.name == "nt":
            import ctypes

            get_key_state = ctypes.windll.user32.GetAsyncKeyState

            def pressed(virtual_key):
                return bool(get_key_state(virtual_key) & 0x8000)

            exit_requested = pressed(ord("X")) or pressed(0x1B)
            hard_stop = exit_requested or pressed(0x20)
            if hard_stop:
                return (0.0, 0.0, 0.0), exit_requested, True
            return ijkl_command(
                pressed(ord("I")),
                pressed(ord("K")),
                pressed(ord("J")),
                pressed(ord("L")),
                vx,
                yaw_rate,
            ), False, False

        command = current_command
        exit_requested = False
        hard_stop = False
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1).lower()
            commands = {
                "i": (vx, 0.0, 0.0),
                "k": (-vx, 0.0, 0.0),
                "j": (0.0, 0.0, yaw_rate),
                "l": (0.0, 0.0, -yaw_rate),
                " ": (0.0, 0.0, 0.0),
            }
            if key in ("x", "\x1b"):
                command = (0.0, 0.0, 0.0)
                exit_requested = True
                hard_stop = True
            elif key in commands:
                command = commands[key]
                hard_stop = key == " "
        return command, exit_requested, hard_stop


def interpolate_command(start, target, elapsed, ramp_seconds):
    if ramp_seconds <= 0.0:
        return target
    alpha = min(1.0, max(0.0, elapsed / ramp_seconds))
    return tuple(start[index] + (target[index] - start[index]) * alpha for index in range(3))


def run_keyboard_loop(client, actor_name, policy, args):
    interval = 1.0 / args.policy_hz
    synchronous = args.control_mode == "sync"
    if synchronous and not math.isclose(args.policy_hz, 50.0, abs_tol=1e-6):
        raise RuntimeError("Synchronous Go1 policy control requires --policy-hz 50")

    if synchronous:
        obs, obs_payload = locomotion.start_synchronous_policy(client, actor_name)
        terrain_mode = (
            "voxel-heightfield"
            if obs_payload.get("local_ground_heightfield_enabled", False)
            else "root-probed-slab"
            if obs_payload.get("local_ground_patch_enabled", False)
            else "static-export"
        )
        print(f"TERRAIN|mode={terrain_mode}")
        if terrain_mode == "root-probed-slab":
            print(
                "TERRAIN_WARNING|root-probed slab cannot represent per-foot "
                "height differences at slopes, curbs, stairs, or road seams"
            )
    else:
        obs = None
        obs_payload = None

    print("KEYBOARD|I/K=forward/back|J/L=turn|Space=stop|X/Esc=exit")
    print(
        f"CONTROL|mode={args.control_mode}|policy_hz={args.policy_hz:.1f}|"
        f"vx={args.vx:.3f}|yaw_rate={args.yaw_rate:.3f}"
    )

    start_time = time.perf_counter()
    next_tick = start_time
    target_command = (0.0, 0.0, 0.0)
    active_command = (0.0, 0.0, 0.0)
    ramp_start_command = active_command
    ramp_start_time = 0.0
    last_sent_command = None
    last_raw_action = [0.0] * locomotion.ACTION_DIM
    step = 0
    fall_active = False

    with KeyboardController() as keyboard:
        while True:
            iteration_start = time.perf_counter()
            elapsed = step * interval if synchronous else iteration_start - start_time
            if args.duration > 0.0 and elapsed >= args.duration:
                break
            if not synchronous and iteration_start < next_tick:
                time.sleep(next_tick - iteration_start)

            new_target, exit_requested, hard_stop = keyboard.read(
                target_command, args.vx, args.yaw_rate
            )
            new_target, _ = locomotion.clamp_command(new_target, args.command_clip)
            if exit_requested:
                active_command = (0.0, 0.0, 0.0)
                locomotion.send_policy_command(client, actor_name, active_command)
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

            if last_sent_command is None or any(
                abs(active_command[index] - last_sent_command[index]) > 1e-6
                for index in range(3)
            ):
                locomotion.send_policy_command(client, actor_name, active_command)
                last_sent_command = active_command

            if not synchronous:
                obs, obs_payload = locomotion.get_policy_observation(client, actor_name)
            policy_obs = locomotion.prepare_policy_obs(policy, obs, last_raw_action)
            if elapsed < args.warmup:
                network_action = [0.0] * locomotion.ACTION_DIM
            else:
                network_action = locomotion.scale_action(policy.infer(policy_obs), args.action_gain)

            if synchronous:
                raw_action = locomotion.clamp_action(
                    network_action, args.raw_action_clip, 1.0
                )
                bridge_action = locomotion.to_bridge_action(policy, raw_action)
                obs, obs_payload = locomotion.step_synchronous_policy(
                    client, actor_name, raw_action
                )
            else:
                raw_action = locomotion.clamp_action(
                    network_action, args.raw_action_clip, 1.0
                )
                bridge_action = locomotion.to_bridge_action(policy, raw_action)
                bridge_action = locomotion.clamp_action(
                    bridge_action, args.bridge_action_clip, 1.0
                )
                locomotion.send_policy_action(client, actor_name, bridge_action)
            last_raw_action = list(raw_action)

            gravity = obs_payload.get("gravity", [0.0, 0.0, -1.0])
            is_fallen = len(gravity) == 3 and float(gravity[2]) > args.fall_gravity_z
            if is_fallen and not fall_active:
                fall_active = True
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

            if step % args.print_every == 0:
                linvel = obs_payload.get("linvel", [0.0, 0.0, 0.0])
                print(
                    f"STEP|{step}|t={elapsed:.3f}|"
                    f"cmd=({active_command[0]:.3f},{active_command[1]:.3f},{active_command[2]:.3f})|"
                    f"linvel=({linvel[0]:.3f},{linvel[1]:.3f},{linvel[2]:.3f})|"
                    f"gravity=({gravity[0]:.3f},{gravity[1]:.3f},{gravity[2]:.3f})|"
                    f"raw_rms={locomotion.vector_rms(raw_action):.3f}|"
                    f"bridge_rms={locomotion.vector_rms(bridge_action):.3f}"
                )
                if args.debug_state:
                    root_cm = obs_payload.get("root_position_ue_cm", [0.0, 0.0, 0.0])
                    foot_positions = obs_payload.get("foot_positions", [])
                    foot_z_cm = [
                        float(foot_positions[index]) * 100.0
                        for index in range(2, len(foot_positions), 3)
                    ]
                    ground_z_cm = obs_payload.get("ground_probe_z_ue_cm")
                    root_ground_clearance_cm = (
                        float(root_cm[2]) - float(ground_z_cm)
                        if len(root_cm) >= 3 and ground_z_cm is not None
                        else None
                    )
                    foot_center_clearance_cm = (
                        [value - float(ground_z_cm) for value in foot_z_cm]
                        if ground_z_cm is not None
                        else []
                    )
                    foot_ground_probe_hits = obs_payload.get("foot_ground_probe_hits", [])
                    foot_ground_z_cm = obs_payload.get("foot_ground_z_ue_cm", [])
                    foot_surface_clearance_cm = obs_payload.get(
                        "foot_surface_clearance_ue_cm", []
                    )
                    terrain_mode = (
                        "voxel-heightfield"
                        if obs_payload.get("local_ground_heightfield_enabled", False)
                        else "root-probed-slab"
                        if obs_payload.get("local_ground_patch_enabled", False)
                        else "static-export"
                    )
                    print(
                        f"STATE|{step}|"
                        f"terrain={terrain_mode}|"
                        f"root_cm={locomotion.format_vector(root_cm)}|"
                        f"ground_probe={obs_payload.get('local_ground_patch_active', False)}|"
                        f"ground_z_cm={ground_z_cm}|"
                        f"ground_support_z_cm={obs_payload.get('ground_support_z_ue_cm')}|"
                        f"root_ground_clearance_cm={root_ground_clearance_cm}|"
                        f"feet={obs_payload.get('foot_contacts', [])}|"
                        f"foot_z_cm={locomotion.format_vector(foot_z_cm)}|"
                        f"foot_center_vs_root_ground_cm={locomotion.format_vector(foot_center_clearance_cm)}|"
                        f"foot_ground_probe_hits={foot_ground_probe_hits}|"
                        f"foot_ground_z_cm={locomotion.format_vector(foot_ground_z_cm)}|"
                        f"foot_surface_clearance_cm={locomotion.format_vector(foot_surface_clearance_cm)}|"
                        f"foot_hfield_vertices={obs_payload.get('foot_heightfield_corrected_vertex_count', 0)}|"
                        f"foot_hfield_max_correction_cm={obs_payload.get('foot_heightfield_max_correction_cm', 0.0)}|"
                        f"foot_penetration_cm={obs_payload.get('foot_contact_penetration_cm', [])}|"
                        f"normal_forces={obs_payload.get('foot_normal_forces', [])}|"
                        f"calf_body_errors_cm={obs_payload.get('calf_body_errors_cm', [])}|"
                        f"raw_action={locomotion.format_named_vector(locomotion.JOINT_NAMES, raw_action)}|"
                        f"bridge_action={locomotion.format_named_vector(locomotion.JOINT_NAMES, bridge_action)}"
                    )
                    print(
                        f"GROUND_COMPONENT|{step}|"
                        f"{obs_payload.get('ground_probe_component', 'unknown')}"
                    )
                    print(
                        f"FOOT_GROUND_COMPONENTS|{step}|"
                        f"{obs_payload.get('foot_ground_probe_components', [])}"
                    )

            step += 1
            if synchronous:
                remaining = interval - (time.perf_counter() - iteration_start)
                if remaining > 0.0:
                    time.sleep(remaining)
            else:
                next_tick += interval

    print(f"DONE|steps={step}")


def parse_args() -> argparse.Namespace:
    default_policy = SCRIPT_DIR / "policies" / "go1_velocity" / "policy.onnx"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--actor", default="", help="Explicit existing Go1 actor name")
    locomotion.add_demo_spawn_arguments(parser)
    parser.add_argument("--policy", default=str(default_policy))
    parser.add_argument("--obs-mean", default="")
    parser.add_argument("--obs-std", default="")
    parser.add_argument("--vx", type=float, default=0.5, help="I/K speed in m/s")
    parser.add_argument("--yaw-rate", type=float, default=0.8, help="J/L yaw rate in rad/s")
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until X/Esc/Ctrl+C")
    parser.add_argument("--warmup", type=float, default=0.8)
    parser.add_argument("--command-ramp", type=float, default=0.0)
    parser.add_argument("--command-clip", type=float, default=1.0)
    parser.add_argument("--policy-hz", type=float, default=50.0)
    parser.add_argument("--control-mode", choices=("sync", "async"), default="sync")
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--raw-action-clip", type=float, default=1.0)
    parser.add_argument("--bridge-action-clip", type=float, default=1.0)
    parser.add_argument(
        "--fall-gravity-z",
        type=float,
        default=-0.5,
        help="Projected-gravity z threshold used only to log fall/recovery events",
    )
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--debug-state", action="store_true")
    args = parser.parse_args()
    if args.duration < 0.0:
        parser.error("--duration must be >= 0")
    if args.policy_hz <= 0.0:
        parser.error("--policy-hz must be > 0")
    if args.print_every <= 0:
        parser.error("--print-every must be > 0")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in the range 1..65535")
    locomotion.validate_demo_spawn_arguments(parser, args)
    return args


def run_online(args: argparse.Namespace, policy, client) -> int:
    actor_name = None
    owns_actor = False
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
        start_result = locomotion.request(
            client,
            f"vset /object/{actor_name}/mujoco_quadruped_pose_preview/start go1",
        )
        locomotion.print_artifact_paths(locomotion.parse_start_result(start_result))
        run_keyboard_loop(client, actor_name, policy, args)
    finally:
        if actor_name:
            try:
                locomotion.send_policy_command(client, actor_name, (0.0, 0.0, 0.0))
                locomotion.send_policy_action(
                    client, actor_name, [0.0] * locomotion.ACTION_DIM
                )
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
    policy = locomotion.make_policy(args)
    print(f"POLICY|{policy.describe()}")
    client = locomotion.connect_client(args.host, args.port)
    try:
        return run_online(args, policy, client)
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
