#!/usr/bin/env python3
"""
Connect to a running UnrealCV session, spawn Unitree Go1, and start MuJoCo.

This demo intentionally stays outside gym_unrealcv.envs registration. It is a
small smoke test for the UE-side MuJoCo Go1 bridge:

1. start Unreal Editor or a packaged UnrealZoo build with UnrealCV enabled
2. enter PIE / game world
3. run this script with --mode pose_preview or --mode freefall
"""
import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unrealcv  # noqa: E402


GO1_BP_PATH = "/Game/robot-dog-unitree-go1/BP_UnitreeGo1.BP_UnitreeGo1"
GO1_ACTOR_PREFIX = "BP_UnitreeGo1"
DEFAULT_SPAWN = (0.0, 0.0, 500.0)
GO1_KEY_BONES = (
    "Trunk",
    "BodyBone",
    "HipFL",
    "ThighFL",
    "CalfFL",
    "HipFR",
    "ThighFR",
    "CalfFR",
    "HipBL",
    "ThighBL",
    "CalfBL",
    "HipBR",
    "ThighBR",
    "CalfBR",
)


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
    response = request(client, f"vset /objects/spawn_from_path {GO1_BP_PATH} {x} {y} {z}")
    actor_name = str(response).strip()
    if not actor_name or actor_name.lower().startswith("error"):
        raise RuntimeError(f"Failed to spawn Go1 actor: {response}")
    if not actor_name.startswith(GO1_ACTOR_PREFIX):
        raise RuntimeError(f"Unexpected Go1 spawn response: {response}")
    time.sleep(1.0)
    return actor_name


def wait_for_file(path, timeout_seconds, min_size_bytes=1):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists() and path.stat().st_size >= min_size_bytes:
            return True
        time.sleep(0.2)
    return path.exists()


def wait_for_status(path, timeout_seconds):
    deadline = time.time() + timeout_seconds
    last = ""
    while time.time() < deadline:
        if path.exists():
            last = path.read_text(encoding="utf-8", errors="ignore").strip()
            if last and last != "running":
                return last
        time.sleep(0.2)
    return last or "timeout"


def parse_start_result(start_result):
    parts = str(start_result).strip().split("|")
    if not parts or not parts[0]:
        raise RuntimeError(f"Unexpected MuJoCo start response: {start_result}")

    trajectory_path = Path(parts[0])
    state_log_path = Path(parts[1]) if len(parts) > 1 else None
    mjcf_path = Path(parts[2]) if len(parts) > 2 else None
    initial_snapshot_path = Path(parts[3]) if len(parts) > 3 else None
    runtime_snapshot_path = Path(parts[4]) if len(parts) > 4 else None
    environment_collision_report_path = Path(parts[5]) if len(parts) > 5 else None
    status_path = trajectory_path.with_name(trajectory_path.name.replace("_trajectory.csv", "_status.txt"))
    mapping_path = trajectory_path.with_name(trajectory_path.name.replace("_trajectory.csv", "_mapping.csv"))

    return {
        "trajectory_path": trajectory_path,
        "state_log_path": state_log_path,
        "mjcf_path": mjcf_path,
        "initial_snapshot_path": initial_snapshot_path,
        "runtime_snapshot_path": runtime_snapshot_path,
        "environment_collision_report_path": environment_collision_report_path,
        "status_path": status_path,
        "mapping_path": mapping_path,
    }


def print_artifact_paths(paths):
    for key, value in paths.items():
        if value is not None:
            print(f"{key.upper()}|{value}")


def print_pose_summary(client, actor_name):
    response = request(client, f"vget /object/{actor_name}/bones {','.join(GO1_KEY_BONES)} world")
    try:
        entries = json.loads(str(response))
    except json.JSONDecodeError:
        print("POSE_SUMMARY|unable_to_parse_bone_response")
        return

    bones = {entry.get("bone_name"): entry.get("transform", {}) for entry in entries if entry.get("bone_name")}
    body = bones.get("BodyBone", {})
    fl = bones.get("CalfFL", {})
    fr = bones.get("CalfFR", {})
    bl = bones.get("CalfBL", {})
    br = bones.get("CalfBR", {})

    if not all((body, fl, fr, bl, br)):
        print("POSE_SUMMARY|missing_required_bones")
        return

    front_mid_z = 0.5 * (fl["Z"] + fr["Z"])
    rear_mid_z = 0.5 * (bl["Z"] + br["Z"])
    print(
        "POSE_SUMMARY|"
        f"body_z={body['Z']:.3f}|"
        f"front_clearance={body['Z'] - front_mid_z:.3f}|"
        f"rear_clearance={body['Z'] - rear_mid_z:.3f}|"
        f"front_span_y={abs(fl['Y'] - fr['Y']):.3f}|"
        f"rear_span_y={abs(bl['Y'] - br['Y']):.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run the UnrealCV MuJoCo Unitree Go1 demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--mode", choices=("pose_preview", "freefall"), default="pose_preview")
    parser.add_argument("--spawn-x", type=float, default=DEFAULT_SPAWN[0])
    parser.add_argument("--spawn-y", type=float, default=DEFAULT_SPAWN[1])
    parser.add_argument("--spawn-z", type=float, default=DEFAULT_SPAWN[2])
    parser.add_argument("--skip-spawn", action="store_true", help="Use --actor instead of spawning a new Go1 actor")
    parser.add_argument("--actor", default="", help="Existing Go1 actor name when --skip-spawn is used")
    parser.add_argument("--wait", type=float, default=10.0, help="Seconds to wait for MuJoCo output artifacts")
    args = parser.parse_args()

    client = connect_client(args.host, args.port)
    try:
        level_name = request(client, "vget /level/name")
        print(f"LEVEL|{level_name}")

        if args.skip_spawn:
            if not args.actor:
                raise RuntimeError("--skip-spawn requires --actor")
            actor_name = args.actor
        else:
            actor_name = spawn_go1(client, (args.spawn_x, args.spawn_y, args.spawn_z))

        print(f"ACTOR|{actor_name}")
        print(f"MODE|{args.mode}")
        print(f"ACTOR_LOCATION|{request(client, f'vget /object/{actor_name}/location')}")

        start_command = (
            f"vset /object/{actor_name}/mujoco_quadruped_pose_preview/start go1"
            if args.mode == "pose_preview"
            else f"vset /object/{actor_name}/mujoco_quadruped_freefall/start go1"
        )
        start_result = request(client, start_command)
        if not start_result or str(start_result).lower().startswith("error"):
            raise RuntimeError(f"Failed to start MuJoCo Go1 {args.mode}: {start_result}")

        paths = parse_start_result(start_result)
        if not wait_for_file(paths["trajectory_path"], args.wait, min_size_bytes=32):
            raise RuntimeError(f"Trajectory file was not created: {paths['trajectory_path']}")

        status = wait_for_status(paths["status_path"], args.wait)
        print(f"STATUS|{status}")
        print_artifact_paths(paths)
        print(f"ACTOR_LOCATION_AFTER|{request(client, f'vget /object/{actor_name}/location')}")
        print(f"ACTOR_ROTATION_AFTER|{request(client, f'vget /object/{actor_name}/rotation')}")
        print_pose_summary(client, actor_name)
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
