#!/usr/bin/env python3
"""Showcase official Microduck skills around SchoolGym's four placed balls."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = SCRIPT_DIR.parent
for import_dir in (SCRIPT_DIR, MUJOCO_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from runtime.microduck_runtime import (  # noqa: E402
    MICRODUCK_BP_PATH,
    PosixKeyboard,
    encode_targets,
    parse_vector,
    read_windows_command,
    request,
    validate_bridge_contract,
)
from common.microduck_policy import MicroDuckPolicySet  # noqa: E402


SCHOOLGYM_LEVEL = "/Game/SchoolGym/Maps/SchoolGymDay"
# The SchoolGym map contains these four placed near-spherical balls. Assets
# 05-08 are elongated sports props despite sharing the SportBall prefix.
PLACED_BALL_PATTERN = re.compile(r"^SportBall01_0([1-4])(?:_|$)", re.IGNORECASE)
SKILL_SEQUENCE = (
    "walking",
    "standing",
    "sitstand",
    "ground_pick",
    "kick_left",
    "kick_right",
    "roulade",
)


@dataclass
class Ball:
    actor: str
    asset: str
    center_cm: np.ndarray
    mass_kg: float
    sliding_friction: float
    rolling_friction: float
    bounciness: float
    radius_cm: float


@dataclass
class DuckLane:
    index: int
    actor: str
    balls: list[Ball]
    skill: str
    policy: MicroDuckPolicySet
    start_delay: float
    active_label: str = "standing"
    state: dict[str, object] = field(default_factory=dict)
    previous_robot_contacts: dict[str, int] = field(default_factory=dict)


def parse_bounds(response: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [float(value) for value in response.replace(",", " ").split()],
        dtype=np.float64,
    )
    if values.size != 6 or not np.all(np.isfinite(values)):
        raise RuntimeError(f"Invalid Unreal object bounds: {response}")
    return values[:3], values[3:]


def spawn_named(
    client,
    asset_path: str,
    actor_name: str,
    location: tuple[float, float, float],
) -> str:
    return request(
        client,
        f"vset /objects/spawn_from_path {asset_path} {actor_name} "
        f"{location[0]:.6f} {location[1]:.6f} {location[2]:.6f}",
        verbose=False,
    )


def make_policy(args, skill: str) -> MicroDuckPolicySet:
    paths = {
        "walking": args.walking,
        "standing": args.standing,
    }
    skill_paths = {
        "sitstand": args.sitstand,
        "ground_pick": args.ground_pick,
        "kick_left": args.kick_left,
        "kick_right": args.kick_right,
        "roulade": args.roulade,
    }
    if skill in skill_paths:
        paths[skill] = skill_paths[skill]
    return MicroDuckPolicySet(
        paths,
        head_lowpass=args.head_lowpass,
        legs_lowpass=args.legs_lowpass,
    )


def discover_placed_balls(client, args) -> list[Ball]:
    objects = str(request(client, "vget /objects", verbose=False)).split()
    actors_by_slot: dict[int, str] = {}
    for actor in objects:
        match = PLACED_BALL_PATTERN.match(actor)
        if match:
            actors_by_slot.setdefault(int(match.group(1)), actor)
    missing = [slot for slot in range(1, 5) if slot not in actors_by_slot]
    if missing:
        raise RuntimeError(
            "SchoolGym placed balls are missing: "
            + ", ".join(f"SportBall01_{slot:02d}" for slot in missing)
        )

    balls: list[Ball] = []
    for slot in range(1, 5):
        actor = actors_by_slot[slot]
        bounds_min, bounds_max = parse_bounds(
            request(client, f"vget /object/{actor}/bounds", verbose=False)
        )
        extents = (bounds_max - bounds_min) * 0.5
        radius_cm = float(np.min(extents))
        if radius_cm <= 0.0 or float(np.max(extents) / radius_cm) > 1.25:
            raise RuntimeError(
                f"Placed ball {actor} is not sufficiently spherical: extents={extents}"
            )
        ball = Ball(
            actor=actor,
            asset=f"placed:SportBall01_{slot:02d}",
            center_cm=(bounds_min + bounds_max) * 0.5,
            mass_kg=args.ball_mass_kg,
            sliding_friction=args.ball_friction,
            rolling_friction=args.ball_rolling_friction,
            bounciness=args.ball_bounciness,
            radius_cm=radius_cm,
        )
        balls.append(ball)
        print(
            f"BALL_REUSE|slot={slot}|actor={actor}|"
            f"center_cm={','.join(f'{value:.2f}' for value in ball.center_cm)}|"
            f"radius_cm={radius_cm:.2f}"
        )
    return balls


def build_formation(client, args) -> tuple[list[DuckLane], list[str]]:
    camera_rotation = parse_vector(
        request(client, f"vget /camera/{args.camera_id}/rotation", verbose=False),
        "read SchoolGym camera rotation",
    )
    yaw_degrees = camera_rotation[1] + args.spawn_yaw_offset
    yaw = math.radians(yaw_degrees)
    forward = np.asarray((math.cos(yaw), math.sin(yaw)), dtype=np.float64)
    balls = discover_placed_balls(client, args)
    activity_xy = np.mean(
        np.stack([ball.center_cm[:2] for ball in balls], axis=0), axis=0
    )
    rng = np.random.default_rng(args.formation_seed)
    owned_actors: list[str] = []

    # Assign policies round-robin to the four existing balls. Multiple ducks
    # targeting one ball receive separate angular slots, and every duck faces
    # its assigned ball from the outside of the cluster.
    duck_specs: list[dict[str, object]] = []
    target_indices = [lane_index % len(balls) for lane_index in range(args.ducks)]
    target_counts = [target_indices.count(index) for index in range(len(balls))]
    target_slots = [0] * len(balls)
    for lane_index in range(args.ducks):
        skill = SKILL_SEQUENCE[lane_index % len(SKILL_SEQUENCE)]
        target_index = target_indices[lane_index]
        target_ball = balls[target_index]
        ball_xy = target_ball.center_cm[:2]
        cluster_offset = ball_xy - activity_xy
        if float(np.linalg.norm(cluster_offset)) < 1.0:
            base_angle = yaw + math.pi + target_index * math.tau / len(balls)
        else:
            base_angle = math.atan2(cluster_offset[1], cluster_offset[0])
        slot = target_slots[target_index]
        target_slots[target_index] += 1
        centered_slot = slot - 0.5 * (target_counts[target_index] - 1)
        radial_angle = base_angle + math.radians(
            centered_slot * args.duck_per_ball_angle_degrees
            + rng.uniform(
                -args.duck_angle_jitter_degrees,
                args.duck_angle_jitter_degrees,
            )
        )
        outward = np.asarray(
            (math.cos(radial_angle), math.sin(radial_angle)), dtype=np.float64
        )
        duck_radius = (
            args.kick_ball_distance_cm
            if skill in ("kick_left", "kick_right")
            else args.duck_ball_radius_cm
            + rng.uniform(-args.duck_radius_jitter_cm, args.duck_radius_jitter_cm)
        )
        duck_xy = ball_xy + outward * duck_radius
        target_delta = ball_xy - duck_xy
        lane_yaw = math.atan2(target_delta[1], target_delta[0]) + math.radians(
            rng.uniform(-args.duck_aim_jitter_degrees, args.duck_aim_jitter_degrees)
        )
        duck_specs.append(
            {
                "skill": skill,
                "xy": duck_xy,
                "forward": np.asarray(
                    (math.cos(lane_yaw), math.sin(lane_yaw)), dtype=np.float64
                ),
                "yaw_degrees": math.degrees(lane_yaw),
                "target_index": target_index,
                "spawn_z_cm": float(target_ball.center_cm[2] + 100.0),
            }
        )

    lanes: list[DuckLane] = []
    for lane_index, duck_spec in enumerate(duck_specs):
        duck_xy = np.asarray(duck_spec["xy"], dtype=np.float64)
        lane_yaw_degrees = float(duck_spec["yaw_degrees"])
        duck = spawn_named(
            client,
            MICRODUCK_BP_PATH,
            f"MjSchoolDuck_{lane_index + 1:02d}",
            (float(duck_xy[0]), float(duck_xy[1]), float(duck_spec["spawn_z_cm"])),
        )
        owned_actors.append(duck)
        request(
            client,
            f"vset /object/{duck}/rotation 0 {lane_yaw_degrees:.6f} 0",
            verbose=False,
        )
        request(
            client,
            f"vset /object/{duck}/settle_to_ground simple 75 "
            f"{args.settle_trace_length_cm:.6f} 0",
            verbose=False,
        )
        duck_location = parse_vector(
            request(client, f"vget /object/{duck}/location", verbose=False),
            f"read {duck} location",
        )
        skill = str(duck_spec["skill"])
        lanes.append(
            DuckLane(
                index=lane_index,
                actor=duck,
                balls=balls,
                skill=skill,
                policy=make_policy(args, skill),
                start_delay=lane_index * args.skill_start_stagger,
            )
        )
        print(
            f"DUCK_SPAWN|lane={lane_index + 1}|actor={duck}|skill={skill}|"
            f"target_ball={balls[int(duck_spec['target_index'])].actor}|"
            f"location_cm={','.join(f'{value:.2f}' for value in duck_location)}|"
            f"yaw={lane_yaw_degrees:.2f}"
        )

    if args.set_camera:
        right = np.asarray((-math.sin(yaw), math.cos(yaw)), dtype=np.float64)
        center_offset = right * args.camera_side_cm
        cluster_radius_cm = max(
            float(np.linalg.norm(ball.center_cm[:2] - activity_xy)) for ball in balls
        )
        view_distance_cm = max(
            args.camera_distance_cm,
            cluster_radius_cm + args.duck_ball_radius_cm + 180.0,
        )
        view_height_cm = max(
            args.camera_height_cm,
            math.tan(math.radians(abs(args.camera_pitch_degrees)))
            * view_distance_cm,
        )
        view_xy = activity_xy - forward * view_distance_cm + center_offset
        request(
            client,
            f"vset /camera/{args.camera_id}/location "
            f"{view_xy[0]:.6f} {view_xy[1]:.6f} "
            f"{max(ball.center_cm[2] for ball in balls) + view_height_cm:.6f}",
            verbose=False,
        )
        request(
            client,
            f"vset /camera/{args.camera_id}/rotation "
            f"{args.camera_pitch_degrees:.6f} {yaw_degrees:.6f} 0",
            verbose=False,
        )
        print(
            f"CAMERA|id={args.camera_id}|mode=formation-wide|"
            f"pitch={args.camera_pitch_degrees:.1f}|yaw={yaw_degrees:.1f}|"
            f"distance_cm={view_distance_cm:.1f}|height_cm={view_height_cm:.1f}"
        )
    return lanes, owned_actors


def configure_lane_props(client, lane: DuckLane) -> None:
    specs = ";".join(
        f"{ball.actor}:{ball.mass_kg:.6g}:{ball.sliding_friction:.6g}:"
        f"{ball.rolling_friction:.6g}:{ball.bounciness:.6g}:{ball.radius_cm:.6g}"
        for ball in lane.balls
    )
    response = request(
        client,
        f"vset /object/{lane.actor}/mujoco_microduck_dynamic_props {specs}",
        verbose=False,
    )
    print(f"DYNAMIC_PROPS|duck={lane.actor}|{response}")


def start_lane(client, lane: DuckLane) -> None:
    artifacts = request(
        client, f"vset /object/{lane.actor}/mujoco_microduck/start", verbose=False
    )
    lane.state = json.loads(
        request(
            client,
            f"vset /object/{lane.actor}/mujoco_microduck_control_sync/start",
            verbose=False,
        )
    )
    validate_bridge_contract(lane.state)
    dynamic_count = int(lane.state.get("dynamic_prop_count", -1))
    if dynamic_count != len(lane.balls):
        raise RuntimeError(
            f"{lane.actor} bridge has {dynamic_count} dynamic props; "
            f"expected {len(lane.balls)}. Rebuild UnrealCV."
        )
    print(
        f"MUJOCO_START|duck={lane.actor}|dynamic_props={dynamic_count}|"
        f"artifacts={artifacts}"
    )


def skill_control(
    lane: DuckLane, elapsed: float, args
) -> tuple[str, np.ndarray, float, str]:
    command = np.zeros(13, dtype=np.float32)
    local_time = elapsed - args.warmup - lane.start_delay
    if local_time < 0.0:
        return "standing", command, args.standing_action_scale, "warmup"

    if lane.skill == "walking":
        command[0] = min(0.4, args.vx)
        command[2] = 0.28 * math.sin(local_time * 0.55)
        return "walking", command, args.walking_action_scale, "walk_curve"
    if lane.skill == "standing":
        command[9] = 0.025 * math.sin(local_time * 0.9)  # body z
        command[10] = 0.12 * math.sin(local_time * 0.7)  # body roll
        command[11] = 0.10 * math.sin(local_time * 0.5)  # body pitch
        return "standing", command, args.standing_action_scale, "body_pose"
    if lane.skill == "sitstand":
        phase = local_time % 6.0
        if phase < 3.0:
            command[0] = 1.0
            return "sitstand", command, 1.0, "sit"
        if phase < 4.0:
            return "sitstand", command, 1.0, "rise"
        return "standing", command, args.standing_action_scale, "stand_recover"
    if lane.skill == "ground_pick":
        phase_time = local_time % 5.0
        if phase_time < 2.8:
            phase = phase_time / 4.0
            command[0] = math.cos(math.tau * phase)
            command[1] = math.sin(math.tau * phase)
            return "ground_pick", command, 1.0, "ground_pick"
        return "standing", command, args.standing_action_scale, "stand_recover"
    if lane.skill in ("kick_left", "kick_right"):
        if local_time % 4.0 < 0.5:
            return lane.skill, command, args.standing_action_scale, lane.skill
        return "standing", command, args.standing_action_scale, "stand_recover"
    if lane.skill == "roulade":
        if local_time % 5.0 < 1.0:
            return "roulade", command, 1.0, "roulade"
        return "standing", command, args.standing_action_scale, "stand_recover"
    raise RuntimeError(f"Unhandled Microduck skill {lane.skill!r}")


def log_lane_state(lane: DuckLane, step: int, elapsed: float) -> None:
    root = lane.state.get("root_position_ue_cm", [0.0, 0.0, 0.0])
    gravity = lane.state.get("projected_gravity", [0.0, 0.0, -1.0])
    print(
        f"DUCK_STEP|step={step}|t={elapsed:.3f}|lane={lane.index + 1}|"
        f"actor={lane.actor}|skill={lane.skill}|active={lane.active_label}|"
        f"root_cm=({root[0]:.1f},{root[1]:.1f},{root[2]:.1f})|"
        f"gravity=({gravity[0]:.3f},{gravity[1]:.3f},{gravity[2]:.3f})"
    )
    for prop in lane.state.get("dynamic_props", []):
        actor = str(prop.get("actor", "unknown"))
        velocity = np.asarray(prop.get("linear_velocity_mps", [0.0, 0.0, 0.0]))
        angular = np.asarray(prop.get("angular_velocity_rps", [0.0, 0.0, 0.0]))
        robot_contacts = int(prop.get("robot_contact_count", 0))
        robot_force = float(prop.get("robot_normal_force_n", 0.0))
        print(
            f"BALL_STATE|step={step}|lane={lane.index + 1}|ball={actor}|"
            f"speed_mps={np.linalg.norm(velocity):.3f}|"
            f"spin_rps={np.linalg.norm(angular):.3f}|"
            f"contacts=robot:{robot_contacts},env:{int(prop.get('environment_contact_count', 0))},"
            f"ball:{int(prop.get('prop_contact_count', 0))}|"
            f"normal_force_n={float(prop.get('normal_force_n', 0.0)):.3f}"
        )


def report_contact_events(lane: DuckLane, step: int) -> None:
    """Report short contacts even when they occur between diagnostic print frames."""
    for prop in lane.state.get("dynamic_props", []):
        actor = str(prop.get("actor", "unknown"))
        robot_contacts = int(prop.get("robot_contact_count", 0))
        previous = lane.previous_robot_contacts.get(actor, 0)
        if robot_contacts > 0 and previous == 0:
            velocity = np.asarray(prop.get("linear_velocity_mps", [0.0, 0.0, 0.0]))
            print(
                f"BALL_HIT|step={step}|lane={lane.index + 1}|ball={actor}|"
                f"robot_force_n={float(prop.get('robot_normal_force_n', 0.0)):.3f}|"
                f"speed_mps={np.linalg.norm(velocity):.3f}"
            )
        lane.previous_robot_contacts[actor] = robot_contacts


def run(client, lanes: list[DuckLane], args) -> None:
    # Tag/configure every ball before the first world samples its heightfield,
    # so balls owned by other lanes cannot become phantom terrain.
    for lane in lanes:
        configure_lane_props(client, lane)
    for lane in lanes:
        start_lane(client, lane)
    print(
        "CONTROL|skill_showcase=" + ("yes" if not args.manual else "manual-walk")
        + "|I/K=forward/back|J/L=turn|Space=stand|X/Esc=exit"
    )

    start = time.perf_counter()
    next_tick = start
    step = 0
    keyboard_context = PosixKeyboard() if args.manual and os.name != "nt" else None
    try:
        if keyboard_context:
            keyboard_context.__enter__()
        while True:
            elapsed = time.perf_counter() - start
            if args.duration > 0.0 and elapsed >= args.duration:
                break
            if os.name == "nt":
                keyboard_command, exit_requested = read_windows_command(
                    args.vx, args.yaw_rate
                )
            elif keyboard_context:
                keyboard_command, exit_requested = keyboard_context.read(
                    args.vx, args.yaw_rate
                )
            else:
                keyboard_command, exit_requested = (0.0, 0.0, 0.0), False
            if exit_requested:
                break

            batch_entries: list[str] = []
            for lane in lanes:
                if args.manual:
                    policy_name = "walking"
                    command = np.zeros(13, dtype=np.float32)
                    command[:3] = keyboard_command
                    action_scale = args.walking_action_scale
                    active_label = "manual_walk"
                else:
                    policy_name, command, action_scale, active_label = skill_control(
                        lane, elapsed, args
                    )
                # Holding Space is an emergency stop in both modes on Windows.
                if os.name == "nt":
                    import ctypes

                    if bool(ctypes.windll.user32.GetAsyncKeyState(0x20) & 0x8000):
                        policy_name = "standing"
                        command = np.zeros(13, dtype=np.float32)
                        action_scale = args.standing_action_scale
                        active_label = "emergency_stand"
                targets, _, _ = lane.policy.infer(
                    lane.state,
                    policy_name,
                    command,
                    action_scale=action_scale,
                )
                lane.active_label = active_label
                batch_entries.append(f"{lane.actor}:{encode_targets(targets)}")

            # One dispatcher invocation advances every world. A Python list of
            # UnrealCV requests is still scheduled command-by-command by UE and
            # costs roughly one game frame per duck.
            batch_response = request(
                client,
                "vset /mujoco_microduck_control_batch " + ";".join(batch_entries),
                verbose=False,
            )
            responses = json.loads(batch_response)
            if not isinstance(responses, list) or len(responses) != len(lanes):
                raise RuntimeError(
                    f"Invalid batched MicroDuck response: expected {len(lanes)}, "
                    f"got {type(responses).__name__}"
                )
            for lane, response in zip(lanes, responses):
                if not isinstance(response, dict):
                    raise RuntimeError(
                        f"Batched MicroDuck state for {lane.actor} is not an object"
                    )
                lane.state = response

            for lane in lanes:
                report_contact_events(lane, step)

            if step % args.print_every == 0:
                for lane in lanes:
                    log_lane_state(lane, step, elapsed)
            step += 1
            next_tick += 0.02
            sleep_seconds = next_tick - time.perf_counter()
            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)
            elif sleep_seconds < -0.25:
                print(f"REALTIME_LAG|step={step}|behind_ms={-sleep_seconds * 1000.0:.1f}")
                next_tick = time.perf_counter()
    finally:
        if keyboard_context:
            keyboard_context.__exit__(None, None, None)
        print(f"DONE|steps={step}|ducks={len(lanes)}")


def parse_args():
    policy_dir = MUJOCO_DIR / "policies" / "microduck"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--camera-id", default="0")
    parser.add_argument("--ducks", type=int, default=7)
    parser.add_argument("--formation-seed", type=int, default=17)
    parser.add_argument("--duck-ball-radius-cm", type=float, default=105.0)
    parser.add_argument("--duck-radius-jitter-cm", type=float, default=15.0)
    parser.add_argument("--duck-per-ball-angle-degrees", type=float, default=38.0)
    parser.add_argument("--duck-angle-jitter-degrees", type=float, default=13.0)
    parser.add_argument("--duck-aim-jitter-degrees", type=float, default=0.75)
    parser.add_argument("--spawn-yaw-offset", type=float, default=0.0)
    parser.add_argument("--kick-ball-distance-cm", type=float, default=34.0)
    parser.add_argument("--settle-trace-length-cm", type=float, default=5000.0)
    parser.add_argument("--set-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-distance-cm", type=float, default=380.0)
    parser.add_argument("--camera-side-cm", type=float, default=0.0)
    parser.add_argument("--camera-height-cm", type=float, default=260.0)
    parser.add_argument("--camera-pitch-degrees", type=float, default=-28.0)
    parser.add_argument("--walking", default=str(policy_dir / "alpha_walking.onnx"))
    parser.add_argument("--standing", default=str(policy_dir / "alpha_stand.onnx"))
    parser.add_argument("--sitstand", default=str(policy_dir / "alpha_sitstand.onnx"))
    parser.add_argument(
        "--ground-pick", default=str(policy_dir / "alpha_ground_pick.onnx")
    )
    parser.add_argument("--kick-left", default=str(policy_dir / "ball_kick_left.onnx"))
    parser.add_argument("--kick-right", default=str(policy_dir / "ball_kick_right.onnx"))
    parser.add_argument("--roulade", default=str(policy_dir / "roulade.onnx"))
    parser.add_argument("--walking-action-scale", type=float, default=0.9)
    parser.add_argument("--standing-action-scale", type=float, default=1.0)
    parser.add_argument("--head-lowpass", type=float, default=0.5)
    parser.add_argument("--legs-lowpass", type=float, default=0.7)
    parser.add_argument("--vx", type=float, default=0.36)
    parser.add_argument("--yaw-rate", type=float, default=0.8)
    parser.add_argument("--warmup", type=float, default=1.2)
    parser.add_argument("--skill-start-stagger", type=float, default=0.32)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--keep-actors", action="store_true")
    parser.add_argument("--allow-other-level", action="store_true")
    parser.add_argument("--ball-mass-kg", type=float, default=0.43)
    parser.add_argument("--ball-friction", type=float, default=0.58)
    parser.add_argument("--ball-rolling-friction", type=float, default=0.003)
    parser.add_argument("--ball-bounciness", type=float, default=0.38)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if not 1 <= args.ducks <= 16:
        parser.error("--ducks must be in [1, 16]")
    if not 0.0 <= args.vx <= 0.4:
        parser.error("--vx must be in [0, 0.4]")
    if args.duration < 0.0 or args.print_every <= 0:
        parser.error("--duration must be non-negative and --print-every positive")
    for name in (
        "duck_radius_jitter_cm",
        "duck_per_ball_angle_degrees",
        "duck_angle_jitter_degrees",
        "duck_aim_jitter_degrees",
        "ball_friction",
        "ball_rolling_friction",
        "ball_bounciness",
        "skill_start_stagger",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.duck_ball_radius_cm <= 0.0:
        parser.error("--duck-ball-radius-cm must be positive")
    if args.kick_ball_distance_cm <= 0.0:
        parser.error("ball placement distances must be positive")
    if args.ball_mass_kg <= 0.0:
        parser.error("--ball-mass-kg must be positive")
    return args


def main() -> int:
    args = parse_args()
    import unrealcv

    client = unrealcv.Client((args.host, args.port))
    client.connect()
    if not client.isconnected():
        raise RuntimeError(f"Failed to connect to UnrealCV at {args.host}:{args.port}")
    lanes: list[DuckLane] = []
    owned_actors: list[str] = []
    exit_code = 0
    try:
        level = request(client, "vget /level/name")
        print(f"LEVEL|{level}|expected={SCHOOLGYM_LEVEL}")
        if "schoolgym" not in level.casefold() and not args.allow_other_level:
            raise RuntimeError(
                "Open /Game/SchoolGym/Maps/SchoolGymDay in PIE before running this demo "
                "(or pass --allow-other-level for diagnostics)."
            )
        lanes, owned_actors = build_formation(client, args)
        run(client, lanes, args)
    except KeyboardInterrupt:
        print("INTERRUPTED|Ctrl+C")
        exit_code = 130
    finally:
        for lane in lanes:
            try:
                request(
                    client,
                    f"vset /object/{lane.actor}/mujoco_microduck/stop",
                    verbose=False,
                )
            except Exception as exc:
                print(f"STOP_WARNING|actor={lane.actor}|error={exc}")
        if not args.keep_actors:
            for actor in reversed(owned_actors):
                try:
                    request(client, f"vset /object/{actor}/destroy", verbose=False)
                except Exception as exc:
                    print(f"DESTROY_WARNING|actor={actor}|error={exc}")
        client.disconnect()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
