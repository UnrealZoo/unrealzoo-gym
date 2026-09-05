#!/usr/bin/env python3
"""Run five Microducks and two Go1 robots around DowntownWest's placed balls."""
from __future__ import annotations

import argparse
import json
import math
import os
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

from runtime import go1_locomotion as go1  # noqa: E402
import schoolgym as duck_demo  # noqa: E402
from runtime.microduck_runtime import (  # noqa: E402
    MICRODUCK_BP_PATH,
    encode_targets,
    parse_vector,
    request,
)


DOWNTOWNWEST_LEVEL_TOKEN = "downtownwest"
PLACED_BALL_ACTORS = (
    "StaticMeshActor_4",
    "StaticMeshActor_6",
    "StaticMeshActor_8",
)
CANONICAL_BALL_LOCATIONS_CM = {
    "StaticMeshActor_4": (5040.0, -3980.0, 20.0),
    "StaticMeshActor_6": (5100.0, -3950.0, 20.0),
    "StaticMeshActor_8": (5020.0, -4040.0, 20.0),
}
DUCK_SKILLS = ("walking", "walking", "roulade", "roulade", "kick_left")


def actor_name(args, kind: str, index: int) -> str:
    """Return a per-process name so rapid PIE reruns cannot hit pending-kill UObjects."""
    return f"MjDowntown{kind}_{args.run_tag}_{index + 1:02d}"


@dataclass
class GoLane:
    index: int
    actor: str
    role: str
    policy: go1.OnnxPolicy
    orbit_radius_m: float
    observation: list[float] = field(default_factory=list)
    payload: dict[str, object] = field(default_factory=dict)
    last_raw_action: list[float] = field(
        default_factory=lambda: [0.0] * go1.ACTION_DIM
    )
    last_command: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class BallPose:
    actor: str
    location_cm: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]


def discover_balls(client, args) -> list[duck_demo.Ball]:
    objects = set(str(request(client, "vget /objects", verbose=False)).split())
    missing = [actor for actor in PLACED_BALL_ACTORS if actor not in objects]
    if missing:
        raise RuntimeError(
            "DowntownWest placed ball actor(s) missing from PIE: "
            + ", ".join(missing)
            + ". "
            "Add/save the three actors, then restart PIE before running the demo."
        )

    balls: list[duck_demo.Ball] = []
    for index, actor in enumerate(PLACED_BALL_ACTORS, start=1):
        bounds_min, bounds_max = duck_demo.parse_bounds(
            request(client, f"vget /object/{actor}/bounds", verbose=False)
        )
        extents = (bounds_max - bounds_min) * 0.5
        radius_cm = float(np.min(extents))
        aspect = float(np.max(extents) / max(radius_cm, 1.0e-6))
        if radius_cm <= 0.0 or aspect > args.max_ball_bounds_aspect:
            raise RuntimeError(
                f"Placed actor {actor} is not sufficiently spherical: extents={extents}"
            )
        ball = duck_demo.Ball(
            actor=actor,
            asset=f"placed:{actor}",
            center_cm=(bounds_min + bounds_max) * 0.5,
            mass_kg=args.ball_mass_kg,
            sliding_friction=args.ball_friction,
            rolling_friction=args.ball_rolling_friction,
            bounciness=args.ball_bounciness,
            radius_cm=radius_cm,
        )
        balls.append(ball)
        print(
            f"BALL_REUSE|slot={index}|actor={actor}|"
            f"center_cm={','.join(f'{value:.2f}' for value in ball.center_cm)}|"
            f"radius_cm={radius_cm:.2f}"
        )
    return balls


def repair_scattered_ball_layout(client, args) -> None:
    if not args.repair_scattered_ball_layout:
        return
    locations = {
        actor: np.asarray(
            parse_vector(
                request(client, f"vget /object/{actor}/location", verbose=False),
                f"read {actor} location",
            ),
            dtype=np.float64,
        )
        for actor in PLACED_BALL_ACTORS
    }
    max_distance = max(
        float(np.linalg.norm(locations[a][:2] - locations[b][:2]))
        for index, a in enumerate(PLACED_BALL_ACTORS)
        for b in PLACED_BALL_ACTORS[index + 1 :]
    )
    if max_distance <= args.max_ball_cluster_span_cm:
        return
    print(
        f"BALL_LAYOUT_REPAIR|max_span_cm={max_distance:.2f}|"
        f"limit_cm={args.max_ball_cluster_span_cm:.2f}"
    )
    for actor in PLACED_BALL_ACTORS:
        target = CANONICAL_BALL_LOCATIONS_CM[actor]
        request(
            client,
            f"vset /object/{actor}/location "
            + " ".join(f"{value:.6f}" for value in target),
            verbose=False,
        )
        print(
            f"BALL_LAYOUT_SET|actor={actor}|location_cm="
            + ",".join(f"{value:.2f}" for value in target)
        )


def snapshot_ball_poses(client, balls: list[duck_demo.Ball]) -> list[BallPose]:
    poses: list[BallPose] = []
    for ball in balls:
        poses.append(
            BallPose(
                actor=ball.actor,
                location_cm=parse_vector(
                    request(client, f"vget /object/{ball.actor}/location", verbose=False),
                    f"read {ball.actor} location",
                ),
                rotation_deg=parse_vector(
                    request(client, f"vget /object/{ball.actor}/rotation", verbose=False),
                    f"read {ball.actor} rotation",
                ),
            )
        )
    return poses


def restore_ball_poses(client, poses: list[BallPose]) -> None:
    for pose in poses:
        request(
            client,
            f"vset /object/{pose.actor}/location "
            + " ".join(f"{value:.6f}" for value in pose.location_cm),
            verbose=False,
        )
        request(
            client,
            f"vset /object/{pose.actor}/rotation "
            + " ".join(f"{value:.6f}" for value in pose.rotation_deg),
            verbose=False,
        )
        print(
            f"BALL_RESTORE|actor={pose.actor}|location_cm="
            + ",".join(f"{value:.2f}" for value in pose.location_cm)
        )


def face_target(client, actor: str, source_xy: np.ndarray, target_xy: np.ndarray) -> float:
    delta = target_xy - source_xy
    yaw_degrees = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    request(
        client,
        f"vset /object/{actor}/rotation 0 {yaw_degrees:.6f} 0",
        verbose=False,
    )
    return yaw_degrees


def settle_actor(client, actor: str, root_height_cm: float, args) -> tuple[float, ...]:
    request(
        client,
        f"vset /object/{actor}/settle_to_ground simple "
        f"{args.settle_trace_start_cm:.6f} {args.settle_trace_length_cm:.6f} "
        f"{root_height_cm:.6f}",
        verbose=False,
    )
    return parse_vector(
        request(client, f"vget /object/{actor}/location", verbose=False),
        f"read settled location for {actor}",
    )


def build_ducks(client, balls: list[duck_demo.Ball], args) -> list[duck_demo.DuckLane]:
    rng = np.random.default_rng(args.formation_seed)
    lanes: list[duck_demo.DuckLane] = []
    assignments = (0, 1, 2, 0, 1)
    angles = (-145.0, -35.0, 145.0, 125.0, 25.0)
    for index, skill in enumerate(DUCK_SKILLS):
        ball = balls[assignments[index]]
        angle = math.radians(angles[index] + rng.uniform(-8.0, 8.0))
        radius = args.kick_ball_distance_cm if skill.startswith("kick_") else (
            args.duck_ball_radius_cm + rng.uniform(-10.0, 10.0)
        )
        xy = ball.center_cm[:2] + radius * np.asarray(
            (math.cos(angle), math.sin(angle)), dtype=np.float64
        )
        actor = duck_demo.spawn_named(
            client,
            MICRODUCK_BP_PATH,
            actor_name(args, "Duck", index),
            (float(xy[0]), float(xy[1]), float(ball.center_cm[2] + 130.0)),
        )
        yaw = face_target(client, actor, xy, ball.center_cm[:2])
        location = settle_actor(client, actor, 0.0, args)
        lane = duck_demo.DuckLane(
            index=index,
            actor=actor,
            balls=balls,
            skill=skill,
            policy=duck_demo.make_policy(args, skill),
            start_delay=index * args.skill_start_stagger,
        )
        lanes.append(lane)
        print(
            f"DUCK_SPAWN|lane={index + 1}|actor={actor}|skill={skill}|"
            f"target_ball={ball.actor}|location_cm="
            f"{','.join(f'{value:.2f}' for value in location)}|yaw={yaw:.2f}"
        )
    return lanes


def build_go1s(client, balls: list[duck_demo.Ball], args) -> list[GoLane]:
    center_xy = np.mean(np.stack([ball.center_cm[:2] for ball in balls]), axis=0)
    max_z = max(float(ball.center_cm[2]) for ball in balls)
    policy_path = Path(args.go_policy).expanduser().resolve()
    lanes: list[GoLane] = []
    # Keep Go1 on the outside of the dynamic-ball area. The present Go bridge
    # does not yet share MicroDuck's dynamic-prop world, so these paths showcase
    # policy motion and static-scene collision without pretending to push balls.
    specs = (
        ("orbit_inner", 215.0, -70.0),
        ("orbit_outer", 275.0, 110.0),
    )
    for index, (role, radius, angle_degrees) in enumerate(specs):
        angle = math.radians(angle_degrees)
        xy = center_xy + radius * np.asarray(
            (math.cos(angle), math.sin(angle)), dtype=np.float64
        )
        actor = duck_demo.spawn_named(
            client,
            go1.GO1_BP_PATH,
            actor_name(args, "Go1", index),
            (float(xy[0]), float(xy[1]), max_z + 170.0),
        )
        # Both Go1 robots start tangent to the ball cluster and turn left at
        # vx/radius, producing stable concentric counter-clockwise orbits.
        yaw = angle_degrees + 90.0
        request(
            client,
            f"vset /object/{actor}/rotation 0 {yaw:.6f} 0",
            verbose=False,
        )
        location = settle_actor(client, actor, args.go_root_height_cm, args)
        lanes.append(
            GoLane(
                index=index,
                actor=actor,
                role=role,
                policy=go1.OnnxPolicy(policy_path),
                orbit_radius_m=radius * 0.01,
            )
        )
        print(
            f"GO1_SPAWN|lane={index + 1}|actor={actor}|role={role}|"
            f"location_cm={','.join(f'{value:.2f}' for value in location)}|yaw={yaw:.2f}"
        )
    return lanes


def configure_camera(client, balls: list[duck_demo.Ball], args) -> None:
    if not args.set_camera:
        return
    camera_rotation = parse_vector(
        request(client, f"vget /camera/{args.camera_id}/rotation", verbose=False),
        "read camera rotation",
    )
    yaw_degrees = camera_rotation[1] + args.camera_yaw_offset_degrees
    yaw = math.radians(yaw_degrees)
    forward = np.asarray((math.cos(yaw), math.sin(yaw)), dtype=np.float64)
    right = np.asarray((-math.sin(yaw), math.cos(yaw)), dtype=np.float64)
    center_xy = np.mean(np.stack([ball.center_cm[:2] for ball in balls]), axis=0)
    location_xy = (
        center_xy - forward * args.camera_distance_cm + right * args.camera_side_cm
    )
    center_z = float(np.mean([ball.center_cm[2] for ball in balls]))
    request(
        client,
        f"vset /camera/{args.camera_id}/location {location_xy[0]:.6f} "
        f"{location_xy[1]:.6f} {center_z + args.camera_height_cm:.6f}",
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
        f"distance_cm={args.camera_distance_cm:.1f}|height_cm={args.camera_height_cm:.1f}"
    )


def configure_go_props(client, lane: GoLane, balls: list[duck_demo.Ball]) -> None:
    specs = ";".join(
        f"{ball.actor}:{ball.mass_kg:.6g}:{ball.sliding_friction:.6g}:"
        f"{ball.rolling_friction:.6g}:{ball.bounciness:.6g}:{ball.radius_cm:.6g}"
        for ball in balls
    )
    response = request(
        client,
        f"vset /object/{lane.actor}/mujoco_go1_dynamic_props {specs}",
        verbose=False,
    )
    print(f"GO1_DYNAMIC_PROPS|actor={lane.actor}|{response}")


def duck_skill_control(
    lane: duck_demo.DuckLane, elapsed: float, args
) -> tuple[str, np.ndarray, float, str]:
    """A faster showcase cadence using the official policy command contracts."""
    command = np.zeros(13, dtype=np.float32)
    local_time = elapsed - args.warmup - lane.start_delay
    if local_time < 0.0:
        return "standing", command, args.standing_action_scale, "warmup"
    if lane.skill == "walking":
        # Deterministic pseudo-random segments keep the two walkers active and
        # crossing the shared ball area without requiring a second RNG thread.
        segment = int(local_time / args.walk_segment_seconds)
        seed = (segment + 1) * (lane.index + 3)
        command[0] = min(0.4, args.vx) * (0.76 + 0.24 * abs(math.sin(seed * 1.73)))
        command[1] = 0.16 * math.sin(seed * 2.31 + lane.index)
        command[2] = args.yaw_rate * math.sin(seed * 1.17 + 0.8 * lane.index)
        return "walking", command, args.walking_action_scale, "random_run"
    if lane.skill == "sitstand":
        phase = local_time % 4.0
        if phase < 2.0:
            command[0] = 1.0
            return "sitstand", command, 1.0, "sit"
        if phase < 2.7:
            return "sitstand", command, 1.0, "rise"
        return "standing", command, args.standing_action_scale, "stand_recover"
    if lane.skill == "ground_pick":
        phase_time = local_time % 3.5
        if phase_time < 2.5:
            phase = phase_time / 3.5
            command[0] = math.cos(math.tau * phase)
            command[1] = math.sin(math.tau * phase)
            return "ground_pick", command, 1.0, "ground_pick"
        return "standing", command, args.standing_action_scale, "stand_recover"
    if lane.skill in ("kick_left", "kick_right"):
        if local_time % 2.6 < 0.55:
            return lane.skill, command, args.standing_action_scale, lane.skill
        return "standing", command, args.standing_action_scale, "stand_recover"
    if lane.skill == "roulade":
        phase = local_time % args.roulade_cycle_seconds
        if phase < args.roulade_active_seconds:
            return "roulade", command, 1.0, "continuous_roulade"
        return "standing", command, args.standing_action_scale, "roll_recover"
    return duck_demo.skill_control(lane, elapsed, args)


def go_command(lane: GoLane, elapsed: float, args) -> tuple[float, float, float]:
    if elapsed < args.warmup:
        return 0.0, 0.0, 0.0
    yaw_rate = args.go_vx / max(lane.orbit_radius_m, 0.5)
    return args.go_vx, 0.0, min(args.go_max_yaw_rate, yaw_rate)


def exit_requested() -> bool:
    if os.name != "nt":
        return False
    import ctypes

    key_state = ctypes.windll.user32.GetAsyncKeyState
    return bool(key_state(ord("X")) & 0x8000) or bool(key_state(0x1B) & 0x8000)


def run(
    client,
    duck_lanes: list[duck_demo.DuckLane],
    go_lanes: list[GoLane],
    args,
) -> None:
    # Configure all props first. The tag/mobility change keeps the balls out of
    # subsequently exported static environment collision.
    for lane in duck_lanes:
        duck_demo.configure_lane_props(client, lane)
    # Creating/configuring Go components before Microduck MJCF generation lets
    # each Duck world discover a live Go peer proxy. Go receives the reciprocal
    # live ball proxies when its own MJCF is generated below.
    for lane in go_lanes:
        configure_go_props(client, lane, duck_lanes[0].balls)
    for lane in duck_lanes:
        duck_demo.start_lane(client, lane)
    for lane in go_lanes:
        lane.observation, lane.payload = go1.start_synchronous_policy(
            client, lane.actor
        )

    print(
        "CONTROL|duck_policy=official-release-skills|"
        "go_policy=go1_velocity|go_roles=concentric_orbits|X/Esc/Ctrl+C=exit"
    )
    print(
        "COLLISION_SCOPE|duck-ball=shared-dynamic|duck-scene=mujoco|"
        "go-scene=mujoco|go-ball=live-kinematic|duck-go=live-kinematic"
    )
    start = time.perf_counter()
    next_tick = start
    step = 0
    while True:
        elapsed = step * 0.02
        if args.duration > 0.0 and elapsed >= args.duration:
            break
        if exit_requested():
            print("EXIT_KEY|X-or-Esc")
            break

        batch_entries: list[str] = []
        for lane in duck_lanes:
            policy_name, command, action_scale, active_label = duck_skill_control(
                lane, elapsed, args
            )
            targets, _, _ = lane.policy.infer(
                lane.state,
                policy_name,
                command,
                action_scale=action_scale,
            )
            lane.active_label = active_label
            batch_entries.append(f"{lane.actor}:{encode_targets(targets)}")
        go_entries: list[str] = []
        pending_go_actions: list[list[float]] = []
        for lane in go_lanes:
            command = go_command(lane, elapsed, args)
            if command != lane.last_command:
                go1.send_policy_command(client, lane.actor, command)
                lane.last_command = command
                print(
                    f"GO1_COMMAND|step={step}|actor={lane.actor}|role={lane.role}|"
                    f"command=({command[0]:.3f},{command[1]:.3f},{command[2]:.3f})"
                )
            policy_obs = go1.prepare_policy_obs(
                lane.policy, lane.observation, lane.last_raw_action
            )
            if elapsed < args.warmup:
                raw_action = [0.0] * go1.ACTION_DIM
            else:
                raw_action = go1.clamp_action(
                    lane.policy.infer(policy_obs), args.go_raw_action_clip, 1.0
                )
            go_entries.append(
                f"{lane.actor}:" + ",".join(f"{float(value):.9f}" for value in raw_action)
            )
            pending_go_actions.append(list(raw_action))

        batch_response = json.loads(
            request(
                client,
                "vset /mujoco/robot_control_batch "
                + ";".join(batch_entries + go_entries),
                verbose=False,
            )
        )
        robot_entries = (
            batch_response.get("robots") if isinstance(batch_response, dict) else None
        )
        if not isinstance(robot_entries, list):
            raise RuntimeError(f"Invalid generic robot batch response: {batch_response}")
        states_by_actor = {
            entry.get("actor"): entry.get("state")
            for entry in robot_entries
            if isinstance(entry, dict)
            and isinstance(entry.get("actor"), str)
            and isinstance(entry.get("state"), dict)
        }
        expected_actors = {lane.actor for lane in duck_lanes + go_lanes}
        if set(states_by_actor) != expected_actors:
            raise RuntimeError(
                "Generic robot batch response actor mismatch: "
                f"expected={sorted(expected_actors)} actual={sorted(states_by_actor)}"
            )
        for lane in duck_lanes:
            response = states_by_actor[lane.actor]
            if not isinstance(response, dict):
                raise RuntimeError(f"Invalid MicroDuck state for {lane.actor}")
            lane.state = response
            duck_demo.report_contact_events(lane, step)
        for lane, raw_action in zip(go_lanes, pending_go_actions):
            response = states_by_actor[lane.actor]
            if not isinstance(response, dict):
                raise RuntimeError(f"Invalid Go1 state for {lane.actor}")
            observation = response.get("obs")
            if not isinstance(observation, list) or len(observation) != go1.OBS_DIM:
                raise RuntimeError(f"Invalid Go1 observation for {lane.actor}")
            lane.observation = [float(value) for value in observation]
            lane.payload = response
            lane.last_raw_action = raw_action

        if step % args.print_every == 0:
            for lane in duck_lanes:
                duck_demo.log_lane_state(lane, step, elapsed)
            for lane in go_lanes:
                root = lane.payload.get("root_position_ue_cm", [0.0, 0.0, 0.0])
                gravity = lane.payload.get("projected_gravity", [0.0, 0.0, -1.0])
                print(
                    f"GO1_STEP|step={step}|t={elapsed:.3f}|lane={lane.index + 1}|"
                    f"actor={lane.actor}|role={lane.role}|"
                    f"root_cm=({root[0]:.1f},{root[1]:.1f},{root[2]:.1f})|"
                    f"gravity=({gravity[0]:.3f},{gravity[1]:.3f},{gravity[2]:.3f})"
                )

        step += 1
        next_tick += 0.02
        delay = next_tick - time.perf_counter()
        if delay > 0.0:
            time.sleep(delay)
        elif delay < -0.25:
            print(f"REALTIME_LAG|step={step}|behind_ms={-delay * 1000.0:.1f}")
            next_tick = time.perf_counter()
    print(f"DONE|steps={step}|ducks={len(duck_lanes)}|go1={len(go_lanes)}")


def parse_args():
    microduck_dir = MUJOCO_DIR / "policies" / "microduck"
    go_policy = MUJOCO_DIR / "policies" / "go1" / "velocity" / "policy.onnx"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--camera-id", default="0")
    parser.add_argument("--formation-seed", type=int, default=23)
    parser.add_argument(
        "--repair-scattered-ball-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-ball-cluster-span-cm", type=float, default=300.0)
    parser.add_argument("--max-ball-bounds-aspect", type=float, default=1.5)
    parser.add_argument("--duck-ball-radius-cm", type=float, default=92.0)
    parser.add_argument("--kick-ball-distance-cm", type=float, default=34.0)
    parser.add_argument("--settle-trace-start-cm", type=float, default=75.0)
    parser.add_argument("--settle-trace-length-cm", type=float, default=5000.0)
    parser.add_argument("--go-root-height-cm", type=float, default=35.0)
    parser.add_argument("--set-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-distance-cm", type=float, default=520.0)
    parser.add_argument("--camera-side-cm", type=float, default=0.0)
    parser.add_argument("--camera-height-cm", type=float, default=320.0)
    parser.add_argument("--camera-pitch-degrees", type=float, default=-30.0)
    parser.add_argument("--camera-yaw-offset-degrees", type=float, default=0.0)
    parser.add_argument("--walking", default=str(microduck_dir / "alpha_walking.onnx"))
    parser.add_argument("--standing", default=str(microduck_dir / "alpha_stand.onnx"))
    parser.add_argument("--sitstand", default=str(microduck_dir / "alpha_sitstand.onnx"))
    parser.add_argument("--ground-pick", default=str(microduck_dir / "alpha_ground_pick.onnx"))
    parser.add_argument("--kick-left", default=str(microduck_dir / "ball_kick_left.onnx"))
    parser.add_argument("--kick-right", default=str(microduck_dir / "ball_kick_right.onnx"))
    parser.add_argument("--roulade", default=str(microduck_dir / "roulade.onnx"))
    parser.add_argument("--walking-action-scale", type=float, default=0.9)
    parser.add_argument("--standing-action-scale", type=float, default=1.0)
    parser.add_argument("--head-lowpass", type=float, default=0.5)
    parser.add_argument("--legs-lowpass", type=float, default=0.7)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--yaw-rate", type=float, default=0.8)
    parser.add_argument("--warmup", type=float, default=0.8)
    parser.add_argument("--skill-start-stagger", type=float, default=0.18)
    parser.add_argument("--walk-segment-seconds", type=float, default=1.2)
    parser.add_argument("--roulade-cycle-seconds", type=float, default=2.2)
    parser.add_argument("--roulade-active-seconds", type=float, default=1.15)
    parser.add_argument("--go-policy", default=str(go_policy))
    parser.add_argument("--go-vx", type=float, default=0.55)
    parser.add_argument("--go-max-yaw-rate", type=float, default=0.45)
    parser.add_argument("--go-raw-action-clip", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--keep-actors", action="store_true")
    parser.add_argument("--allow-other-level", action="store_true")
    parser.add_argument("--ball-mass-kg", type=float, default=0.43)
    parser.add_argument("--ball-friction", type=float, default=0.58)
    parser.add_argument("--ball-rolling-friction", type=float, default=0.003)
    parser.add_argument("--ball-bounciness", type=float, default=0.38)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.duration < 0.0 or args.print_every <= 0:
        parser.error("--duration must be non-negative and --print-every positive")
    if not 0.0 <= args.vx <= 0.4:
        parser.error("--vx must be in [0, 0.4]")
    if args.go_raw_action_clip <= 0.0:
        parser.error("--go-raw-action-clip must be positive")
    if args.walk_segment_seconds <= 0.0:
        parser.error("--walk-segment-seconds must be positive")
    if not 0.0 < args.roulade_active_seconds < args.roulade_cycle_seconds:
        parser.error("--roulade-active-seconds must be in (0, --roulade-cycle-seconds)")
    if args.max_ball_cluster_span_cm <= 0.0:
        parser.error("--max-ball-cluster-span-cm must be positive")
    if args.max_ball_bounds_aspect < 1.0:
        parser.error("--max-ball-bounds-aspect must be at least 1")
    if args.ball_mass_kg <= 0.0:
        parser.error("--ball-mass-kg must be positive")
    return args


def main() -> int:
    args = parse_args()
    args.run_tag = f"{os.getpid()}_{time.time_ns() % 1_000_000_000:09d}"
    client = go1.connect_client(args.host, args.port)
    duck_lanes: list[duck_demo.DuckLane] = []
    go_lanes: list[GoLane] = []
    owned_actors: list[str] = []
    ball_poses: list[BallPose] = []
    exit_code = 0
    try:
        level = str(request(client, "vget /level/name", verbose=False))
        print(f"LEVEL|{level}|expected=DowntownWest")
        if DOWNTOWNWEST_LEVEL_TOKEN not in level.casefold() and not args.allow_other_level:
            raise RuntimeError(
                "Open DowntownWest in PIE before running this demo "
                "(or pass --allow-other-level for diagnostics)."
            )
        repair_scattered_ball_layout(client, args)
        balls = discover_balls(client, args)
        ball_poses = snapshot_ball_poses(client, balls)
        duck_lanes = build_ducks(client, balls, args)
        owned_actors.extend(lane.actor for lane in duck_lanes)
        go_lanes = build_go1s(client, balls, args)
        owned_actors.extend(lane.actor for lane in go_lanes)
        configure_camera(client, balls, args)
        run(client, duck_lanes, go_lanes, args)
    except KeyboardInterrupt:
        print("INTERRUPTED|Ctrl+C")
        exit_code = 130
    finally:
        for lane in duck_lanes:
            try:
                request(
                    client,
                    f"vset /object/{lane.actor}/mujoco_microduck/stop",
                    verbose=False,
                )
            except Exception as exc:
                print(f"STOP_WARNING|actor={lane.actor}|error={exc}")
        for lane in go_lanes:
            try:
                go1.stop_go1_simulation(client, lane.actor)
            except Exception as exc:
                print(f"STOP_WARNING|actor={lane.actor}|error={exc}")
        if not args.keep_actors:
            for actor in reversed(owned_actors):
                try:
                    request(client, f"vset /object/{actor}/destroy", verbose=False)
                except Exception as exc:
                    print(f"DESTROY_WARNING|actor={actor}|error={exc}")
        if ball_poses:
            try:
                restore_ball_poses(client, ball_poses)
            except Exception as exc:
                print(f"BALL_RESTORE_WARNING|error={exc}")
        client.disconnect()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
