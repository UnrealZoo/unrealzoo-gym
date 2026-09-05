#!/usr/bin/env python3
"""Run one Go1 leader followed in single file by three Microducks."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MUJOCO_DIR = SCRIPT_DIR.parent
for import_dir in (SCRIPT_DIR, MUJOCO_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import downtown_showcase as showcase  # noqa: E402
from runtime import go1_locomotion as go1  # noqa: E402
import schoolgym as duck_demo  # noqa: E402
from runtime.microduck_runtime import (  # noqa: E402
    MICRODUCK_BP_PATH,
    encode_targets,
    parse_vector,
    request,
)


FOLLOWER_COUNT = 3
CONTROL_DT = 0.02


@dataclass
class Follower:
    lane: duck_demo.DuckLane
    target_actor: str
    heading_degrees: float
    desired_spacing_cm: float
    distance_cm: float = 0.0
    heading_error_degrees: float = 0.0
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def root_xy(state: dict[str, object], actor: str) -> np.ndarray:
    root = state.get("root_position_ue_cm")
    if not isinstance(root, list) or len(root) != 3:
        raise RuntimeError(f"Invalid root position for {actor}: {root}")
    result = np.asarray(root[:2], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"Non-finite root position for {actor}: {root}")
    return result


def actor_heading(client, actor: str) -> float:
    rotation = parse_vector(
        request(client, f"vget /object/{actor}/rotation", verbose=False),
        f"read rotation for {actor}",
    )
    return float(rotation[1])


def spawn_formation(client, balls: list[duck_demo.Ball], args):
    center_xy = np.mean(np.stack([ball.center_cm[:2] for ball in balls]), axis=0)
    max_z = max(float(ball.center_cm[2]) for ball in balls)
    angle_degrees = args.start_angle_degrees
    angle = math.radians(angle_degrees)
    radial = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
    tangent = np.asarray((-math.sin(angle), math.cos(angle)), dtype=np.float64)
    leader_xy = center_xy + args.orbit_radius_cm * radial
    heading = angle_degrees + 90.0

    leader_actor = duck_demo.spawn_named(
        client,
        go1.GO1_BP_PATH,
        showcase.actor_name(args, "FollowGo1", 0),
        (float(leader_xy[0]), float(leader_xy[1]), max_z + 170.0),
    )
    request(
        client,
        f"vset /object/{leader_actor}/rotation 0 {heading:.6f} 0",
        verbose=False,
    )
    leader_location = showcase.settle_actor(
        client, leader_actor, args.go_root_height_cm, args
    )
    leader = showcase.GoLane(
        index=0,
        actor=leader_actor,
        role="queue_leader",
        policy=go1.OnnxPolicy(Path(args.go_policy).expanduser().resolve()),
        orbit_radius_m=args.orbit_radius_cm * 0.01,
    )
    print(
        f"LEADER_SPAWN|actor={leader_actor}|location_cm="
        f"{','.join(f'{value:.2f}' for value in leader_location)}|yaw={heading:.2f}"
    )

    followers: list[Follower] = []
    predecessor = leader_actor
    for index in range(FOLLOWER_COUNT):
        # Put the first duck slightly farther from Go1 because Go1 has a much
        # larger rear collision envelope. Subsequent ducks use the requested
        # queue spacing directly.
        offset = args.leader_gap_cm + index * args.follow_spacing_cm
        xy = leader_xy - tangent * offset
        actor = duck_demo.spawn_named(
            client,
            MICRODUCK_BP_PATH,
            showcase.actor_name(args, "FollowDuck", index),
            (float(xy[0]), float(xy[1]), max_z + 130.0),
        )
        request(
            client,
            f"vset /object/{actor}/rotation 0 {heading:.6f} 0",
            verbose=False,
        )
        location = showcase.settle_actor(client, actor, 0.0, args)
        lane = duck_demo.DuckLane(
            index=index,
            actor=actor,
            balls=balls,
            skill="walking",
            policy=duck_demo.make_policy(args, "walking"),
            start_delay=0.0,
        )
        followers.append(
            Follower(
                lane=lane,
                target_actor=predecessor,
                heading_degrees=heading,
                desired_spacing_cm=(
                    args.leader_gap_cm if index == 0 else args.follow_spacing_cm
                ),
            )
        )
        predecessor = actor
        print(
            f"FOLLOWER_SPAWN|queue={index + 1}|actor={actor}|target="
            f"{followers[-1].target_actor}|location_cm="
            f"{','.join(f'{value:.2f}' for value in location)}|yaw={heading:.2f}"
        )
    return leader, followers


def follower_control(
    follower: Follower,
    own_xy: np.ndarray,
    target_xy: np.ndarray,
    elapsed: float,
    args,
) -> tuple[str, np.ndarray, float]:
    command = np.zeros(13, dtype=np.float32)
    delta = target_xy - own_xy
    distance_cm = float(np.linalg.norm(delta))
    desired_heading = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    heading_error = wrap_degrees(desired_heading - follower.heading_degrees)
    follower.distance_cm = distance_cm
    follower.heading_error_degrees = heading_error

    if elapsed < args.warmup:
        follower.command = (0.0, 0.0, 0.0)
        return "standing", command, args.standing_action_scale

    # Use leader-speed feed-forward so an intact queue starts together. Pure
    # distance-error control leaves every follower stationary until its
    # predecessor has opened a gap, causing a long cascade delay (and, if the
    # leader checkpoint does not break into gait, an indefinitely frozen line).
    spacing_error_cm = distance_cm - follower.desired_spacing_cm
    if abs(spacing_error_cm) <= args.spacing_deadband_cm:
        spacing_error_cm = 0.0
    vx = float(
        np.clip(
            args.duck_cruise_vx + args.distance_gain * spacing_error_cm * 0.01,
            0.0,
            args.duck_max_vx,
        )
    )
    # Turn in place when the predecessor is well off the nose. This prevents
    # a follower from cutting across the inside of the orbit and breaking rank.
    alignment = max(0.0, math.cos(math.radians(heading_error)))
    if abs(heading_error) >= args.turn_in_place_degrees:
        alignment = 0.0
    vx *= alignment
    yaw_rate = float(
        np.clip(
            args.heading_gain * math.radians(heading_error),
            -args.duck_max_yaw_rate,
            args.duck_max_yaw_rate,
        )
    )
    if distance_cm < follower.desired_spacing_cm * args.emergency_stop_ratio:
        vx = 0.0
    command[0] = vx
    # The official Microduck checkpoint uses the opposite yaw sign from
    # Unreal's actor-yaw convention. Keep the controller/error in UE space and
    # invert only at the policy boundary.
    command[2] = -yaw_rate
    follower.command = (vx, 0.0, -yaw_rate)
    follower.heading_degrees = wrap_degrees(
        follower.heading_degrees + math.degrees(yaw_rate * CONTROL_DT)
    )
    moving = vx > 0.015 or abs(yaw_rate) > 0.04
    return (
        "walking" if moving else "standing",
        command,
        args.walking_action_scale if moving else args.standing_action_scale,
    )


def parse_batch_response(response: str, expected: set[str]) -> dict[str, dict]:
    payload = json.loads(response)
    entries = payload.get("robots") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError(f"Invalid generic robot batch response: {payload}")
    states = {
        entry.get("actor"): entry.get("state")
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("actor"), str)
        and isinstance(entry.get("state"), dict)
    }
    if set(states) != expected:
        raise RuntimeError(
            "Generic robot batch response actor mismatch: "
            f"expected={sorted(expected)} actual={sorted(states)}"
        )
    return states


def run(client, leader: showcase.GoLane, followers: list[Follower], args) -> None:
    for follower in followers:
        duck_demo.configure_lane_props(client, follower.lane)
    showcase.configure_go_props(client, leader, followers[0].lane.balls)
    for follower in followers:
        duck_demo.start_lane(client, follower.lane)
    leader.observation, leader.payload = go1.start_synchronous_policy(
        client, leader.actor
    )

    expected = {leader.actor, *(follower.lane.actor for follower in followers)}
    print(
        "CONTROL|formation=go1-leader+3-microduck-single-file|"
        "tracking=immediate-predecessor|X/Esc/Ctrl+C=exit"
    )
    print(
        "COLLISION_SCOPE|scene=mujoco-runtime-environment|balls=shared-dynamic|"
        "robots=live-kinematic-peer-proxies"
    )
    next_tick = time.perf_counter()
    step = 0
    while True:
        elapsed = step * CONTROL_DT
        if args.duration > 0.0 and elapsed >= args.duration:
            break
        if showcase.exit_requested():
            print("EXIT_KEY|X-or-Esc")
            break

        if step % args.heading_refresh_steps == 0:
            for follower in followers:
                follower.heading_degrees = actor_heading(client, follower.lane.actor)

        leader_command = (0.0, 0.0, 0.0)
        if elapsed >= args.warmup:
            leader_command = (
                args.go_vx,
                0.0,
                min(
                    args.go_max_yaw_rate,
                    args.go_vx / max(leader.orbit_radius_m, 0.5),
                ),
            )
        if leader_command != leader.last_command:
            go1.send_policy_command(client, leader.actor, leader_command)
            leader.last_command = leader_command
            print(
                f"LEADER_COMMAND|step={step}|command="
                f"({leader_command[0]:.3f},0.000,{leader_command[2]:.3f})"
            )
        policy_obs = go1.prepare_policy_obs(
            leader.policy, leader.observation, leader.last_raw_action
        )
        if elapsed < args.warmup:
            leader_action = [0.0] * go1.ACTION_DIM
        else:
            leader_action = go1.clamp_action(
                leader.policy.infer(policy_obs), args.go_raw_action_clip, 1.0
            )

        positions = {leader.actor: root_xy(leader.payload, leader.actor)}
        positions.update(
            {
                follower.lane.actor: root_xy(
                    follower.lane.state, follower.lane.actor
                )
                for follower in followers
            }
        )
        batch_entries: list[str] = []
        for follower in followers:
            lane = follower.lane
            policy_name, command, action_scale = follower_control(
                follower,
                positions[lane.actor],
                positions[follower.target_actor],
                elapsed,
                args,
            )
            targets, _, _ = lane.policy.infer(
                lane.state, policy_name, command, action_scale=action_scale
            )
            lane.active_label = "follow" if policy_name == "walking" else "hold"
            batch_entries.append(f"{lane.actor}:{encode_targets(targets)}")
        batch_entries.append(
            f"{leader.actor}:"
            + ",".join(f"{float(value):.9f}" for value in leader_action)
        )

        states = parse_batch_response(
            request(
                client,
                "vset /mujoco/robot_control_batch " + ";".join(batch_entries),
                verbose=False,
            ),
            expected,
        )
        for follower in followers:
            follower.lane.state = states[follower.lane.actor]
            duck_demo.report_contact_events(follower.lane, step)
        leader_state = states[leader.actor]
        observation = leader_state.get("obs")
        if not isinstance(observation, list) or len(observation) != go1.OBS_DIM:
            raise RuntimeError(f"Invalid Go1 observation for {leader.actor}")
        leader.observation = [float(value) for value in observation]
        leader.payload = leader_state
        leader.last_raw_action = list(leader_action)

        if step % args.print_every == 0:
            leader_root = leader.payload.get("root_position_ue_cm", [0.0, 0.0, 0.0])
            print(
                f"QUEUE_STEP|step={step}|t={elapsed:.2f}|leader="
                f"({leader_root[0]:.1f},{leader_root[1]:.1f},{leader_root[2]:.1f})"
            )
            for index, follower in enumerate(followers, start=1):
                root = follower.lane.state.get(
                    "root_position_ue_cm", [0.0, 0.0, 0.0]
                )
                print(
                    f"FOLLOW_STEP|queue={index}|actor={follower.lane.actor}|"
                    f"target={follower.target_actor}|"
                    f"root_cm=({root[0]:.1f},{root[1]:.1f},{root[2]:.1f})|"
                    f"distance_cm={follower.distance_cm:.1f}|"
                    f"heading_error_deg={follower.heading_error_degrees:.1f}|"
                    f"cmd=({follower.command[0]:.3f},{follower.command[2]:.3f})"
                )

        step += 1
        next_tick += CONTROL_DT
        delay = next_tick - time.perf_counter()
        if delay > 0.0:
            time.sleep(delay)
        elif delay < -0.25:
            print(f"REALTIME_LAG|step={step}|behind_ms={-delay * 1000.0:.1f}")
            next_tick = time.perf_counter()
    print(f"DONE|steps={step}|leader=1|followers={len(followers)}")


def parse_args():
    policy_dir = MUJOCO_DIR / "policies" / "microduck"
    go_policy = MUJOCO_DIR / "policies" / "go1" / "velocity" / "policy.onnx"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--camera-id", default="0")
    parser.add_argument("--start-angle-degrees", type=float, default=-70.0)
    parser.add_argument("--orbit-radius-cm", type=float, default=280.0)
    parser.add_argument("--leader-gap-cm", type=float, default=105.0)
    parser.add_argument("--follow-spacing-cm", type=float, default=72.0)
    parser.add_argument("--spacing-deadband-cm", type=float, default=8.0)
    parser.add_argument("--distance-gain", type=float, default=1.2)
    parser.add_argument("--heading-gain", type=float, default=1.8)
    parser.add_argument("--turn-in-place-degrees", type=float, default=58.0)
    parser.add_argument("--duck-cruise-vx", type=float, default=0.32)
    parser.add_argument("--duck-max-vx", type=float, default=0.4)
    parser.add_argument("--emergency-stop-ratio", type=float, default=0.62)
    parser.add_argument("--duck-max-yaw-rate", type=float, default=0.8)
    parser.add_argument("--heading-refresh-steps", type=int, default=5)
    parser.add_argument("--settle-trace-start-cm", type=float, default=75.0)
    parser.add_argument("--settle-trace-length-cm", type=float, default=5000.0)
    parser.add_argument("--go-root-height-cm", type=float, default=35.0)
    parser.add_argument("--set-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-distance-cm", type=float, default=650.0)
    parser.add_argument("--camera-side-cm", type=float, default=0.0)
    parser.add_argument("--camera-height-cm", type=float, default=420.0)
    parser.add_argument("--camera-pitch-degrees", type=float, default=-32.0)
    parser.add_argument("--camera-yaw-offset-degrees", type=float, default=0.0)
    parser.add_argument("--walking", default=str(policy_dir / "alpha_walking.onnx"))
    parser.add_argument("--standing", default=str(policy_dir / "alpha_stand.onnx"))
    # make_policy shares one argument contract across all Microduck demos even
    # though this formation intentionally loads only walking and standing.
    parser.add_argument("--sitstand", default=str(policy_dir / "alpha_sitstand.onnx"))
    parser.add_argument("--ground-pick", default=str(policy_dir / "alpha_ground_pick.onnx"))
    parser.add_argument("--kick-left", default=str(policy_dir / "ball_kick_left.onnx"))
    parser.add_argument("--kick-right", default=str(policy_dir / "ball_kick_right.onnx"))
    parser.add_argument("--roulade", default=str(policy_dir / "roulade.onnx"))
    parser.add_argument("--walking-action-scale", type=float, default=0.9)
    parser.add_argument("--standing-action-scale", type=float, default=1.0)
    parser.add_argument("--head-lowpass", type=float, default=0.5)
    parser.add_argument("--legs-lowpass", type=float, default=0.7)
    parser.add_argument("--go-policy", default=str(go_policy))
    parser.add_argument("--go-vx", type=float, default=0.30)
    parser.add_argument("--go-max-yaw-rate", type=float, default=0.35)
    parser.add_argument("--go-raw-action-clip", type=float, default=1.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--keep-actors", action="store_true")
    parser.add_argument("--allow-other-level", action="store_true")
    parser.add_argument(
        "--repair-scattered-ball-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-ball-cluster-span-cm", type=float, default=300.0)
    parser.add_argument("--max-ball-bounds-aspect", type=float, default=1.5)
    parser.add_argument("--ball-mass-kg", type=float, default=0.43)
    parser.add_argument("--ball-friction", type=float, default=0.58)
    parser.add_argument("--ball-rolling-friction", type=float, default=0.003)
    parser.add_argument("--ball-bounciness", type=float, default=0.38)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.duration < 0.0 or args.print_every <= 0:
        parser.error("--duration must be non-negative and --print-every positive")
    if args.orbit_radius_cm <= 0.0 or args.follow_spacing_cm <= 0.0:
        parser.error("orbit radius and follow spacing must be positive")
    if args.leader_gap_cm < args.follow_spacing_cm:
        parser.error("--leader-gap-cm must be at least --follow-spacing-cm")
    if not 0.0 <= args.duck_max_vx <= 0.4:
        parser.error("--duck-max-vx must be in [0, 0.4]")
    if not 0.0 <= args.duck_cruise_vx <= args.duck_max_vx:
        parser.error("--duck-cruise-vx must be in [0, --duck-max-vx]")
    if not 0.0 < args.emergency_stop_ratio < 1.0:
        parser.error("--emergency-stop-ratio must be in (0, 1)")
    if args.heading_refresh_steps <= 0:
        parser.error("--heading-refresh-steps must be positive")
    if args.go_raw_action_clip <= 0.0 or args.ball_mass_kg <= 0.0:
        parser.error("action clip and ball mass must be positive")
    return args


def main() -> int:
    args = parse_args()
    args.run_tag = f"{os.getpid()}_{time.time_ns() % 1_000_000_000:09d}"
    client = go1.connect_client(args.host, args.port)
    leader: showcase.GoLane | None = None
    followers: list[Follower] = []
    owned_actors: list[str] = []
    ball_poses: list[showcase.BallPose] = []
    exit_code = 0
    try:
        level = str(request(client, "vget /level/name", verbose=False))
        print(f"LEVEL|{level}|expected=DowntownWest")
        if (
            showcase.DOWNTOWNWEST_LEVEL_TOKEN not in level.casefold()
            and not args.allow_other_level
        ):
            raise RuntimeError(
                "Open DowntownWest in PIE before running this demo "
                "(or pass --allow-other-level for diagnostics)."
            )
        showcase.repair_scattered_ball_layout(client, args)
        balls = showcase.discover_balls(client, args)
        ball_poses = showcase.snapshot_ball_poses(client, balls)
        leader, followers = spawn_formation(client, balls, args)
        owned_actors.append(leader.actor)
        owned_actors.extend(follower.lane.actor for follower in followers)
        showcase.configure_camera(client, balls, args)
        run(client, leader, followers, args)
    except KeyboardInterrupt:
        print("INTERRUPTED|Ctrl+C")
        exit_code = 130
    finally:
        for follower in followers:
            try:
                request(
                    client,
                    f"vset /object/{follower.lane.actor}/mujoco_microduck/stop",
                    verbose=False,
                )
            except Exception as exc:
                print(f"STOP_WARNING|actor={follower.lane.actor}|error={exc}")
        if leader is not None:
            try:
                go1.stop_go1_simulation(client, leader.actor)
            except Exception as exc:
                print(f"STOP_WARNING|actor={leader.actor}|error={exc}")
        if not args.keep_actors:
            for actor in reversed(owned_actors):
                try:
                    request(client, f"vset /object/{actor}/destroy", verbose=False)
                except Exception as exc:
                    print(f"DESTROY_WARNING|actor={actor}|error={exc}")
        if ball_poses:
            try:
                showcase.restore_ball_poses(client, ball_poses)
            except Exception as exc:
                print(f"BALL_RESTORE_WARNING|error={exc}")
        client.disconnect()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
