#!/usr/bin/env python3
"""OpenAI Gym style demo for Go1, G1, and MicroDuck."""
import argparse
import atexit
import time

import gym_unrealcv
import numpy as np

from common.command_source import FixedCommand, KeyboardCommand
from common.pretrained_policy import PretrainedPolicy


ENV_IDS = {
    "go1": "UnrealMujoco-Go1-v0",
    "g1": "UnrealMujoco-G1-v0",
    "microduck": "UnrealMujoco-MicroDuck-v0",
}

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=tuple(ENV_IDS))
    parser.add_argument(
        "mode", choices=("keyboard", "checkpoint"), default="keyboard", nargs="?"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--actor", default="")
    parser.add_argument("--spawn-x", type=float, default=None)
    parser.add_argument("--spawn-y", type=float, default=None)
    parser.add_argument("--spawn-z", type=float, default=None)
    parser.add_argument("--spawn-camera-id", default="0")
    parser.add_argument("--spawn-yaw-offset", type=float, default=0.0)
    parser.add_argument("--keep-actor", action="store_true")
    parser.add_argument("--policy", default="")
    parser.add_argument("--walking", default="")
    parser.add_argument("--standing", default="")
    parser.add_argument("--command-vx", type=float, default=None)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--vx", type=float, default=None)
    parser.add_argument("--yaw-rate", type=float, default=None)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args(argv)


def robot_defaults(robot):
    velocity = 0.4 if robot == "microduck" else 0.5
    yaw_rate = 0.2 if robot == "g1" else 0.8
    return velocity, yaw_rate


def main(argv=None):
    args = parse_args(argv)
    default_velocity, default_yaw_rate = robot_defaults(args.robot)
    velocity = default_velocity if args.vx is None else args.vx
    yaw_rate = default_yaw_rate if args.yaw_rate is None else args.yaw_rate
    command_vx = (
        default_velocity if args.command_vx is None else args.command_vx
    )
    spawn_location = (
        None
        if args.spawn_x is None
        else (args.spawn_x, args.spawn_y, args.spawn_z)
    )

    env = gym_unrealcv.make_mujoco_env(
        args.robot,
        host=args.host,
        port=args.port,
        actor_name=args.actor,
        spawn_location=spawn_location,
        spawn_camera_id=args.spawn_camera_id,
        spawn_yaw_offset=args.spawn_yaw_offset,
        keep_actor=args.keep_actor,
    )
    atexit.register(env.close)

    policy = PretrainedPolicy(
        args.robot,
        policy_path=args.policy,
        walking_path=args.walking,
        standing_path=args.standing,
    )
    fixed_command = [command_vx, args.command_vy, args.command_yaw]
    command_source = (
        KeyboardCommand(velocity, yaw_rate)
        if args.mode == "keyboard"
        else FixedCommand(fixed_command)
    )

    command = (
        np.zeros(3, dtype=np.float32)
        if args.mode == "keyboard"
        else np.asarray(fixed_command, dtype=np.float32)
    )
    exit_requested = False
    env.unwrapped.set_command(command)
    observation = env.reset()
    policy.reset()

    start_time = time.perf_counter()
    next_step = start_time
    step = 0
    done = False
    print(
        "GYM|env={}|robot={}|mode={}|observation={}|action={}".format(
            ENV_IDS[args.robot],
            args.robot,
            args.mode,
            env.observation_space.shape,
            env.action_space.shape,
        )
    )
    if args.mode == "keyboard":
        print("KEYBOARD|I/K=forward/back|J/L=turn|Space=stop|X/Esc=exit")

    with command_source:
        while not done and not exit_requested:
            elapsed = time.perf_counter() - start_time
            if args.duration > 0.0 and elapsed >= args.duration:
                break

            command, exit_requested = command_source.read()
            active_command = (
                np.zeros(3, dtype=np.float32)
                if elapsed < args.warmup
                else command
            )
            env.unwrapped.set_command(active_command)

            action = policy.act(observation, active_command)
            observation, reward, done, info = env.step(action)

            if step % args.print_every == 0:
                print(
                    "STEP|{}|reward={:.3f}|command={}|actor={}".format(
                        step,
                        reward,
                        np.round(active_command, 3).tolist(),
                        info["actor"],
                    )
                )

            step += 1
            next_step += 1.0 / args.hz
            delay = next_step - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)

    env.close()
    print("DONE|steps={}".format(step))


if __name__ == "__main__":
    main()
