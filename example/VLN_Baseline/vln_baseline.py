import argparse
import os
import time
from collections import deque

import cv2
import gym
import numpy as np

import gym_unrealcv  # noqa: F401 - importing registers UnrealZoo gym environments.
from agents import add_agent_cli_args, available_methods, build_vln_agent
from gym_unrealcv.envs.wrappers import augmentation, configUE, early_done, monitor, time_dilation


DEFAULT_ENV_ID = "UnrealNavigation-SuburbNeighborhood_Day-MixedColor-v0"
DEFAULT_INSTRUCTION = "Navigate to the gray car and stop near it."


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a VLN baseline agent in an UnrealZoo navigation environment."
    )
    parser.add_argument(
        "--method",
        default="uni_navid",
        choices=available_methods(),
        help="VLN method to run.",
    )
    parser.add_argument("-e", "--env-id", default=DEFAULT_ENV_ID, help="UnrealZoo gym environment id.")
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Natural-language navigation instruction passed to the VLN agent.",
    )
    parser.add_argument(
        "--target",
        default="BP_Car_Gray",
        help="Optional Unreal object target name for Navigation reward/success. Use '' to skip.",
    )
    parser.add_argument(
        "--unreal-env",
        default=None,
        help="Optional path assigned to the UnrealEnv environment variable before env launch.",
    )
    parser.add_argument("--resolution", nargs=2, type=int, default=(240, 240), metavar=("W", "H"))
    parser.add_argument("--seed", type=int, default=10, help="Random seed passed to the gym env.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum env steps for one episode.")
    parser.add_argument("--forward-speed", type=float, default=100.0, help="UnrealZoo forward velocity.")
    parser.add_argument("--turn-speed", type=float, default=30.0, help="UnrealZoo yaw velocity.")
    parser.add_argument(
        "--stop-ends-episode",
        action="store_true",
        default=True,
        help="End the episode after executing a VLN stop action.",
    )
    parser.add_argument(
        "--no-stop-ends-episode",
        dest="stop_ends_episode",
        action="store_false",
        help="Keep stepping after a VLN stop action until the env returns done.",
    )
    parser.add_argument("-r", "--render", action="store_true", help="Show first-person observation.")
    parser.add_argument("-m", "--monitor", action="store_true", help="Wrap env with DisplayWrapper.")
    parser.add_argument("-t", "--time-dilation", type=int, default=-1)
    parser.add_argument("-d", "--early-done", type=int, default=-1)
    parser.add_argument("--offscreen", action="store_true", help="Launch UE with offscreen rendering.")
    parser.add_argument("--display", default=None, help="Display id passed to RunUnreal.")
    add_agent_cli_args(parser)
    return parser


def make_env(args):
    if args.unreal_env:
        os.environ["UnrealEnv"] = os.path.abspath(args.unreal_env)

    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(
        env,
        offscreen=args.offscreen,
        resolution=tuple(args.resolution),
        display=args.display,
    )
    env.unwrapped.agents_category = ["player"]

    if args.target:
        env.unwrapped.set_navigation_targets([args.target])

    if args.time_dilation > 0:
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    if args.early_done > 0:
        env = early_done.EarlyDoneWrapper(env, args.early_done)
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    env = augmentation.RandomPopulationWrapper(env, 1, 1, random_target=False)
    return env


def vln_action_to_unreal(action, forward_speed, turn_speed):
    """Map a VLN text action to UnrealZoo's player Mixed action tuple."""
    action = action.lower()
    if action == "forward":
        move = (0.0, float(forward_speed))
    elif action == "left":
        move = (-float(turn_speed), 0.0)
    elif action == "right":
        move = (float(turn_speed), 0.0)
    elif action == "stop":
        move = (0.0, 0.0)
    else:
        raise ValueError(f"Unsupported VLN action: {action!r}")
    return (move, 0, 0)


def first_observation(obs):
    if isinstance(obs, (list, tuple)):
        return obs[0]
    obs_arr = np.asarray(obs)
    if obs_arr.ndim == 4:
        return obs_arr[0]
    return obs


def render_observation(obs):
    img = first_observation(obs)
    if img is None:
        return
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    cv2.imshow("vln_baseline_obs", img)
    cv2.waitKey(1)


def run_episode(args):
    vln_agent = build_vln_agent(args.method, args)
    env = make_env(args)
    pending_actions = deque()
    step_count = 0
    total_reward = 0.0
    info = {}

    try:
        env.seed(args.seed)
        obs = env.reset()
        vln_agent.reset()
        start_time = time.time()

        while step_count < args.max_steps:
            if not pending_actions:
                new_actions = vln_agent.predict(first_observation(obs), args.instruction, info=info)
                pending_actions.extend(new_actions)
                print(f"VLN predicted actions: {list(pending_actions)}")

            vln_action = pending_actions.popleft()
            unreal_action = vln_action_to_unreal(vln_action, args.forward_speed, args.turn_speed)
            obs, reward, done, info = env.step([unreal_action])
            step_count += 1
            total_reward += float(np.asarray(reward).sum())

            success = bool(info.get("Success", False))
            print(
                f"step={step_count} action={vln_action} reward={reward} "
                f"done={done} success={success}"
            )

            if args.render:
                render_observation(obs)

            if vln_action == "stop" and args.stop_ends_episode:
                print("VLN emitted stop; ending episode.")
                break
            if done:
                break

        fps = step_count / max(time.time() - start_time, 1e-6)
        print(
            f"Episode finished: steps={step_count}, total_reward={total_reward:.3f}, "
            f"success={bool(info.get('Success', False))}, fps={fps:.2f}"
        )
    finally:
        env.close()
        if args.render:
            cv2.destroyAllWindows()


def main():
    args = build_parser().parse_args()
    run_episode(args)


if __name__ == "__main__":
    main()
