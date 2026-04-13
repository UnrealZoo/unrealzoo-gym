import argparse
import random
import time

import cv2
import gym
import gym_unrealcv  # noqa: F401
import numpy as np
from gym_unrealcv.envs.tracking.baseline import PoseTracker, DronePoseTracker
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, augmentation, configUE
from pynput import keyboard
import os
os.environ['UnrealEnv']='/media/wuk/T9/UnrealEnv/'

def pick_reachable_goal(env, leader_obj, max_trials=12):
    """Sample a reachable goal for leader by probing nav path."""
    candidates = list(env.unwrapped.safe_start)
    random.shuffle(candidates)
    for goal in candidates[:max_trials]:
        path = env.unwrapped.unrealcv.find_path(leader_obj, goal)
        if path and len(path) > 1:
            return goal, path
    fallback = random.choice(candidates)
    return fallback, []


key_state = {
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    "e": False,
    "q": False,
    "j": False,
    "l": False,
}


def on_press(key):
    try:
        if key.char in key_state:
            key_state[key.char] = True
    except AttributeError:
        pass


def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        pass


def get_drone_keyboard_action():
    # Drone action format: [vx, vy, vz, vyaw]
    action = [0.0, 0.0, 0.0, 0.0]
    if key_state["w"]:
        action[0] = 1.0
    if key_state["s"]:
        action[0] = -1.0
    if key_state["a"]:
        action[1] = -1.0
    if key_state["d"]:
        action[1] = 1.0
    if key_state["e"]:
        action[2] = 1.0
    if key_state["q"]:
        action[2] = -1.0
    if key_state["j"]:
        action[3] = -1.0
    if key_state["l"]:
        action[3] = 1.0
    is_override = any(key_state.values())
    return action, is_override


def _to_bgr_uint8(frame):
    """Ensure frame is uint8 BGR-like image for cv2 visualization."""
    img = np.asarray(frame)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def build_mosaic(obs, leader_id, follower_1, follower_2, drone_id):
    f0 = _to_bgr_uint8(obs[leader_id]).copy()
    f1 = _to_bgr_uint8(obs[follower_1]).copy()
    f2 = _to_bgr_uint8(obs[follower_2]).copy()
    f3 = _to_bgr_uint8(obs[drone_id]).copy()

    cv2.putText(f0, "Leader (0)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(f1, "Follower (1)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(f2, "Follower (2)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(f3, "Drone (3)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    top = cv2.hconcat([f0, f1])
    bottom = cv2.hconcat([f2, f3])
    return cv2.vconcat([top, bottom])


def setup_episode_formation(env, players, leader_id, follower_1, follower_2, drone_id):
    """Apply deterministic spawn/layout + skins for one episode."""
    # Manually reset to a stable formation start (avoid random bad spawn overlaps).
    env.unwrapped.unrealcv.set_obj_location(players[leader_id], (16054, -11732, -12680))
    env.unwrapped.unrealcv.set_obj_location(players[follower_1], (17474, -11755, -12280))
    env.unwrapped.unrealcv.set_obj_location(players[follower_2], (15808, -12109, -12680))
    env.unwrapped.unrealcv.set_obj_location(players[drone_id], (15208, -12109, -12680))

    # Ground agents (0,1,2) use robot dog skins: set_app id range [20, 33].
    for idx in (leader_id, follower_1, follower_2):
        app_id = int(np.random.randint(20, 34))
        env.unwrapped.unrealcv.set_appearance(players[idx], app_id)

    # Lift drone spawn height by +2m (Unreal uses centimeters -> +200).
    drone_loc = env.unwrapped.unrealcv.get_obj_location(players[drone_id])
    env.unwrapped.unrealcv.set_obj_location(
        players[drone_id],
        [drone_loc[0], drone_loc[1], drone_loc[2] + 200.0],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial-ground cooperative formation demo.")
    parser.add_argument("-e", "--env_id", nargs="?", default="UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0",
                        help="Environment id")
    parser.add_argument("-r", "--render", dest="render", action="store_true", help="Show cv2 windows")
    parser.add_argument("-s", "--seed", dest="seed", default=0, help="Random seed")
    parser.add_argument("-t", "--time-dilation", dest="time_dilation", default=10, help="Simulator time dilation")
    parser.add_argument("-d", "--early-done", dest="early_done", default=-1, help="Early done lost steps")
    parser.add_argument("-m", "--monitor", dest="monitor", action="store_true", help="Enable monitor wrapper")
    parser.add_argument("--episodes", dest="episodes", default=30, type=int, help="Number of episodes")
    args = parser.parse_args()

    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(640, 480))
    # Mixed category population: 3 ground agents + 1 drone.
    env.unwrapped.agents_category = ["player", "player", "player", "drone"]

    if int(args.time_dilation) > 0:
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    env = augmentation.RandomPopulationWrapper(env, 4, 4, random_target=False)
    env.seed(int(args.seed))

    # Formation indices:
    # 0 leader(navmesh), 1->follow 0, 2->follow 1, 3(drone)->follow 2.
    leader_id = 0
    follower_1 = 1
    follower_2 = 2
    drone_id = 3

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("Drone keyboard override enabled: W/S A/D E/Q J/L.")

    try:
        for eps in range(args.episodes):
            obs = env.reset()
            if len(env.unwrapped.player_list) < 4:
                raise RuntimeError(f"Need 4 agents, got {len(env.unwrapped.player_list)}")

            players = env.unwrapped.player_list
            setup_episode_formation(env, players, leader_id, follower_1, follower_2, drone_id)
            exp_distance = 400

            ground_tracker_1 = PoseTracker(env.action_space[follower_1], exp_distance)
            ground_tracker_2 = PoseTracker(env.action_space[follower_2], exp_distance)
            # Replace old keyboard slot with a drone-specific pose tracker.
            drone_tracker = DronePoseTracker(expected_distance=exp_distance)

            leader_goal, path = pick_reachable_goal(env, players[leader_id])
            env.unwrapped.unrealcv.nav_to_goal(players[leader_id], leader_goal)

            print(f"[Episode {eps}] leader_goal={leader_goal}, path_points={len(path)}")
            t0 = time.time()
            step_count = 0
            cum_rewards = np.zeros(len(players))

            while True:
                obj_poses = env.unwrapped.obj_poses
                actions = [None] * len(players)
                actions[follower_1] = ground_tracker_1.act(obj_poses[follower_1], obj_poses[leader_id])
                actions[follower_2] = ground_tracker_2.act(obj_poses[follower_2], obj_poses[follower_1])
                drone_action = drone_tracker.act(obj_poses[drone_id], obj_poses[follower_2])
                manual_action, override = get_drone_keyboard_action()
                # Keyboard control overrides DronePoseTracker while any control key is held.
                actions[drone_id] = manual_action if override else drone_action

                obs, rewards, done, info = env.step(actions)
                cum_rewards += rewards
                step_count += 1

                mosaic = build_mosaic(obs, leader_id, follower_1, follower_2, drone_id)
                cv2.imshow("Aerial-Ground Cooperative (2x2)", mosaic)
                cv2.waitKey(1)

                if done:
                    fps = step_count / max(1e-6, time.time() - t0)
                    print(f"[Episode {eps}] done, fps={fps:.2f}, rewards={cum_rewards}")
                    break
    finally:
        listener.stop()
        env.close()
