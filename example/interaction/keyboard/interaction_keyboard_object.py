import argparse
import time

import cv2
import gym
from pynput import keyboard

from gym_unrealcv.envs.wrappers import augmentation, configUE
import os
os.environ['UnrealEnv']='E:\\UnrealEnv'


# Optional init pose placeholders (Unreal world coordinates in cm).
# Set to [x, y, z] to enable, keep None to use defaults.
PLAYER_INIT_POSE = [-17,-112,100]
PICKABLE_INIT_POSE = [-90,94,100]


key_state = {
    "i": False,
    "j": False,
    "k": False,
    "l": False,
    "up": False,
    "down": False,
    "e": False,   # pickup
    "q": False,   # drop
    "m": False,   # switch object mesh
    "esc": False,
}


def on_press(key):
    try:
        if key.char in key_state:
            key_state[key.char] = True
    except AttributeError:
        if key == keyboard.Key.up:
            key_state["up"] = True
        elif key == keyboard.Key.down:
            key_state["down"] = True
        elif key == keyboard.Key.esc:
            key_state["esc"] = True


def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        if key == keyboard.Key.up:
            key_state["up"] = False
        elif key == keyboard.Key.down:
            key_state["down"] = False
        elif key == keyboard.Key.esc:
            key_state["esc"] = False


def edge_trigger(key, prev_state):
    return key_state[key] and (not prev_state.get(key, False))


def get_player_action():
    move = [0.0, 0.0]
    head = 0
    anim = 0
    if key_state["i"]:
        move[1] = 100
    if key_state["k"]:
        move[1] = -100
    if key_state["j"]:
        move[0] = -30
    if key_state["l"]:
        move[0] = 30
    if key_state["up"]:
        head = 1
    if key_state["down"]:
        head = 2
    return (tuple(move), head, anim)


def safe_set_mesh(unrealcv, obj_name, mesh_token):
    try:
        return unrealcv.client.request(f"vbp {obj_name} set_app {mesh_token}", -1)
    except Exception as err:
        return f"set_app failed: {err}"


def spawn_pickable(env, player_name):
    unrealcv = env.unwrapped.unrealcv
    base_pos = unrealcv.get_obj_location(player_name)
    cfg = getattr(env.unwrapped, "env_configs", {})
    if not isinstance(cfg, dict):
        cfg = {}
    pickable_cfg = cfg.get("Pickable_object", {})
    if not isinstance(pickable_cfg, dict) or "class_name" not in pickable_cfg:
        # Compatibility fallback for full setting dict style.
        env_cfg = cfg.get("env", {}) if isinstance(cfg.get("env", {}), dict) else {}
        pickable_cfg = env_cfg.get("Pickable_object", {}) if isinstance(env_cfg, dict) else {}
    pickable_class = pickable_cfg.get("class_name", "BP_GrabMoveDrop_C")
    obj_name = "demo_pickable_ep0"
    if PICKABLE_INIT_POSE is None:
        obj_pos = [base_pos[0] + 120, base_pos[1], base_pos[2] + 20]
    else:
        obj_pos = list(PICKABLE_INIT_POSE)
    unrealcv.new_obj(pickable_class, obj_name, obj_pos)
    unrealcv.set_phy(obj_name, 1)
    print(f"[InitPose] pickable -> {obj_pos}")
    return obj_name


def main():
    parser = argparse.ArgumentParser(description="Object interaction demo: pickup/drop and switch object mesh.")
    parser.add_argument("-e", "--env_id", nargs="?", default="UnrealNavigation-SuburbNeighborhood_Day-MixedColor-v0")
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(480, 360))
    env.unwrapped.agents_category = ["player"]
    env = augmentation.RandomPopulationWrapper(env, 1, 1, random_target=False)
    env.seed(args.seed)
    obs = env.reset()

    player_name = env.unwrapped.player_list[0]
    if PLAYER_INIT_POSE is not None:
        env.unwrapped.unrealcv.set_obj_location(player_name, list(PLAYER_INIT_POSE))
        print(f"[InitPose] player -> {PLAYER_INIT_POSE}")
    pickable_name = spawn_pickable(env, player_name)
    object_mesh_index = 0
    prev = {k: False for k in key_state.keys()}

    print("=== Object Interaction Demo ===")
    print("I/J/K/L move, Up/Down look")
    print("E pickup, Q drop, M switch object mesh")
    print(f"Init pose placeholders: player={PLAYER_INIT_POSE}, pickable={PICKABLE_INIT_POSE}")
    print("Esc quit")

    try:
        while True:
            if edge_trigger("esc", prev):
                break

            if edge_trigger("e", prev):
                env.unwrapped.unrealcv.set_pickup(player_name)
                print("[Object] pickup")

            if edge_trigger("q", prev):
                env.unwrapped.unrealcv.drop_body(player_name)
                print("[Object] drop")

            if edge_trigger("m", prev):
                object_mesh_index += 1
                resp = safe_set_mesh(env.unwrapped.unrealcv, pickable_name, object_mesh_index)
                print(f"[ObjectMesh] index {object_mesh_index} -> {resp}")

            obs, _, _, _ = env.step([get_player_action()])
            cv2.imshow("interaction_object_obs", obs[0])
            cv2.waitKey(1)

            for k in prev.keys():
                prev[k] = key_state[k]
            time.sleep(0.02)
    finally:
        listener.stop()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
