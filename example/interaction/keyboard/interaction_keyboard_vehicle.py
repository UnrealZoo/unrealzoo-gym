import argparse
import time
import threading

import cv2
import gym
from pynput import keyboard

from gym_unrealcv.envs.wrappers import augmentation, configUE
import os
os.environ['UnrealEnv']='I:\\UnrealProject\\UnrealZoo_UE5_6_PKG'

# Optional init pose placeholders (Unreal world coordinates in cm).
# Set to [x, y, z] to enable, keep None to use default random spawn.
INIT_POSES = {
    "player": [-1423,-10,180],
    "car": [-90,94,100],
    "motorbike": [936,47,100],
}

key_state = {
    "i": False,
    "j": False,
    "k": False,
    "l": False,
    "up": False,
    "down": False,
    "h": False,   # enter/exit selected vehicle
    "1": False,   # control player
    "2": False,   # control car
    "3": False,   # control bike
    "esc": False,
}
ONE_SHOT_KEYS = {"h", "1", "2", "3", "esc"}
CONTROL_MODE_TO_ROLE = {"player": "player", "car": "car", "bike": "motorbike"}
_pending_events = set()
_event_lock = threading.Lock()


def on_press(key):
    try:
        if key.char in key_state:
            k = key.char.lower()
            if k in key_state:
                key_state[k] = True
                if k in ONE_SHOT_KEYS:
                    with _event_lock:
                        _pending_events.add(k)
    except AttributeError:
        if key == keyboard.Key.up:
            key_state["up"] = True
        elif key == keyboard.Key.down:
            key_state["down"] = True
        elif key == keyboard.Key.esc:
            key_state["esc"] = True
            with _event_lock:
                _pending_events.add("esc")


def on_release(key):
    try:
        k = key.char.lower()
        if k in key_state:
            key_state[k] = False
    except AttributeError:
        if key == keyboard.Key.up:
            key_state["up"] = False
        elif key == keyboard.Key.down:
            key_state["down"] = False
        elif key == keyboard.Key.esc:
            key_state["esc"] = False


def consume_event(key):
    with _event_lock:
        if key in _pending_events:
            _pending_events.remove(key)
            return True
    return False


def get_player_action(enter_exit=False):
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
    if enter_exit:
        # animation_action index 4 -> "enter_vehicle"
        anim = 4
    elif move[0] == 0.0 and move[1] == 0.0 and head == 0:
        # No keys: None so action_mapping skips this agent (not all-zero tuple).
        return None
    return (tuple(move), head, anim)


def get_vehicle_move():
    steer = 0.0
    throttle = 0.0
    if key_state["i"]:
        throttle = 1.0
    if key_state["k"]:
        throttle = -1
    if key_state["j"]:
        steer = -1.0
    if key_state["l"]:
        steer = 1.0
    if steer == 0.0 and throttle == 0.0:
        return None
    return [steer, throttle]


def apply_init_positions(env, players, slots):
    unrealcv = env.unwrapped.unrealcv
    for role, idx in slots.items():
        loc = INIT_POSES.get(role)
        if loc is None:
            continue
        unrealcv.set_obj_location(players[idx], list(loc))
        print(f"[InitPose] {role} -> {loc}")
        time.sleep(1)



def _format_action_for_hud(act):
    if act is None:
        return "None"
    if isinstance(act, (list, tuple)):
        parts = []
        for x in act:
            if isinstance(x, (list, tuple)):
                parts.append(f"({','.join(f'{float(v):.2f}' for v in x)})")
            else:
                try:
                    fv = float(x)
                    parts.append(str(int(fv)) if fv == int(fv) else f"{fv:.2f}")
                except (TypeError, ValueError):
                    parts.append(str(x))
        return "(" + ", ".join(parts) + ")"
    return str(act)


def render_hud(frame, control_mode, players, slots, actions):
    img = frame.copy()
    role = CONTROL_MODE_TO_ROLE.get(control_mode, control_mode)
    if role not in slots:
        cv2.putText(img, f"Control: {control_mode} (unavailable)", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return img
    selected_idx = slots[role]
    selected_name = players[selected_idx]
    cv2.putText(img, f"Control: {control_mode} ({selected_name})", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Switch:1-p 2-car 3-bike | H:enter/exit | Esc:quit", (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y = 80
    cv2.putText(img, "Actions -> env.step:", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    y += 22
    for i, name in enumerate(players):
        short = name if len(name) <= 28 else name[:25] + "..."
        line = f"  [{i}] {short}: {_format_action_for_hud(actions[i])}"
        if len(line) > 85:
            line = line[:82] + "..."
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)
        y += 18
    return img

def main():
    parser = argparse.ArgumentParser(description="Vehicle interaction demo: enter/exit, drive, switch mesh.")
    parser.add_argument("-e", "--env_id", nargs="?", default="UnrealNavigation-SuburbNeighborhood_Day-MixedColor-v0")
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(1080, 640))
    # Fully use the new template-based spawn logic.
    env.unwrapped.agents_category = ["player", "car", "motorbike"]
    env = augmentation.RandomPopulationWrapper(env, 3, 3, random_target=False)
    env.seed(args.seed)
    obs = env.reset()

    players = env.unwrapped.player_list
    slots = {}
    for idx, obj in enumerate(players):
        slots[env.unwrapped.agents[obj]["agent_type"]] = idx

    required = ("player", "car", "motorbike")
    if any(x not in slots for x in required):
        raise RuntimeError(f"spawn categories mismatch, got: {slots}")
    apply_init_positions(env, players, slots)

    player_idx = slots["player"]
    car_idx = slots["car"]
    bike_idx = slots["motorbike"]
    player_name = players[player_idx]
    car_name = players[car_idx]
    bike_name = players[bike_idx]
    control_mode = "player"

    print("=== Vehicle Interaction Demo ===")
    print("I/J/K/L move/steer, Up/Down look")
    print("1=player, 2=car, 3=bike")
    print("No keys -> action None for current role (not zeros)")
    print("H enter/exit selected vehicle")
    print("Spawn pipeline: template category -> set_population(3)")
    print(f"Init pose placeholders: {INIT_POSES}")
    print("Esc quit")
    # Re-enable vehicle physics before interactive loop so direct control/interaction works as expected.
    env.unwrapped.unrealcv.set_phy(car_name, 1)
    env.unwrapped.unrealcv.set_phy(bike_name, 1)
    print("[InitState] set car/bike phy=1 before loop")

    try:
        while True:
            if consume_event("esc"):
                break
            if consume_event("1"):
                control_mode = "player"
            if consume_event("2"):
                control_mode = "car"
            if consume_event("3"):
                control_mode = "bike"

            # Default to no-op control: None means action_mapping won't emit set_move for that agent.
            actions = [None for _ in players]
            if control_mode == "player":
                actions[player_idx] = get_player_action(enter_exit=consume_event("h"))
            elif control_mode == "car":
                vm = get_vehicle_move()
                if vm is not None:
                    actions[car_idx] = tuple(vm)
            elif control_mode == "bike":
                vm = get_vehicle_move()
                if vm is not None:
                    actions[bike_idx] = tuple(vm)

            obs, _, _, _ = env.step(actions)
            show = render_hud(obs[player_idx], control_mode, players, slots, actions)
            cv2.imshow("interaction_vehicle_obs", show)
            cv2.waitKey(1)
            time.sleep(0.005)
    finally:
        listener.stop()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
