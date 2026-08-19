#!/usr/bin/env python3
"""Cycle drone meshes inside a registered gym_unrealcv Navigation episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from gym_navigation_demo import (  # noqa: E402
    add_navigation_arguments,
    make_navigation_env,
    navigation_asset_path,
    navigation_client,
    validate_navigation_arguments,
)


DRONE_BLUEPRINT = "/Game/Drone_Pack/Drone_Bp/BP_Drone_customized.BP_Drone_customized"
APPEARANCES = ("spy", "fpv", "police", "template", "baba", "delivery")


def request_text(client, command: str) -> str:
    response = client.request(command)
    text = str(response).strip()
    if not text or text.casefold().startswith("error"):
        raise RuntimeError(f"{command} failed: {response!r}")
    return text


def supports_set_app(client, actor: str) -> bool:
    try:
        payload = json.loads(request_text(client, f"vreflect {actor} functions"))
    except Exception:
        return False
    return "set_app" in json.dumps(payload).casefold()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_navigation_arguments(parser)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--order", default="0,1,2,3,4,5")
    parser.add_argument("--render", action="store_true", help="Show the Navigation observation with OpenCV")
    parser.add_argument("--no-reset", action="store_true", help="Do not restore spy appearance on exit")
    args = parser.parse_args()
    validate_navigation_arguments(parser, args)
    return args


def run(args: argparse.Namespace) -> int:
    order = [int(value) for value in args.order.split(",") if value.strip()]
    if not order or any(value < 0 or value >= len(APPEARANCES) for value in order):
        raise ValueError("--order must contain comma-separated indices in the range 0..5")
    if args.interval <= 0 or args.cycles <= 0:
        raise ValueError("--interval and --cycles must be positive")

    def configure_drone_task(base_env) -> None:
        template = dict(base_env.agent_templates["drone"])
        drone_path = navigation_asset_path(base_env, "drone", DRONE_BLUEPRINT)
        template["asset_path"] = [drone_path]
        blueprint_name = drone_path.rsplit("/", 1)[-1].split(".", 1)[0]
        template["class_name"] = [f"{blueprint_name.removesuffix('_C')}_C"]
        template["agent_type"] = "drone"
        base_env.agent_templates["drone"] = template

    env = make_navigation_env(
        args,
        agent_category="drone",
        population=1,
        configure_task=configure_drone_task,
    )
    client = None
    actor = None
    try:
        observations = env.reset()
        client = navigation_client(env)
        actor = env.unwrapped.player_list[0]
        if not supports_set_app(client, actor):
            raise RuntimeError(
                f"{actor!r} does not expose set_app; use BP_Drone_customized built on "
                "DroneAppearancePawnBase"
            )

        print(f"env={args.env_id} actor={actor} camera={env.unwrapped.cam_list[0]}")
        transition = 0
        total_transitions = args.cycles * len(order)
        next_switch = 0.0
        while True:
            now = time.monotonic()
            if transition < total_transitions and now >= next_switch:
                appearance_index = order[transition % len(order)]
                name = APPEARANCES[appearance_index]
                response = request_text(client, f"vbp {actor} set_app {appearance_index}")
                print(
                    f"cycle={transition // len(order) + 1}/{args.cycles} index={appearance_index} "
                    f"appearance={name} response={response}"
                )
                transition += 1
                next_switch = now + args.interval
            observations, rewards, done, info = env.step([None])
            if args.render:
                import cv2

                cv2.imshow("Gym Navigation drone mesh switch", observations[0])
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if transition >= total_transitions:
                break
            if done:
                observations = env.reset()
                client = navigation_client(env)
                actor = env.unwrapped.player_list[0]
            time.sleep(0.01)
    finally:
        if client is not None and client.isconnected():
            if actor and not args.no_reset:
                try:
                    client.request(f"vbp {actor} set_app 0")
                except Exception:
                    pass
        env.close()
        if args.render:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
    return 0


def main() -> int:
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
