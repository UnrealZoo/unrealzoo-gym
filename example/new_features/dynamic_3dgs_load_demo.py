#!/usr/bin/env python3
"""Load cooked NanoGS assets while a gym_unrealcv Navigation episode runs.

PLY import remains an editor/cook operation.  This demo switches already
imported ``GaussianSplatAsset`` objects at runtime through UnrealCV reflection.
"""

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
    navigation_client,
    validate_navigation_arguments,
)


DEFAULT_ACTOR = "UnrealZooDynamic3DGS"


def request_text(client, command: str) -> str:
    response = client.request(command)
    text = str(response).strip()
    if not text or text.casefold().startswith("error"):
        raise RuntimeError(f"{command} failed: {response!r}")
    return text


def asset_reference(path: str) -> str:
    path = path.strip()
    if not path.startswith("/Game/"):
        raise ValueError(f"3DGS asset must be a /Game/... object path, got {path!r}")
    if "." not in path.rsplit("/", 1)[-1]:
        name = path.rsplit("/", 1)[-1]
        path = f"{path}.{name}"
    return f"GaussianSplatAsset'{path}'"


def call_component(client, component: str, function: str, arguments: dict) -> dict:
    payload = json.dumps(arguments, separators=(",", ":"))
    response = request_text(client, f"vreflect {component} call_json {function} {payload}")
    return json.loads(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_navigation_arguments(parser)
    parser.add_argument("--asset", action="append", required=True, help="Imported /Game/... GaussianSplatAsset")
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--component", default="GaussianSplatComponent")
    parser.add_argument("--location", nargs=3, type=float, metavar=("X", "Y", "Z"), default=(0.0, 0.0, 0.0))
    parser.add_argument("--rotation", nargs=3, type=float, metavar=("PITCH", "YAW", "ROLL"), default=(0.0, 0.0, 0.0))
    parser.add_argument("--scale", nargs=3, type=float, metavar=("X", "Y", "Z"), default=(1.0, 1.0, 1.0))
    parser.add_argument("--interval", type=float, default=4.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--keep-actor", action="store_true")
    parser.add_argument("--unload-between", action="store_true")
    args = parser.parse_args()
    validate_navigation_arguments(parser, args)
    if args.interval <= 0 or args.cycles <= 0:
        parser.error("--interval and --cycles must be positive")
    return args


def run(args: argparse.Namespace) -> int:
    assets = [asset_reference(value) for value in args.asset]

    env = make_navigation_env(args, agent_category="player", population=1)
    client = None
    spawned = False
    actor = args.actor
    try:
        observations = env.reset()
        client = navigation_client(env)
        objects = request_text(client, "vget /objects")
        if args.actor not in objects:
            location = " ".join(f"{value:.6f}" for value in args.location)
            actor = request_text(
                client,
                f"vset /objects/spawn GaussianSplatActor {args.actor} {location}",
            )
            spawned = True
        else:
            actor = args.actor
        request_text(client, f"vset /object/{actor}/rotation " + " ".join(map(str, args.rotation)))
        request_text(client, f"vset /object/{actor}/scale " + " ".join(map(str, args.scale)))

        # SetSplatAsset is used instead of assigning SplatAsset directly because
        # the component must invalidate bounds and recreate its scene proxy.
        cycle = 0
        asset_index = 0
        next_switch = 0.0
        while True:
            if cycle >= args.cycles:
                break
            now = time.monotonic()
            if now >= next_switch:
                reference = assets[asset_index]
                result = call_component(
                    client,
                    args.component,
                    "SetSplatAsset",
                    {"NewAsset": reference},
                )
                count = call_component(client, args.component, "GetSplatCount", {})
                print(
                    f"cycle={cycle + 1}/{args.cycles} actor={actor} asset={reference} "
                    f"set_result={result} splat_count={count}"
                )
                if args.unload_between:
                    call_component(client, args.component, "SetSplatAsset", {"NewAsset": None})
                asset_index += 1
                if asset_index == len(assets):
                    asset_index = 0
                    cycle += 1
                next_switch = now + args.interval

            observations, rewards, done, info = env.step([None])
            if done:
                observations = env.reset()
                client = navigation_client(env)
            time.sleep(0.01)
    finally:
        if client is not None and client.isconnected():
            if spawned and not args.keep_actor:
                try:
                    client.request(f"vset /object/{actor}/destroy")
                except Exception:
                    pass
        env.close()
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
