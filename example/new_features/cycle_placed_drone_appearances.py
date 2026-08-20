#!/usr/bin/env python3
"""Cycle appearances on BP_Drone_customized actors already placed in PIE.

Start PIE with UnrealCV enabled before running this script.  The script does
not launch a Gym environment or spawn actors; it connects directly to the
existing UnrealCV server and updates every matching placed actor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_APPEARANCES = (1, 2, 3, 4, 5)
DEFAULT_NAME_PATTERN = "bp_drone_customized"
APPEARANCE_SCALES = {
    0: 0.2,
    1: 1.0,
    2: 0.2,
    3: 1.0,
    4: 1.0,
    5: 1.0,
}


def request_text(client, command: str) -> str:
    """Run an UnrealCV command and turn protocol errors into exceptions."""
    response = client.request(command)
    text = str(response).strip()
    if not text or text.casefold().startswith("error"):
        raise RuntimeError(f"{command!r} failed: {response!r}")
    return text


def connect(host: str, port: int):
    try:
        import unrealcv
    except ImportError as exc:
        raise RuntimeError(
            "the unrealcv Python package is not installed; install the project "
            "dependencies first (for example: pip install -e .)"
        ) from exc

    client = unrealcv.Client((host, port))
    client.connect()
    if not client.isconnected():
        raise RuntimeError(
            f"cannot connect to UnrealCV at {host}:{port}; "
            "make sure the editor is in PIE and UnrealCV is listening"
        )
    return client


def exposes_set_app(client, actor: str) -> bool:
    """Confirm that the matched actor is the expected customized drone."""
    try:
        functions = json.loads(request_text(client, f"vreflect {actor} functions"))
    except (json.JSONDecodeError, RuntimeError, TypeError):
        return False
    return "set_app" in json.dumps(functions).casefold()


def find_drones(client, name_pattern: str) -> list[str]:
    objects = request_text(client, "vget /objects").split()
    candidates = [name for name in objects if name_pattern in name.casefold()]
    return [name for name in candidates if exposes_set_app(client, name)]


def parse_appearances(value: str) -> tuple[int, ...]:
    try:
        appearances = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("appearance IDs must be comma-separated integers") from exc
    if not appearances:
        raise argparse.ArgumentTypeError("at least one appearance ID is required")
    return appearances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="UnrealCV server address")
    parser.add_argument("--port", type=int, default=9000, help="UnrealCV server port")
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="seconds to keep each appearance (default: 3)",
    )
    parser.add_argument(
        "--appearances",
        type=parse_appearances,
        default=DEFAULT_APPEARANCES,
        metavar="IDS",
        help="comma-separated set_app IDs (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="number of full cycles; 0 means keep cycling (default: 0)",
    )
    parser.add_argument(
        "--actor",
        action="append",
        default=[],
        help="explicit actor name; repeat to target multiple actors and skip auto-detection",
    )
    parser.add_argument(
        "--name-pattern",
        default=DEFAULT_NAME_PATTERN,
        help=f"case-insensitive actor-name pattern (default: {DEFAULT_NAME_PATTERN})",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.cycles < 0:
        parser.error("--cycles must be zero or greater")
    unsupported = sorted(set(args.appearances) - APPEARANCE_SCALES.keys())
    if unsupported:
        parser.error(
            "--appearances contains IDs without a configured scale: "
            + ", ".join(map(str, unsupported))
        )
    if not args.name_pattern.strip() and not args.actor:
        parser.error("--name-pattern cannot be empty")
    args.name_pattern = args.name_pattern.casefold()
    return args


def run(args: argparse.Namespace) -> int:
    client = connect(args.host, args.port)
    try:
        actors = list(dict.fromkeys(args.actor)) if args.actor else find_drones(
            client, args.name_pattern
        )
        if not actors:
            raise RuntimeError(
                "no BP_Drone_customized instance exposing set_app was found in PIE"
            )

        invalid = [actor for actor in actors if not exposes_set_app(client, actor)]
        if invalid:
            raise RuntimeError(f"actors do not expose set_app: {', '.join(invalid)}")

        cycle_label = "infinite" if args.cycles == 0 else str(args.cycles)
        print(
            f"connected={args.host}:{args.port} actors={','.join(actors)} "
            f"appearances={args.appearances} interval={args.interval:g}s cycles={cycle_label}"
        )

        cycle = 0
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            for appearance_id in args.appearances:
                for actor in actors:
                    app_response = request_text(
                        client, f"vbp {actor} set_app {appearance_id}"
                    )
                    scale = APPEARANCE_SCALES[appearance_id]
                    scale_text = f"{scale:g} {scale:g} {scale:g}"
                    scale_response = request_text(
                        client, f"vset /object/{actor}/scale {scale_text}"
                    )
                    print(
                        f"cycle={cycle} actor={actor} set_app={appearance_id} "
                        f"app_response={app_response} scale=({scale_text}) "
                        f"scale_response={scale_response}"
                    )
                # Start the interval after every target has switched, so each
                # appearance remains visible for at least the requested time.
                time.sleep(args.interval)
    finally:
        if client.isconnected():
            client.disconnect()
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("stopped by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
