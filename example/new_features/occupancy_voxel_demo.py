#!/usr/bin/env python3
"""Continuously visualize occupancy from a gym_unrealcv Navigation episode."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from gym_navigation_demo import (  # noqa: E402
    add_navigation_arguments,
    make_navigation_env,
    navigation_camera_id,
    navigation_client,
    validate_navigation_arguments,
)


def request_ok(client, command: str):
    response = client.request(command)
    if response is None or (
        isinstance(response, str) and response.casefold().startswith("error")
    ):
        raise RuntimeError(f"{command} failed: {response!r}")
    return response


def vector3(response, label: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(value) for value in str(response).split())
    except ValueError as exc:
        raise RuntimeError(f"Invalid {label}: {response!r}") from exc
    if len(values) != 3:
        raise RuntimeError(f"Invalid {label}: {response!r}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_navigation_arguments(parser)
    parser.add_argument("--camera-id", default="auto", help="auto uses the Navigation agent camera")
    parser.add_argument("--profile", choices=("lingo_vis", "lingo_train"), default="lingo_vis")
    parser.add_argument("--method", choices=("bounds", "mesh"), default="mesh")
    parser.add_argument("--include-dynamic", action="store_true")
    parser.add_argument(
        "--world-origin",
        action="store_true",
        help="Use the profile's default world origin instead of camera pose.",
    )
    parser.add_argument("--max-3d-points", type=int, default=30000)
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between occupancy updates")
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs until the window closes/Ctrl+C")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    validate_navigation_arguments(parser, args)
    if args.interval < 0 or args.max_steps < 0 or args.max_3d_points <= 0:
        parser.error("--interval/--max-steps must be non-negative and --max-3d-points positive")
    return args


def fetch_occupancy(client, camera_id: int, args: argparse.Namespace, spec: dict) -> np.ndarray:
    command = f"vget /scene/occupancy npy {args.profile} {args.method}"
    if not args.world_origin:
        location = vector3(request_ok(client, f"vget /camera/{camera_id}/location"), "camera location")
        rotation = vector3(request_ok(client, f"vget /camera/{camera_id}/rotation"), "camera rotation")
        command += " " + " ".join(
            (
                *(f"{value:.6f}" for value in location),
                f"{rotation[1]:.6f}",
                "1" if args.include_dynamic else "0",
            )
        )
    payload = request_ok(client, command)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise RuntimeError(f"Expected an NPY payload, got {payload!r}")
    occupancy = np.load(BytesIO(bytes(payload)), allow_pickle=False)
    expected_shape = tuple(int(value) for value in spec["shape"])
    if occupancy.dtype != np.bool_ or occupancy.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected occupancy array: dtype={occupancy.dtype}, "
            f"shape={occupancy.shape}, expected bool {expected_shape}"
        )
    return occupancy


def draw_occupancy(figure, occupancy: np.ndarray, spec: dict, args: argparse.Namespace, tick: int) -> None:
    minimum = np.asarray(spec["min_m"], dtype=np.float32)
    maximum = np.asarray(spec["max_m"], dtype=np.float32)
    occupied = np.argwhere(occupancy)
    print(
        f"profile={args.profile} method={args.method} shape={occupancy.shape} "
        f"occupied={occupied.shape[0]:,}/{occupancy.size:,} "
        f"ratio={occupied.shape[0] / occupancy.size:.4%}"
    )

    figure.clear()
    top = figure.add_subplot(1, 3, 1)
    middle = figure.add_subplot(1, 3, 2)
    cloud = figure.add_subplot(1, 3, 3, projection="3d")

    # Axis order is [x, y_up, z].  Images therefore plot local z horizontally
    # and local x vertically, while y is the vertical slice index.
    top.imshow(
        occupancy.any(axis=1),
        origin="lower",
        aspect="auto",
        extent=(minimum[2], maximum[2], minimum[0], maximum[0]),
        cmap="gray_r",
    )
    top.set(title="Top-down occupancy (max over y_up)", xlabel="local z (m)", ylabel="local x (m)")

    mid_index = occupancy.shape[1] // 2
    middle.imshow(
        occupancy[:, mid_index, :],
        origin="lower",
        aspect="auto",
        extent=(minimum[2], maximum[2], minimum[0], maximum[0]),
        cmap="magma",
    )
    mid_height = minimum[1] + (mid_index + 0.5) * (maximum[1] - minimum[1]) / occupancy.shape[1]
    middle.set(
        title=f"Horizontal slice at y_up={mid_height:.2f} m",
        xlabel="local z (m)",
        ylabel="local x (m)",
    )

    if occupied.shape[0] > args.max_3d_points:
        indices = np.linspace(0, occupied.shape[0] - 1, args.max_3d_points, dtype=np.int64)
        occupied = occupied[indices]
    voxel_size = (maximum - minimum) / np.asarray(occupancy.shape)
    points = minimum + (occupied + 0.5) * voxel_size
    cloud.scatter(points[:, 0], points[:, 2], points[:, 1], s=0.35, c=points[:, 1], cmap="viridis")
    cloud.set(title="Occupied voxels", xlabel="x (m)", ylabel="z (m)", zlabel="y_up (m)")

    figure.suptitle(
        f"Gym Navigation occupancy | tick {tick} | {args.profile} / {args.method} | "
        f"{np.count_nonzero(occupancy):,} occupied voxels"
    )


def run(args: argparse.Namespace) -> int:
    env = make_navigation_env(args, agent_category="player", population=1)
    figure = None
    try:
        observations = env.reset()
        client = navigation_client(env)
        camera_id = navigation_camera_id(env) if args.camera_id == "auto" else int(args.camera_id)
        spec = json.loads(str(request_ok(client, f"vget /scene/occupancy/spec {args.profile} {args.method}")))
        if not args.no_show or args.save:
            figure = plt.figure(figsize=(15, 5.2), constrained_layout=True)
        print(
            f"env={args.env_id} agent={env.unwrapped.player_list[0]} camera={camera_id} "
            f"resolution={args.width}x{args.height}"
        )
        tick = 0
        while True:
            iteration_start = time.perf_counter()
            observations, rewards, done, info = env.step([None])
            occupancy = fetch_occupancy(client, camera_id, args, spec)
            tick += 1
            if figure is not None:
                draw_occupancy(figure, occupancy, spec, args, tick)
                if args.save:
                    args.save.parent.mkdir(parents=True, exist_ok=True)
                    figure.savefig(args.save, dpi=160)
                    print(f"saved={args.save.resolve()}")
                if not args.no_show:
                    plt.show(block=False)
                    plt.pause(0.001)
                    if not plt.fignum_exists(figure.number):
                        break
            if done:
                observations = env.reset()
                client = navigation_client(env)
                camera_id = navigation_camera_id(env) if args.camera_id == "auto" else int(args.camera_id)
            if args.max_steps and tick >= args.max_steps:
                break
            remaining = args.interval - (time.perf_counter() - iteration_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close()
        if figure is not None and args.no_show:
            plt.close(figure)
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
