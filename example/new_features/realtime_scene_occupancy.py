"""Continuously capture and visualize a shared-memory scene occupancy grid.

The default region matches the interactive UnrealCV demo: mesh occupancy in a
20 m cube centered on the Unreal world origin, sampled at 5 cm per voxel.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


WORKFLOW_DIR = Path(__file__).resolve().parent
PLUGIN_ROOT = WORKFLOW_DIR.parent
for candidate in (
    PLUGIN_ROOT / "client" / "python",
    PLUGIN_ROOT / "distribution" / "python",
    PLUGIN_ROOT / "Source" / "uezoo",
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

import unrealcv  # noqa: E402


CAMERA_VIEWS = (
    (
        "Camera forward",
        np.asarray((1, 0, 0), dtype=np.float32),
        "Camera right / Y up",
    ),
    (
        "Camera backward",
        np.asarray((-1, 0, 0), dtype=np.float32),
        "Camera left / Y up",
    ),
    (
        "Camera left",
        np.asarray((0, 0, -1), dtype=np.float32),
        "Camera forward / Y up",
    ),
    (
        "Camera right",
        np.asarray((0, 0, 1), dtype=np.float32),
        "Camera backward / Y up",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live four-view visualization of UnrealCV mesh occupancy."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--extent-m", type=float, default=20.0)
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    parser.add_argument("--method", choices=("mesh", "bounds"), default="mesh")
    parser.add_argument("--max-points", type=int, default=75_000)
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=1000)
    parser.add_argument("--include-dynamic", action="store_true")
    parser.add_argument("--camera-id", default="0", help="Camera used as the live region center.")
    parser.add_argument(
        "--origin-cm",
        nargs=3,
        type=float,
        help="Fixed origin override; by default the camera location is read every frame.",
    )
    parser.add_argument(
        "--yaw-degrees",
        type=float,
        help="Fixed yaw override; by default camera yaw is read every frame.",
    )
    parser.add_argument("--output", type=Path, help="Overwrite this PNG with the latest frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Zero runs until Q or Esc.")
    parser.add_argument("--no-display", action="store_true", help="Capture without cv2.imshow.")
    return parser.parse_args()


def open_shared_memory(metadata: dict) -> mmap.mmap:
    size = int(metadata["offset_bytes"]) + int(metadata["num_bytes"])
    transport = metadata["transport"]
    if transport == "windows_shared_memory":
        return mmap.mmap(-1, size, tagname=metadata["name"], access=mmap.ACCESS_READ)
    if transport == "posix_shared_memory" and sys.platform.startswith("linux"):
        descriptor = os.open(
            Path("/dev/shm") / str(metadata["name"]).lstrip("/"), os.O_RDONLY
        )
        try:
            return mmap.mmap(descriptor, size, access=mmap.ACCESS_READ)
        finally:
            os.close(descriptor)
    raise RuntimeError(f"Unsupported shared-memory transport: {transport}")


def make_command(
    args: argparse.Namespace,
    origin_cm: tuple[float, float, float],
    yaw_degrees: float,
) -> str:
    voxel_count = int(math.ceil(args.extent_m / args.voxel_size_m))
    half = voxel_count * args.voxel_size_m / 2.0
    bounds = (-half, half, -half, half, -half, half)
    values = " ".join(f"{value:g}" for value in bounds)
    origin = " ".join(f"{value:g}" for value in origin_cm)
    return (
        f"vget /scene/occupancy_shared_region {args.method} {values} "
        f"{args.voxel_size_m:g} {origin} {yaw_degrees:g} "
        f"{int(args.include_dynamic)}"
    )


def parse_vector(response: object, label: str) -> tuple[float, float, float]:
    values = str(response).replace(",", " ").split()
    if len(values) != 3:
        raise ValueError(f"Unexpected camera {label} response: {response!r}")
    return tuple(float(value) for value in values)


def occupied_xyz(metadata: dict, shared: mmap.mmap, max_points: int) -> tuple[np.ndarray, int]:
    offset = int(metadata["offset_bytes"])
    shape = tuple(int(value) for value in metadata["shape"])
    grid = np.ndarray(
        shape,
        dtype=np.bool_,
        buffer=shared,
        offset=offset,
        order="C",
    )
    flat = np.flatnonzero(grid)
    occupied_count = int(flat.size)
    if occupied_count > max_points:
        step = int(np.ceil(occupied_count / max_points))
        flat = flat[::step]

    yz = shape[1] * shape[2]
    x_index = flat // yz
    remainder = flat % yz
    y_index = remainder // shape[2]
    z_index = remainder % shape[2]
    indices = np.column_stack((x_index, y_index, z_index)).astype(np.float32)
    minimum = np.asarray(metadata["min_meters"], dtype=np.float32)
    maximum = np.asarray(metadata["max_meters"], dtype=np.float32)
    voxel = (maximum - minimum) / np.asarray(shape, dtype=np.float32)
    return minimum + (indices + 0.5) * voxel, occupied_count


def render_panel(
    xyz: np.ndarray,
    voxel_size: np.ndarray,
    extent_m: float,
    horizontal_forward: np.ndarray,
    size: tuple[int, int],
    title: str,
    axis_label: str,
) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), 250, dtype=np.uint8)
    plot_left, plot_top = 55, 42
    plot_width, plot_height = width - 75, height - 72
    cv2.rectangle(
        panel,
        (plot_left, plot_top),
        (plot_left + plot_width, plot_top + plot_height),
        (205, 205, 205),
        1,
    )
    if xyz.size:
        centered = xyz - xyz.mean(axis=0, keepdims=True)
        elevation = np.deg2rad(20.0)
        view_forward = horizontal_forward * np.cos(elevation)
        view_forward = view_forward.copy()
        view_forward[1] = -np.sin(elevation)
        view_forward /= np.linalg.norm(view_forward)
        world_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        view_right = np.cross(view_forward, world_up)
        view_right /= np.linalg.norm(view_right)
        view_up = np.cross(view_right, view_forward)
        view_rotation = np.stack((view_right, view_up, view_forward))
        camera_position = -view_forward * (extent_m * 1.65)
        camera_centers = centered - camera_position
        projected = camera_centers @ view_rotation.T
        focal_pixels = 0.5 * min(plot_width, plot_height) / np.tan(np.deg2rad(48.0) / 2.0)
        corner_signs = np.asarray(
            (
                (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
            ),
            dtype=np.float32,
        )
        corners = centered[:, None, :] + corner_signs[None, :, :] * voxel_size[None, None, :] * 0.5
        camera_corners = corners - camera_position
        projected_corners = camera_corners @ view_rotation.T
        corner_depth = np.maximum(projected_corners[:, :, 2], 1e-4)
        px = np.rint(
            plot_left + plot_width / 2.0
            + projected_corners[:, :, 0] / corner_depth * focal_pixels
        ).astype(np.int32)
        py = np.rint(
            plot_top + plot_height / 2.0
            - projected_corners[:, :, 1] / corner_depth * focal_pixels
        ).astype(np.int32)
        heights = xyz[:, 1]
        height_u8 = np.clip((heights / extent_m + 0.5) * 255.0, 0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(height_u8.reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0]
        # Paint far voxels first. Filling the projected cube hull makes adjacent
        # occupied cells touch instead of appearing as separated point samples.
        for point_index in np.argsort(-projected[:, 2]):
            polygon = cv2.convexHull(
                np.column_stack((px[point_index], py[point_index])).astype(np.int32)
            )
            color = tuple(int(channel) for channel in colors[point_index])
            cv2.fillConvexPoly(panel, polygon, color, lineType=cv2.LINE_8)

    cv2.putText(panel, title, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{axis_label} (m), color = Y up", (16, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 50, 50), 1, cv2.LINE_AA)
    return panel


def render_frame(
    xyz: np.ndarray,
    args: argparse.Namespace,
    metadata: dict,
    occupied_count: int,
    capture_fps: float,
    loop_fps: float,
) -> np.ndarray:
    minimum = np.asarray(metadata["min_meters"], dtype=np.float32)
    maximum = np.asarray(metadata["max_meters"], dtype=np.float32)
    actual_extent_m = float(np.max(maximum - minimum))
    voxel_size = (maximum - minimum) / np.asarray(metadata["shape"], dtype=np.float32)
    header_height = 62
    panel_width = args.window_width // 2
    panel_height = (args.window_height - header_height) // 2
    panels = [
        render_panel(xyz, voxel_size, actual_extent_m, horizontal_forward, (panel_width, panel_height), title, axis_label)
        for title, horizontal_forward, axis_label in CAMERA_VIEWS
    ]
    body = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    frame = np.full((body.shape[0] + header_height, body.shape[1], 3), 255, dtype=np.uint8)
    frame[header_height:] = body
    total = int(np.prod(metadata["shape"]))
    occupancy_rate = 100.0 * occupied_count / total
    headline = (
        f"UnrealCV live {args.method} occupancy | {actual_extent_m:g}m cube | "
        f"shape={tuple(metadata['shape'])} | target={args.target_fps:g} FPS"
    )
    status = (
        f"capture={capture_fps:.2f} FPS  loop={loop_fps:.2f} FPS  "
        f"occupied={occupied_count:,}/{total:,} ({occupancy_rate:.3f}%)  "
        f"origin_cm={tuple(round(value, 1) for value in metadata['origin_cm'])}  "
        f"yaw={float(metadata['yaw_degrees']):.1f}  Q/Esc: quit"
    )
    cv2.putText(frame, headline, (18, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (15, 15, 15), 1, cv2.LINE_AA)
    cv2.putText(frame, status, (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 50, 50), 1, cv2.LINE_AA)
    return frame


def main() -> int:
    args = parse_args()
    if args.target_fps <= 0 or args.extent_m <= 0 or args.voxel_size_m <= 0:
        raise ValueError("target FPS, extent, and voxel size must be positive")
    if args.max_points <= 0:
        raise ValueError("max-points must be positive")

    client = unrealcv.Client((args.host, args.port))
    if not client.connect():
        raise ConnectionError(f"Unable to connect to UnrealCV at {args.host}:{args.port}")

    title = "UnrealCV Live Mesh Occupancy"
    if not args.no_display:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, args.window_width, args.window_height)

    capture_times: deque[float] = deque(maxlen=30)
    loop_times: deque[float] = deque(maxlen=30)
    period = 1.0 / args.target_fps
    frame_index = 0
    try:
        while not args.max_frames or frame_index < args.max_frames:
            loop_start = time.perf_counter()
            request_start = loop_start
            if args.origin_cm is None:
                location_response = client.request(
                    f"vget /camera/{args.camera_id}/location", timeout=10
                )
                origin_cm = parse_vector(location_response, "location")
            else:
                origin_cm = tuple(args.origin_cm)
            if args.yaw_degrees is None:
                rotation_response = client.request(
                    f"vget /camera/{args.camera_id}/rotation", timeout=10
                )
                pitch, yaw_degrees, roll = parse_vector(rotation_response, "rotation")
            else:
                pitch, yaw_degrees, roll = 0.0, args.yaw_degrees, 0.0
            command = make_command(args, origin_cm, yaw_degrees)
            response = client.request(command, timeout=max(30.0, period * 4.0))
            request_end = time.perf_counter()
            if isinstance(response, str) and response.lower().startswith("error"):
                raise RuntimeError(response)
            metadata = json.loads(str(response))
            shared = open_shared_memory(metadata)
            try:
                xyz, occupied_count = occupied_xyz(metadata, shared, args.max_points)
            finally:
                shared.close()

            capture_times.append(request_end - request_start)
            capture_fps = len(capture_times) / sum(capture_times)
            elapsed_samples = sum(loop_times)
            loop_fps = len(loop_times) / elapsed_samples if elapsed_samples else 0.0
            frame = render_frame(
                xyz, args, metadata, occupied_count, capture_fps, loop_fps
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.output), frame)
            if not args.no_display:
                cv2.imshow(title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_index += 1
            remaining = period - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
            loop_times.append(time.perf_counter() - loop_start)
            current_loop_fps = len(loop_times) / sum(loop_times)
            print(
                f"frame={frame_index} capture_fps={capture_fps:.2f} "
                f"loop_fps={current_loop_fps:.2f} occupied={occupied_count} "
                f"origin_cm=({origin_cm[0]:.1f},{origin_cm[1]:.1f},{origin_cm[2]:.1f}) "
                f"camera_rotation=({pitch:.1f},{yaw_degrees:.1f},{roll:.1f})",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
