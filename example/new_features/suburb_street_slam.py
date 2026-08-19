#!/usr/bin/env python3
"""Interactive street LiDAR SLAM diagnostic in UnrealZoo's Suburb map.

The example uses the registered gym_unrealcv Navigation environment to launch
SuburbNeighborhood_Day, configure one player and its camera, apply keyboard
actions through ``env.step()``, capture pose-conditioned LiDAR observations, and
update a 2-D voxel map.
It is deliberately a pose-assisted mapping example, not a SLAM optimizer: the
ground-truth Unreal camera pose is used so LiDAR transport/geometry problems can
be isolated from odometry or scan-matching errors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from gym_navigation_demo import (  # noqa: E402
    DEFAULT_ENV_ID,
    add_navigation_arguments,
    make_navigation_env,
    validate_navigation_arguments,
)

from lidar_visualization import (
    downsample,
    request_pose_conditioned_scans,
    transform_cloud_to_world,
    unreal_rotation_matrix,
)


def wrapped_angle_error_deg(values: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Smallest absolute cyclic distance from every angle to a reference."""
    delta = values[:, None] - references[None, :]
    return np.min(np.abs((delta + 180.0) % 360.0 - 180.0), axis=1)


def lidar_angles_deg(scan_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = scan_local[:, :3]
    yaw = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    pitch = np.rad2deg(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    return pitch, yaw


def bgr_to_display_rgb(image: np.ndarray) -> np.ndarray:
    """Convert UnrealZoo/OpenCV BGR(A) observations for Matplotlib display."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected an H x W x 3/4 color image, got {image.shape}")
    return image[..., [2, 1, 0]]


def expected_lidar_metadata(metadata: dict[str, Any] | None) -> dict[str, float | int]:
    """Use server metadata when available, otherwise current plugin defaults."""
    defaults: dict[str, float | int] = {
        "ray_count": 5760,
        "channels": 32,
        "horizontal_samples": 180,
        "min_range_m": 0.1,
        "max_range_m": 100.0,
        "upper_fov_deg": 10.0,
        "lower_fov_deg": -30.0,
        "horizontal_fov_start_deg": -180.0,
        "horizontal_fov_end_deg": 180.0,
    }
    if metadata:
        for key in defaults:
            if key in metadata:
                defaults[key] = metadata[key]
    return defaults


@dataclass
class LidarHealth:
    status: str
    hits: int
    rays: int
    hit_rate: float
    max_pitch_error_deg: float
    max_yaw_error_deg: float
    duplicate_bins: int
    ordering_errors: int
    range_errors: int
    pose_span_cm: float
    pose_span_deg: float
    static_common_rays: int = 0
    static_common_fraction: float = float("nan")
    static_p95_range_delta_m: float = float("nan")
    static_max_range_delta_m: float = float("nan")

    def compact(self) -> str:
        static_text = ""
        if self.static_common_rays:
            static_text = (
                f" | static common={self.static_common_fraction:.1%} "
                f"Δr P95/max={self.static_p95_range_delta_m * 1000.0:.2f}/"
                f"{self.static_max_range_delta_m * 1000.0:.2f} mm"
            )
        return (
            f"LiDAR {self.status} | hits {self.hits:,}/{self.rays:,} "
            f"({self.hit_rate:.1%}) | grid err P/Y "
            f"{self.max_pitch_error_deg:.4f}°/{self.max_yaw_error_deg:.4f}° | "
            f"duplicates {self.duplicate_bins} | order {self.ordering_errors} | "
            f"range {self.range_errors} | sync "
            f"{self.pose_span_cm:.4f} cm/{self.pose_span_deg:.4f}°{static_text}"
        )


class LidarDiagnostics:
    """Validate that XYZI matches the configured LiDAR ray layout."""

    def __init__(self, metadata: dict[str, Any] | None):
        self.config = expected_lidar_metadata(metadata)

    def ray_ids(self, scan_local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pitch, yaw = lidar_angles_deg(scan_local)
        channels = int(self.config["channels"])
        samples = int(self.config["horizontal_samples"])
        upper = float(self.config["upper_fov_deg"])
        lower = float(self.config["lower_fov_deg"])
        yaw_start = float(self.config["horizontal_fov_start_deg"])
        yaw_end = float(self.config["horizontal_fov_end_deg"])
        expected_pitch = np.linspace(upper, lower, channels, dtype=np.float64)
        expected_yaw = yaw_start + (np.arange(samples) + 0.5) / samples * (yaw_end - yaw_start)
        pitch_index = np.argmin(np.abs(pitch[:, None] - expected_pitch[None, :]), axis=1)
        yaw_delta = yaw[:, None] - expected_yaw[None, :]
        yaw_index = np.argmin(np.abs((yaw_delta + 180.0) % 360.0 - 180.0), axis=1)
        return pitch_index * samples + yaw_index, expected_pitch, expected_yaw

    def inspect(
        self,
        scan_local: np.ndarray,
        pose_span_cm: float,
        pose_span_deg: float,
        static_reference: np.ndarray | None = None,
    ) -> LidarHealth:
        ranges = np.linalg.norm(scan_local[:, :3], axis=1)
        pitch, yaw = lidar_angles_deg(scan_local)
        ray_ids, expected_pitch, expected_yaw = self.ray_ids(scan_local)
        pitch_error = np.min(np.abs(pitch[:, None] - expected_pitch[None, :]), axis=1)
        yaw_error = wrapped_angle_error_deg(yaw, expected_yaw)
        duplicate_bins = int(ray_ids.size - np.unique(ray_ids).size)
        ordering_errors = int(np.count_nonzero(np.diff(ray_ids) < 0))
        min_range = float(self.config["min_range_m"])
        max_range = float(self.config["max_range_m"])
        range_errors = int(np.count_nonzero((ranges < min_range - 1e-3) | (ranges > max_range + 1e-3)))

        static_common = 0
        static_common_fraction = float("nan")
        static_p95_delta = float("nan")
        static_delta = float("nan")
        if static_reference is not None:
            reference_ids, _, _ = self.ray_ids(static_reference)
            common, current_index, reference_index = np.intersect1d(
                ray_ids, reference_ids, assume_unique=False, return_indices=True
            )
            static_common = int(common.size)
            static_common_fraction = static_common / max(
                scan_local.shape[0], static_reference.shape[0], 1
            )
            if static_common:
                reference_ranges = np.linalg.norm(static_reference[reference_index, :3], axis=1)
                static_deltas = np.abs(ranges[current_index] - reference_ranges)
                static_p95_delta = float(np.percentile(static_deltas, 95.0))
                static_delta = float(np.max(static_deltas))

        max_pitch_error = float(np.max(pitch_error, initial=0.0))
        max_yaw_error = float(np.max(yaw_error, initial=0.0))
        failed = (
            scan_local.shape[0] > int(self.config["ray_count"])
            or duplicate_bins > 0
            or ordering_errors > 0
            or range_errors > 0
            or max_pitch_error > 0.02
            or max_yaw_error > 0.02
            or pose_span_cm > 0.1
            or pose_span_deg > 0.1
            or (
                static_common > 0
                and (static_common_fraction < 0.90 or static_p95_delta > 0.02)
            )
        )
        rays = int(self.config["ray_count"])
        return LidarHealth(
            status="FAIL" if failed else "PASS",
            hits=scan_local.shape[0],
            rays=rays,
            hit_rate=scan_local.shape[0] / max(rays, 1),
            max_pitch_error_deg=max_pitch_error,
            max_yaw_error_deg=max_yaw_error,
            duplicate_bins=duplicate_bins,
            ordering_errors=ordering_errors,
            range_errors=range_errors,
            pose_span_cm=pose_span_cm,
            pose_span_deg=pose_span_deg,
            static_common_rays=static_common,
            static_common_fraction=static_common_fraction,
            static_p95_range_delta_m=static_p95_delta,
            static_max_range_delta_m=static_delta,
        )


def filter_street_returns(
    scan_local: np.ndarray,
    scan_world: np.ndarray,
    camera_height_m: float,
    half_height_m: float,
    min_range_m: float,
    max_range_m: float,
) -> np.ndarray:
    """Keep wall-like returns near camera height before map accumulation.

    Filtering in world Z after applying the full pitch/yaw/roll pose prevents
    fixed LiDAR rings hitting the road and sky/roof from producing concentric
    circles in the 2-D map. The band follows the player height so gentle street
    slopes do not disappear from the map.
    """
    ranges = np.linalg.norm(scan_local[:, :3], axis=1)
    relative_z = scan_world[:, 2] - camera_height_m
    keep = (
        (ranges >= min_range_m)
        & (ranges <= max_range_m)
        & (np.abs(relative_z) <= half_height_m)
    )
    return scan_world[keep]


class VoxelMap2D:
    """Bounded XY occupancy map, keeping the newest representative per cell."""

    def __init__(self, voxel_size_m: float, max_points: int):
        self.voxel_size_m = voxel_size_m
        self.max_points = max_points
        self.cloud = np.empty((0, 4), dtype=np.float32)

    def clear(self) -> None:
        self.cloud = np.empty((0, 4), dtype=np.float32)

    def update(self, scan_world: np.ndarray) -> tuple[np.ndarray, float]:
        if scan_world.size == 0:
            return self.cloud, 0.0
        scan_voxels = np.floor(scan_world[:, :2] / self.voxel_size_m).astype(np.int64)
        scan_voxels = np.unique(scan_voxels, axis=0)
        if self.cloud.size:
            map_voxels = np.floor(self.cloud[:, :2] / self.voxel_size_m).astype(np.int64)
            scan_keys = np.ascontiguousarray(scan_voxels).view(
                np.dtype((np.void, scan_voxels.dtype.itemsize * 2))
            ).ravel()
            map_keys = np.ascontiguousarray(map_voxels).view(
                np.dtype((np.void, map_voxels.dtype.itemsize * 2))
            ).ravel()
            overlap = float(np.count_nonzero(np.isin(scan_keys, map_keys)) / max(scan_keys.size, 1))
        else:
            overlap = 0.0

        combined = np.concatenate((self.cloud, scan_world), axis=0)
        voxels = np.floor(combined[:, :2] / self.voxel_size_m).astype(np.int64)
        reversed_cloud = combined[::-1]
        reversed_voxels = voxels[::-1]
        _, newest = np.unique(reversed_voxels, axis=0, return_index=True)
        self.cloud = reversed_cloud[newest]
        if self.cloud.shape[0] > self.max_points:
            self.cloud = downsample(self.cloud, self.max_points).copy()
        return self.cloud, overlap


class KeyboardState:
    """Global keyboard state so focus remains in the Unreal player window."""

    def __init__(self):
        from pynput import keyboard

        self.keyboard = keyboard
        self.held: set[str] = set()
        self.events: set[str] = set()
        self.listener = keyboard.Listener(on_press=self._press, on_release=self._release)

    @staticmethod
    def _name(key: Any) -> str | None:
        try:
            return str(key.char).lower()
        except AttributeError:
            name = getattr(key, "name", None)
            return str(name).lower() if name else None

    def _press(self, key: Any) -> None:
        name = self._name(key)
        if name:
            if name not in self.held:
                self.events.add(name)
            self.held.add(name)

    def _release(self, key: Any) -> None:
        name = self._name(key)
        if name:
            self.held.discard(name)

    def start(self) -> None:
        self.listener.start()

    def stop(self) -> None:
        self.listener.stop()

    def consume(self, name: str) -> bool:
        if name not in self.events:
            return False
        self.events.remove(name)
        return True

    def action(self) -> tuple[tuple[float, float], int, int]:
        speed = 100.0 * (float("i" in self.held) - float("k" in self.held))
        turn = 30.0 * (float("l" in self.held) - float("j" in self.held))
        head = 1 if "up" in self.held else 2 if "down" in self.held else 0
        crouch = bool({"ctrl", "ctrl_l", "ctrl_r"}.intersection(self.held))
        animation = 1 if "space" in self.held else 2 if crouch else 0
        return (turn, speed), head, animation


class StreetSlamFigure:
    def __init__(self, initial_rgb: np.ndarray, max_render_points: int, map_range_m: float):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.max_render_points = max_render_points
        self.map_range_m = map_range_m
        self.origin_m: np.ndarray | None = None
        self.bounds: tuple[float, float, float, float] | None = None
        self.figure, (self.rgb_axis, self.scan_axis, self.map_axis) = plt.subplots(
            1, 3, figsize=(18, 6.5), gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]}
        )
        self.figure.canvas.manager.set_window_title("UnrealZoo Suburb LiDAR Voxel SLAM")
        self.figure.subplots_adjust(bottom=0.18, wspace=0.25)
        self.rgb_artist = self.rgb_axis.imshow(bgr_to_display_rgb(initial_rgb))
        self.rgb_axis.set_title("UnrealZoo player camera")
        self.rgb_axis.axis("off")

        self.scan_axis.set_facecolor("#071016")
        self.scan_axis.set_aspect("equal", adjustable="box")
        self.scan_axis.set_xlim(-map_range_m, map_range_m)
        self.scan_axis.set_ylim(-map_range_m, map_range_m)
        self.scan_axis.set_xlabel("sensor X / forward (m)")
        self.scan_axis.set_ylabel("sensor Y / right (m)")
        self.scan_axis.grid(True, color="white", alpha=0.1)
        self.scan_points = self.scan_axis.scatter(
            [], [], c=[], cmap="coolwarm", vmin=-2.0, vmax=2.0, s=2.0, linewidths=0
        )
        self.scan_axis.arrow(0, 0, 2, 0, color="#ffc107", width=0.08)

        self.map_axis.set_facecolor("#071016")
        self.map_axis.set_aspect("equal", adjustable="box")
        self.map_axis.set_xlabel("world X from start (m)")
        self.map_axis.set_ylabel("world Y from start (m)")
        self.map_axis.grid(True, color="white", alpha=0.1)
        self.map_points = self.map_axis.scatter([], [], c="#9be7ff", s=1.0, linewidths=0)
        (self.trajectory,) = self.map_axis.plot([], [], color="#ff5252", linewidth=1.5)
        self.player = self.map_axis.scatter([], [], c="#ff3d00", marker="^", s=65)
        (self.forward,) = self.map_axis.plot([], [], color="#ffc107", linewidth=2.5)
        self.status_text = self.figure.text(
            0.5, 0.075, "Starting LiDAR diagnostics...", ha="center", va="center", fontsize=9
        )
        self.figure.text(
            0.5,
            0.025,
            "I/K forward/back | J/L turn | Up/Down look | Space jump | Ctrl crouch | "
            "C clear map | P pause mapping | Esc quit",
            ha="center",
            fontsize=9,
        )

    def reset(self) -> None:
        self.origin_m = None
        self.bounds = None
        self.map_points.set_offsets(np.empty((0, 2)))
        self.trajectory.set_data([], [])

    def _limits(self, xy: np.ndarray) -> None:
        if xy.size == 0:
            return
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        if self.bounds is None:
            self.bounds = xmin, xmax, ymin, ymax
        else:
            ox0, ox1, oy0, oy1 = self.bounds
            self.bounds = min(ox0, xmin), max(ox1, xmax), min(oy0, ymin), max(oy1, ymax)
        x0, x1, y0, y1 = self.bounds
        span = max(x1 - x0, y1 - y0, 12.0) + 4.0
        xc, yc = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        self.map_axis.set_xlim(xc - span / 2.0, xc + span / 2.0)
        self.map_axis.set_ylim(yc - span / 2.0, yc + span / 2.0)

    def update(
        self,
        rgb: np.ndarray,
        scan_local: np.ndarray,
        map_world: np.ndarray,
        location_cm: np.ndarray,
        rotation_deg: np.ndarray,
        trajectory_world: np.ndarray,
        health: LidarHealth,
        overlap: float,
        accepted_points: int,
        tick: int,
        batch_ms: float,
        paused: bool,
        alignment_warning: bool,
    ) -> None:
        position_m = location_cm / 100.0
        if self.origin_m is None:
            self.origin_m = position_m.copy()
        self.rgb_artist.set_data(bgr_to_display_rgb(rgb))

        scan = downsample(scan_local, self.max_render_points)
        ranges = np.linalg.norm(scan[:, :3], axis=1)
        visible = ranges <= self.map_range_m
        scan = scan[visible]
        self.scan_points.set_offsets(scan[:, :2])
        self.scan_points.set_array(scan[:, 2])
        self.scan_axis.set_title(
            f"Current LiDAR scan | {scan.shape[0]:,}/{scan_local.shape[0]:,} shown\n"
            f"XYZ sensor-local metres | batch {batch_ms:.1f} ms"
        )

        shown_map = downsample(map_world, self.max_render_points)
        map_xy = shown_map[:, :2] - self.origin_m[:2]
        trajectory_xy = trajectory_world[:, :2] - self.origin_m[:2]
        current_xy = position_m[:2] - self.origin_m[:2]
        self.map_points.set_offsets(map_xy)
        self.trajectory.set_data(trajectory_xy[:, 0], trajectory_xy[:, 1])
        self.player.set_offsets(current_xy[None, :])
        heading = unreal_rotation_matrix(rotation_deg)[0, :2]
        self.forward.set_data(
            [current_xy[0], current_xy[0] + 2.0 * heading[0]],
            [current_xy[1], current_xy[1] + 2.0 * heading[1]],
        )
        self._limits(np.concatenate((map_xy, trajectory_xy, current_xy[None, :]), axis=0))
        state = "PAUSED" if paused else "ALIGNMENT WARN" if alignment_warning else "mapping"
        self.map_axis.set_title(
            f"Suburb street 2-D voxel map | {state} | tick {tick:,}\n"
            f"{map_world.shape[0]:,} cells | accepted {accepted_points:,} | overlap {overlap:.1%}"
        )
        self.status_text.set_text(
            health.compact()
            + (" | MAP LOW OVERLAP" if alignment_warning else "")
            + "\nPose source: camera; camera→LiDAR extrinsic assumed identity"
        )
        self.figure.canvas.draw_idle()


def capture(
    client: Any,
    camera_id: int,
    transport: str,
    timeout: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None, float, float, float]:
    source_name = "Shared memory" if transport == "shared" else "TCP NPY"
    scans, location, rotation, metadata, batch_ms, span_cm, span_deg = (
        request_pose_conditioned_scans(client, camera_id, [source_name], timeout)
    )
    return scans[source_name], location, rotation, metadata, batch_ms, span_cm, span_deg


class GymSession:
    """Own a packaged UnrealZoo process launched through the Gym environment."""

    def __init__(self, env: Any, initial_rgb: np.ndarray):
        self.env = env
        self.client = env.unwrapped.unrealcv.client
        self.player_name = env.unwrapped.player_list[0]
        self.camera_id = int(env.unwrapped.cam_list[0])
        self.level_name = str(env.unwrapped.env_name)
        self.initial_rgb = initial_rgb

    def step(self, action: tuple[tuple[float, float], int, int]) -> tuple[np.ndarray, bool, dict[str, Any]]:
        observations, _, done, info = self.env.step([action])
        return np.asarray(observations[0]), bool(done), info

    def close(self) -> None:
        self.env.close()


def create_session(args: argparse.Namespace) -> GymSession:
    env = make_navigation_env(args, agent_category="player", population=1)
    observations = env.reset()
    if len(env.unwrapped.player_list) != 1 or len(env.unwrapped.cam_list) != 1:
        raise RuntimeError(
            "Expected exactly one spawned player/camera, got "
            f"players={env.unwrapped.player_list}, cameras={env.unwrapped.cam_list}"
        )
    player_name = env.unwrapped.player_list[0]
    camera_id = int(env.unwrapped.cam_list[0])
    if camera_id < 0:
        raise RuntimeError(f"Spawned player {player_name!r} has no assigned camera")
    return GymSession(env, np.asarray(observations[0]))
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_navigation_arguments(parser, default_env_id=DEFAULT_ENV_ID)
    parser.add_argument("--transport", choices=("auto", "shared", "npy"), default="auto")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--voxel-size", type=float, default=0.15)
    parser.add_argument("--map-half-height", type=float, default=0.30)
    parser.add_argument("--map-min-range", type=float, default=0.5)
    parser.add_argument("--map-max-range", type=float, default=40.0)
    parser.add_argument("--max-map-points", type=int, default=150_000)
    parser.add_argument("--max-render-points", type=int, default=60_000)
    parser.add_argument("--diagnostic-every", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs until Esc/Ctrl+C")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save", type=Path, help="Continuously replace this dashboard PNG")
    args = parser.parse_args()
    if args.voxel_size <= 0 or args.map_half_height <= 0:
        parser.error("voxel size and map half-height must be positive")
    if not 0 <= args.map_min_range < args.map_max_range:
        parser.error("--map-max-range must exceed non-negative --map-min-range")
    if args.max_map_points <= 0 or args.max_render_points <= 0:
        parser.error("map/render point limits must be positive")
    if args.diagnostic_every <= 0 or args.max_steps < 0:
        parser.error("diagnostic interval must be positive and max steps non-negative")
    validate_navigation_arguments(parser, args)
    return args


def run(args: argparse.Namespace) -> int:
    session = None
    keys = None
    figure = None
    try:
        session = create_session(args)
        rgb = session.initial_rgb
        player_name = session.player_name
        camera_id = session.camera_id
        client = session.client
        print(
            f"Mode: gym Navigation | level: {session.level_name} | "
            f"spawned player: {player_name} | camera: {camera_id}"
        )

        transport = "shared" if args.transport == "auto" else args.transport
        try:
            first = capture(client, camera_id, transport, args.timeout)
        except Exception as exc:
            if args.transport != "auto" or transport != "shared":
                raise
            print(f"Shared-memory LiDAR unavailable ({exc}); falling back to TCP NPY")
            transport = "npy"
            first = capture(client, camera_id, transport, args.timeout)
        scan_reference, reference_location, reference_rotation, metadata, _, _, _ = first
        diagnostics = LidarDiagnostics(metadata)

        # Two back-to-back stationary captures expose nondeterministic ray output
        # before keyboard movement is accepted.
        scan, location, rotation, metadata, batch_ms, span_cm, span_deg = capture(
            client, camera_id, transport, args.timeout
        )
        static_translation_cm = float(np.linalg.norm(location - reference_location))
        static_rotation_deg = float(
            np.linalg.norm((rotation - reference_rotation + 180.0) % 360.0 - 180.0)
        )
        static_reference = (
            scan_reference
            if static_translation_cm <= 0.1 and static_rotation_deg <= 0.1
            else None
        )
        health = diagnostics.inspect(scan, span_cm, span_deg, static_reference)
        print(f"Startup static test: {health.compact()}")
        if static_reference is None:
            print(
                "Static repeat comparison skipped because the camera moved between captures: "
                f"{static_translation_cm:.4f} cm/{static_rotation_deg:.4f} deg"
            )
        print(
            "Protocol limitation: lidar_shared does not include the LiDAR component capture pose; "
            "mapping assumes the spawned camera and LiDAR component have identity extrinsics."
        )

        point_map = VoxelMap2D(args.voxel_size, args.max_map_points)
        trajectory: list[np.ndarray] = []
        paused = False
        if not args.no_show or args.save:
            figure = StreetSlamFigure(rgb, args.max_render_points, args.map_max_range)
        keys = KeyboardState()
        keys.start()
        print(
            "Controls: I/K forward/back, J/L turn, Up/Down look, Space jump, Ctrl crouch, "
            "C clear, P pause, Esc quit"
        )

        tick = 0
        low_overlap_streak = 0
        while True:
            if keys.consume("esc"):
                break
            if args.max_steps and tick >= args.max_steps:
                break
            if keys.consume("p"):
                paused = not paused
                print("Map accumulation paused" if paused else "Map accumulation resumed")
            if keys.consume("c"):
                point_map.clear()
                trajectory.clear()
                if figure is not None:
                    figure.reset()
                print("Street map cleared")

            rgb, done, info = session.step(keys.action())
            scan, location, rotation, _, batch_ms, span_cm, span_deg = capture(
                client, camera_id, transport, args.timeout
            )
            health = diagnostics.inspect(scan, span_cm, span_deg)
            scan_world = transform_cloud_to_world(scan, location, rotation)
            filtered = filter_street_returns(
                scan,
                scan_world,
                location[2] / 100.0,
                args.map_half_height,
                args.map_min_range,
                args.map_max_range,
            )
            previous_cells = point_map.cloud.shape[0]
            if health.status == "PASS" and not paused:
                map_world, overlap = point_map.update(filtered)
            else:
                map_world, overlap = point_map.cloud, 0.0
            overlap_observable = (
                health.status == "PASS"
                and not paused
                and previous_cells >= 500
                and filtered.shape[0] >= 100
            )
            if overlap_observable and overlap < 0.05:
                low_overlap_streak += 1
            elif overlap_observable:
                low_overlap_streak = 0
            alignment_warning = low_overlap_streak >= 5
            trajectory.append(location / 100.0)
            trajectory_world = np.asarray(trajectory, dtype=np.float64)
            tick += 1

            if (
                tick == 1
                or tick % args.diagnostic_every == 0
                or health.status != "PASS"
                or alignment_warning
            ):
                print(
                    f"tick={tick} transport={transport} camera={camera_id} "
                    f"accepted={filtered.shape[0]:,} cells={map_world.shape[0]:,} "
                    f"overlap={overlap:.1%} collision={info.get('Collision', 0)} | "
                    f"{health.compact()}"
                )
            if done:
                print("Navigation episode reports done; keeping the SLAM session alive for diagnosis")

            if figure is not None:
                figure.update(
                    rgb,
                    scan,
                    map_world,
                    location,
                    rotation,
                    trajectory_world,
                    health,
                    overlap,
                    filtered.shape[0],
                    tick,
                    batch_ms,
                    paused,
                    alignment_warning,
                )
                if args.save:
                    args.save.parent.mkdir(parents=True, exist_ok=True)
                    figure.figure.savefig(args.save, dpi=140, bbox_inches="tight")
                if not args.no_show:
                    figure.plt.show(block=False)
                    figure.plt.pause(0.001)
                    if not figure.plt.fignum_exists(figure.figure.number):
                        break
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        if keys is not None:
            keys.stop()
        if session is not None:
            session.close()
    return 0


def main() -> int:
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
