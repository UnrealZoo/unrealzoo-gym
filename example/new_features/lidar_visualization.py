#!/usr/bin/env python3
"""Build a live, pose-conditioned point-cloud map from UnrealCV LiDAR.

Start an UnrealCV-enabled Unreal Zoo scene before running this demo. The LiDAR
payload is expected to be an N x 4 float32 array with columns X, Y, Z and
intensity in the sensor-local coordinate frame. Each scan is transformed by
the camera/LiDAR world pose and accumulated into a voxelized world-space map.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import mmap
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


LIDAR_FIELDS = ["x", "y", "z", "intensity"]


def _error_text(response: Any) -> str | None:
    """Return a readable UnrealCV error, while leaving binary NPY untouched."""
    if response is None:
        return "UnrealCV returned no data"
    if isinstance(response, bytes):
        if response.startswith(b"\x93NUMPY"):
            return None
        try:
            text = response.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
    else:
        text = str(response).strip()
    if not text:
        return "UnrealCV returned an empty response"
    if text.lower().startswith(("error", "argument invalid")):
        return text
    return None


def validate_cloud(cloud: np.ndarray, source: str) -> np.ndarray:
    """Validate and normalize the UnrealCV XYZI contract."""
    cloud = np.asarray(cloud)
    if cloud.ndim != 2 or cloud.shape[1] != 4:
        raise ValueError(f"{source}: expected an N x 4 XYZI array, got {cloud.shape}")
    if cloud.shape[0] == 0:
        raise ValueError(f"{source}: LiDAR scan contains no hits")
    if not np.issubdtype(cloud.dtype, np.floating):
        raise ValueError(f"{source}: expected floating-point data, got {cloud.dtype}")
    if not np.isfinite(cloud).all():
        raise ValueError(f"{source}: point cloud contains NaN or infinity")
    if np.any((cloud[:, 3] < 0.0) | (cloud[:, 3] > 1.0)):
        raise ValueError(f"{source}: intensity values must be in [0, 1]")
    return cloud.astype(np.float32, copy=False)


def decode_npy_response(response: Any, command: str) -> np.ndarray:
    """Decode an NPY response already returned by UnrealCV."""
    error = _error_text(response)
    if error:
        raise RuntimeError(f"{command}: {error}")
    if not isinstance(response, (bytes, bytearray, memoryview)):
        raise TypeError(f"{command}: expected binary NPY data, got {type(response).__name__}")
    try:
        cloud = np.load(BytesIO(bytes(response)), allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"{command}: invalid NPY response") from exc
    return validate_cloud(cloud, "TCP NPY")


def request_npy(client: Any, camera_id: int, timeout: float) -> np.ndarray:
    """Fetch ``vget /camera/<id>/lidar npy`` over UnrealCV TCP."""
    command = f"vget /camera/{camera_id}/lidar npy"
    return decode_npy_response(client.request(command, timeout), command)


def parse_shared_metadata(response: Any) -> dict[str, Any]:
    """Decode and validate the JSON returned by ``lidar_shared``."""
    error = _error_text(response)
    if error:
        raise RuntimeError(error)
    if isinstance(response, bytes):
        response = response.decode("utf-8")
    try:
        meta = json.loads(str(response))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("lidar_shared returned invalid JSON metadata") from exc

    required = {
        "transport",
        "name",
        "num_bytes",
        "offset_bytes",
        "shape",
        "dtype",
        "layout",
        "channel_order",
        "point_count",
        "fields",
        "coordinate_frame",
        "axes",
        "units",
    }
    missing = sorted(required.difference(meta))
    if missing:
        raise ValueError(f"lidar_shared metadata is missing: {', '.join(missing)}")

    point_count = int(meta["point_count"])
    shape = tuple(int(value) for value in meta["shape"])
    num_bytes = int(meta["num_bytes"])
    offset_bytes = int(meta["offset_bytes"])
    expected_bytes = point_count * 4 * np.dtype(np.float32).itemsize
    if point_count <= 0 or shape != (point_count, 4):
        raise ValueError(f"lidar_shared has inconsistent point count/shape: {point_count}, {shape}")
    if num_bytes != expected_bytes or offset_bytes < 0:
        raise ValueError(
            f"lidar_shared has invalid byte range: offset={offset_bytes}, "
            f"bytes={num_bytes}, expected={expected_bytes}"
        )
    if meta["dtype"] != "float32" or meta["layout"] != "NC":
        raise ValueError(f"lidar_shared expected float32 NC data, got {meta['dtype']} {meta['layout']}")
    if meta["channel_order"] != "XYZI" or meta["fields"] != LIDAR_FIELDS:
        raise ValueError("lidar_shared field order is not XYZI")
    if meta["coordinate_frame"] != "sensor_local" or meta["units"] != "metres":
        raise ValueError("lidar_shared coordinate metadata does not match the LiDAR contract")
    return meta


def _open_shared_memory(meta: dict[str, Any]) -> mmap.mmap:
    """Open an UnrealCV shared-memory mapping on Windows or Linux."""
    size = int(meta["offset_bytes"]) + int(meta["num_bytes"])
    transport = str(meta["transport"])
    name = str(meta["name"])
    if transport == "windows_shared_memory" and os.name == "nt":
        return mmap.mmap(-1, size, tagname=name, access=mmap.ACCESS_READ)
    if transport == "posix_shared_memory" and sys.platform.startswith("linux"):
        shared_memory_path = Path("/dev/shm") / name.lstrip("/")
        file_descriptor = os.open(shared_memory_path, os.O_RDONLY)
        try:
            return mmap.mmap(file_descriptor, size, access=mmap.ACCESS_READ)
        finally:
            os.close(file_descriptor)
    raise RuntimeError(
        f"shared-memory transport {transport!r} is unavailable on {sys.platform}; "
        "the UnrealCV client and server must run on the same machine"
    )


def decode_shared_response(
    response: Any, command: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Open and copy a shared-memory response already returned by UnrealCV."""
    try:
        meta = parse_shared_metadata(response)
    except Exception as exc:
        raise type(exc)(f"{command}: {exc}") from exc

    offset = int(meta["offset_bytes"])
    num_bytes = int(meta["num_bytes"])
    shape = tuple(int(value) for value in meta["shape"])
    shared_memory = _open_shared_memory(meta)
    try:
        # Copy immediately: UnrealCV can reuse/replace the mapping on the next request.
        payload = shared_memory[offset : offset + num_bytes]
    finally:
        shared_memory.close()
    cloud = np.frombuffer(payload, dtype=np.float32).reshape(shape).copy()
    return validate_cloud(cloud, "shared memory"), meta


def request_shared(
    client: Any, camera_id: int, timeout: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fetch a point cloud through UnrealCV's platform shared memory."""
    command = f"vget /camera/{camera_id}/lidar_shared"
    return decode_shared_response(client.request(command, timeout), command)


def _parse_vector(response: Any, length: int, command: str) -> np.ndarray:
    error = _error_text(response)
    if error:
        raise RuntimeError(f"{command}: {error}")
    if isinstance(response, bytes):
        response = response.decode("utf-8")
    try:
        values = np.asarray([float(value) for value in str(response).split()], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{command}: expected {length} floating-point values") from exc
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(f"{command}: expected {length} finite values, got {response!r}")
    return values


def unreal_rotation_matrix(rotation_pyr_deg: np.ndarray) -> np.ndarray:
    """Return Unreal's local-to-world matrix for [pitch, yaw, roll] degrees.

    The coefficients match ``FRotationTranslationMatrix`` in Unreal Engine.
    Points in this demo are row vectors, so ``world = local @ matrix``.
    """
    pitch, yaw, roll = np.deg2rad(rotation_pyr_deg)
    sp, sy, sr = np.sin((pitch, yaw, roll))
    cp, cy, cr = np.cos((pitch, yaw, roll))
    return np.asarray(
        [
            [cp * cy, cp * sy, sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
            [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp],
        ],
        dtype=np.float64,
    )


def transform_cloud_to_world(
    cloud_local: np.ndarray,
    location_ue_cm: np.ndarray,
    rotation_pyr_deg: np.ndarray,
) -> np.ndarray:
    """Transform sensor-local metres into Unreal world-space metres."""
    world = cloud_local.astype(np.float32, copy=True)
    rotation = unreal_rotation_matrix(rotation_pyr_deg)
    translation_m = np.asarray(location_ue_cm, dtype=np.float64) / 100.0
    world[:, :3] = cloud_local[:, :3].astype(np.float64) @ rotation + translation_m
    return world


def angle_delta_deg(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Shortest signed per-axis Euler delta in degrees."""
    return (np.asarray(current) - np.asarray(previous) + 180.0) % 360.0 - 180.0


class PoseMotionEstimator:
    """Estimate observed camera speed from consecutive synchronized poses."""

    def __init__(self):
        self.previous_time: float | None = None
        self.previous_location_m: np.ndarray | None = None
        self.previous_rotation_deg: np.ndarray | None = None

    def update(
        self,
        location_ue_cm: np.ndarray,
        rotation_pyr_deg: np.ndarray,
        timestamp: float,
    ) -> tuple[float, float]:
        location_m = np.asarray(location_ue_cm, dtype=np.float64) / 100.0
        if self.previous_time is None:
            linear_speed = 0.0
            angular_speed = 0.0
        else:
            dt = max(timestamp - self.previous_time, 1e-6)
            linear_speed = float(np.linalg.norm(location_m - self.previous_location_m) / dt)
            angular_speed = float(
                np.linalg.norm(angle_delta_deg(rotation_pyr_deg, self.previous_rotation_deg)) / dt
            )
        self.previous_time = timestamp
        self.previous_location_m = location_m.copy()
        self.previous_rotation_deg = np.asarray(rotation_pyr_deg, dtype=np.float64).copy()
        return linear_speed, angular_speed


def request_same_tick_batch(
    client: Any,
    commands: list[str],
    timeout: float,
) -> list[Any]:
    """Use UnrealCV's vbatch protocol so all commands run in one UE tick."""
    marker = client.request(f"vbatch{len(commands)}", timeout)
    marker_error = _error_text(marker)
    if marker_error and "empty response" not in marker_error.lower():
        raise RuntimeError(f"vbatch{len(commands)}: {marker_error}")
    responses = client.request(commands, timeout)
    if not isinstance(responses, list) or len(responses) != len(commands):
        raise RuntimeError(
            f"UnrealCV vbatch returned "
            f"{len(responses) if isinstance(responses, list) else 'invalid'} responses "
            f"for {len(commands)} commands"
        )
    return responses


def request_pose_conditioned_scans(
    client: Any,
    camera_id: int,
    source_names: list[str],
    timeout: float,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, Any] | None,
    float,
    float,
    float,
]:
    """Capture LiDAR between before/after poses in one Unreal Engine tick."""
    location_command = f"vget /camera/{camera_id}/location"
    rotation_command = f"vget /camera/{camera_id}/rotation"
    commands = [location_command, rotation_command]
    if "TCP NPY" in source_names:
        commands.append(f"vget /camera/{camera_id}/lidar npy")
    if "Shared memory" in source_names:
        commands.append(f"vget /camera/{camera_id}/lidar_shared")
    commands.extend((location_command, rotation_command))

    started = time.perf_counter()
    responses = request_same_tick_batch(client, commands, timeout)
    batch_ms = (time.perf_counter() - started) * 1000.0

    location_before = _parse_vector(responses[0], 3, location_command)
    rotation_before = _parse_vector(responses[1], 3, rotation_command)
    location_after = _parse_vector(responses[-2], 3, location_command)
    rotation_after = _parse_vector(responses[-1], 3, rotation_command)
    location = (location_before + location_after) / 2.0
    rotation_span = angle_delta_deg(rotation_after, rotation_before)
    rotation = rotation_before + rotation_span / 2.0
    pose_span_cm = float(np.linalg.norm(location_after - location_before))
    pose_span_deg = float(np.linalg.norm(rotation_span))
    scans: dict[str, np.ndarray] = {}
    metadata = None
    response_index = 2
    if "TCP NPY" in source_names:
        command = f"vget /camera/{camera_id}/lidar npy"
        scans["TCP NPY"] = decode_npy_response(responses[response_index], command)
        response_index += 1
    if "Shared memory" in source_names:
        command = f"vget /camera/{camera_id}/lidar_shared"
        scans["Shared memory"], metadata = decode_shared_response(
            responses[response_index], command
        )
    return scans, location, rotation, metadata, batch_ms, pose_span_cm, pose_span_deg


def connect(host: str, port: int) -> Any:
    try:
        from unrealcv import Client
    except ImportError as exc:
        raise RuntimeError(
            "The unrealcv Python package is not installed. Run: pip install unrealcv"
        ) from exc

    client = Client((host, port))
    client.connect()
    if not client.isconnected():
        raise ConnectionError(f"Could not connect to UnrealCV at {host}:{port}")
    return client


def downsample(cloud: np.ndarray, max_points: int) -> np.ndarray:
    if cloud.shape[0] <= max_points:
        return cloud
    indices = np.linspace(0, cloud.shape[0] - 1, max_points, dtype=np.int64)
    return cloud[indices]


class VoxelPointCloudMap:
    """Bounded point map that keeps the newest point in each occupied voxel."""

    def __init__(self, voxel_size_m: float, max_points: int):
        self.voxel_size_m = voxel_size_m
        self.max_points = max_points
        self.cloud = np.empty((0, 4), dtype=np.float32)

    def clear(self) -> None:
        self.cloud = np.empty((0, 4), dtype=np.float32)

    def update(self, scan_world: np.ndarray) -> np.ndarray:
        combined = np.concatenate((self.cloud, scan_world), axis=0)
        voxel_indices = np.floor(combined[:, :3] / self.voxel_size_m).astype(np.int64)

        # Search backwards so a newly observed point replaces an older sample
        # in the same voxel. np.unique also prevents unbounded duplicate growth.
        reversed_cloud = combined[::-1]
        reversed_voxels = voxel_indices[::-1]
        _, newest_indices = np.unique(reversed_voxels, axis=0, return_index=True)
        self.cloud = reversed_cloud[newest_indices]
        if self.cloud.shape[0] > self.max_points:
            self.cloud = downsample(self.cloud, self.max_points).copy()
        return self.cloud


class LidarFigure:
    def __init__(self, source_names: list[str], max_points: int):
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        self.plt = plt
        self.max_points = max_points
        self.figure = plt.figure(figsize=(8 * len(source_names), 7))
        self.figure.canvas.manager.set_window_title("UnrealZoo LiDAR 3D Mapping")
        self.axes = [
            self.figure.add_subplot(1, len(source_names), index + 1, projection="3d")
            for index in range(len(source_names))
        ]
        self.source_names = source_names
        self.point_artists: dict[str, Any] = {}
        self.sensor_artists: dict[str, Any] = {}
        self.trajectory_artists: dict[str, Any] = {}
        self.heading_artists: dict[str, Any] = {}
        color_scale = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="viridis")
        color_scale.set_array([])
        self.figure.colorbar(
            color_scale,
            ax=self.axes,
            label="Intensity",
            shrink=0.72,
            pad=0.08,
        )

    def update(
        self,
        captures: dict[str, tuple[np.ndarray, float]],
        tick: int,
        tick_ms: float,
        location_ue_cm: np.ndarray,
        rotation_pyr_deg: np.ndarray,
        trajectory_m: np.ndarray,
    ) -> None:
        sensor_position_m = location_ue_cm / 100.0
        sensor_forward = unreal_rotation_matrix(rotation_pyr_deg)[0]
        for axis, source_name in zip(self.axes, self.source_names):
            cloud, latency_ms = captures[source_name]
            shown = downsample(cloud, self.max_points)
            points = self.point_artists.get(source_name)
            if points is None:
                points = axis.scatter(
                    shown[:, 0],
                    shown[:, 1],
                    shown[:, 2],
                    c=shown[:, 3],
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    s=1.2,
                    linewidths=0,
                )
                self.point_artists[source_name] = points
                self.sensor_artists[source_name] = axis.scatter(
                    [sensor_position_m[0]],
                    [sensor_position_m[1]],
                    [sensor_position_m[2]],
                    c="red",
                    marker="^",
                    s=55,
                    label="current LiDAR pose",
                )
                (self.trajectory_artists[source_name],) = axis.plot(
                    trajectory_m[:, 0],
                    trajectory_m[:, 1],
                    trajectory_m[:, 2],
                    color="red",
                    linewidth=1.5,
                    label="sensor trajectory",
                )
                heading_end = sensor_position_m + sensor_forward
                (self.heading_artists[source_name],) = axis.plot(
                    [sensor_position_m[0], heading_end[0]],
                    [sensor_position_m[1], heading_end[1]],
                    [sensor_position_m[2], heading_end[2]],
                    color="orange",
                    linewidth=2.5,
                    label="sensor forward",
                )
                axis.set_xlabel("World X (m)")
                axis.set_ylabel("World Y (m)")
                axis.set_zlabel("World Z (m)")
                axis.view_init(elev=24, azim=-135)
                axis.set_box_aspect((1, 1, 1))
                axis.grid(True, alpha=0.3)
                axis.legend(loc="upper right")
            else:
                # Reuse the existing 3D collection to avoid flicker and the cost
                # of clearing/rebuilding the axes on every capture tick.
                points._offsets3d = (shown[:, 0], shown[:, 1], shown[:, 2])
                points.set_array(shown[:, 3])
                sensor = self.sensor_artists[source_name]
                sensor._offsets3d = (
                    [sensor_position_m[0]],
                    [sensor_position_m[1]],
                    [sensor_position_m[2]],
                )
                trajectory = self.trajectory_artists[source_name]
                trajectory.set_data(trajectory_m[:, 0], trajectory_m[:, 1])
                trajectory.set_3d_properties(trajectory_m[:, 2])
                heading_end = sensor_position_m + sensor_forward
                heading = self.heading_artists[source_name]
                heading.set_data(
                    [sensor_position_m[0], heading_end[0]],
                    [sensor_position_m[1], heading_end[1]],
                )
                heading.set_3d_properties([sensor_position_m[2], heading_end[2]])
            axis.set_title(
                f"tick {tick:,} | pose-conditioned {source_name} map | "
                f"{cloud.shape[0]:,} voxels\n"
                f"pose xyz=({sensor_position_m[0]:.2f}, {sensor_position_m[1]:.2f}, "
                f"{sensor_position_m[2]:.2f}) m | batch {latency_ms:.1f} ms | "
                f"update {tick_ms:.1f} ms"
            )

            xyz = np.concatenate((shown[:, :3], trajectory_m, sensor_position_m[None, :]), axis=0)
            lower = xyz.min(axis=0)
            upper = xyz.max(axis=0)
            center = (lower + upper) / 2.0
            radius = max(float((upper - lower).max()) / 2.0, 0.5)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
        self.figure.canvas.draw_idle()


class FirstPersonLidarFigure:
    """Camera-like angular projection of the accumulated map at the live pose."""

    def __init__(
        self,
        source_names: list[str],
        max_points: int,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
        near_m: float,
        far_m: float,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LogNorm

        self.plt = plt
        self.max_points = max_points
        self.horizontal_fov_deg = horizontal_fov_deg
        self.vertical_fov_deg = vertical_fov_deg
        self.near_m = near_m
        self.far_m = far_m
        self.depth_norm = LogNorm(vmin=max(near_m, 0.05), vmax=far_m)
        self.figure, axes = plt.subplots(
            1,
            len(source_names),
            figsize=(10 * len(source_names), 7),
            squeeze=False,
        )
        self.figure.canvas.manager.set_window_title("UnrealZoo LiDAR First Person")
        self.axes = list(axes[0])
        self.source_names = source_names
        self.point_artists: dict[str, Any] = {}

        for axis in self.axes:
            axis.set_facecolor("#05070a")
            axis.set_xlim(-horizontal_fov_deg / 2.0, horizontal_fov_deg / 2.0)
            axis.set_ylim(-vertical_fov_deg / 2.0, vertical_fov_deg / 2.0)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("horizontal angle (deg)")
            axis.set_ylabel("vertical angle (deg)")
            axis.axhline(0.0, color="white", linewidth=0.5, alpha=0.3)
            axis.axvline(0.0, color="white", linewidth=0.5, alpha=0.3)
            axis.tick_params(colors="white")
            axis.xaxis.label.set_color("white")
            axis.yaxis.label.set_color("white")
            for spine in axis.spines.values():
                spine.set_color("#75808f")

        depth_scale = ScalarMappable(
            norm=self.depth_norm,
            cmap="turbo_r",
        )
        depth_scale.set_array([])
        colorbar = self.figure.colorbar(
            depth_scale,
            ax=self.axes,
            label="forward depth (m)",
            shrink=0.78,
            pad=0.04,
        )
        colorbar.ax.tick_params(colors="#20242a")

    def _project_visible_points(
        self,
        cloud_world: np.ndarray,
        location_ue_cm: np.ndarray,
        rotation_pyr_deg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sensor_position_m = location_ue_cm / 100.0
        rotation = unreal_rotation_matrix(rotation_pyr_deg)
        local_xyz = (cloud_world[:, :3] - sensor_position_m) @ rotation.T

        forward = local_xyz[:, 0]
        horizontal_deg = np.rad2deg(np.arctan2(local_xyz[:, 1], forward))
        vertical_deg = np.rad2deg(
            np.arctan2(local_xyz[:, 2], np.hypot(forward, local_xyz[:, 1]))
        )
        visible = (
            (forward >= self.near_m)
            & (forward <= self.far_m)
            & (np.abs(horizontal_deg) <= self.horizontal_fov_deg / 2.0)
            & (np.abs(vertical_deg) <= self.vertical_fov_deg / 2.0)
        )
        horizontal_deg = horizontal_deg[visible]
        vertical_deg = vertical_deg[visible]
        forward = forward[visible]

        if forward.size > self.max_points:
            indices = np.linspace(0, forward.size - 1, self.max_points, dtype=np.int64)
            horizontal_deg = horizontal_deg[indices]
            vertical_deg = vertical_deg[indices]
            forward = forward[indices]

        # Draw far points first so nearby surfaces remain visible on top.
        order = np.argsort(forward)[::-1]
        return horizontal_deg[order], vertical_deg[order], forward[order]

    def update(
        self,
        captures: dict[str, tuple[np.ndarray, float]],
        tick: int,
        tick_ms: float,
        location_ue_cm: np.ndarray,
        rotation_pyr_deg: np.ndarray,
        trajectory_m: np.ndarray,
    ) -> None:
        del trajectory_m  # The trajectory is retained in the map but not overlaid in first person.
        sensor_position_m = location_ue_cm / 100.0
        for axis, source_name in zip(self.axes, self.source_names):
            cloud_world, latency_ms = captures[source_name]
            horizontal, vertical, depth = self._project_visible_points(
                cloud_world,
                location_ue_cm,
                rotation_pyr_deg,
            )
            screen_points = np.column_stack((horizontal, vertical))
            # Angular size approximates perspective: nearby returns appear larger.
            sizes = np.clip(16.0 / np.maximum(depth, self.near_m), 0.25, 7.0)

            points = self.point_artists.get(source_name)
            if points is None:
                points = axis.scatter(
                    horizontal,
                    vertical,
                    c=depth,
                    s=sizes,
                    cmap="turbo_r",
                    norm=self.depth_norm,
                    linewidths=0,
                    alpha=0.9,
                )
                self.point_artists[source_name] = points
            else:
                points.set_offsets(screen_points)
                points.set_array(depth)
                points.set_sizes(sizes)

            axis.set_title(
                f"tick {tick:,} | first-person {source_name} map | "
                f"visible {depth.size:,}/{cloud_world.shape[0]:,}\n"
                f"pose=({sensor_position_m[0]:.2f}, {sensor_position_m[1]:.2f}, "
                f"{sensor_position_m[2]:.2f}) m | "
                f"pyr=({rotation_pyr_deg[0]:.1f}, {rotation_pyr_deg[1]:.1f}, "
                f"{rotation_pyr_deg[2]:.1f}) deg | batch {latency_ms:.1f} ms | "
                f"update {tick_ms:.1f} ms",
                color="#111820",
            )
        self.figure.canvas.draw_idle()


class KeyboardCameraController:
    """Track held Matplotlib keys and produce smooth camera pose updates."""

    def __init__(self, figure: Any, move_speed_m_s: float, turn_speed_deg_s: float):
        self.move_speed_m_s = move_speed_m_s
        self.turn_speed_deg_s = turn_speed_deg_s
        self.pressed: set[str] = set()
        self.clear_requested = False
        self.paused = False
        self.quit_requested = False
        figure.canvas.mpl_connect("key_press_event", self._on_press)
        figure.canvas.mpl_connect("key_release_event", self._on_release)

    @staticmethod
    def _key(event: Any) -> str:
        return str(event.key or "").lower()

    def _on_press(self, event: Any) -> None:
        key = self._key(event)
        if key == "c":
            self.clear_requested = True
        elif key == "p":
            self.paused = not self.paused
        elif key == "escape":
            self.quit_requested = True
        else:
            self.pressed.add(key)

    def _on_release(self, event: Any) -> None:
        self.pressed.discard(self._key(event))

    def consume_clear(self) -> bool:
        requested = self.clear_requested
        self.clear_requested = False
        return requested

    def _held(self, key: str) -> bool:
        return key in self.pressed or f"shift+{key}" in self.pressed

    def next_pose(
        self,
        location_ue_cm: np.ndarray,
        rotation_pyr_deg: np.ndarray,
        delta_seconds: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        forward = float(self._held("w")) - float(self._held("s"))
        right = float(self._held("d")) - float(self._held("a"))
        up = float(self._held("pageup")) - float(self._held("pagedown"))
        yaw_input = float(self._held("right")) - float(self._held("left"))
        pitch_input = float(self._held("up")) - float(self._held("down"))
        if forward == right == up == yaw_input == pitch_input == 0.0:
            return None

        boost = (
            3.0
            if "shift" in self.pressed
            or any(key.startswith("shift+") for key in self.pressed)
            else 1.0
        )
        dt = min(max(delta_seconds, 0.0), 0.1)
        new_rotation = rotation_pyr_deg.astype(np.float64, copy=True)
        new_rotation[0] = np.clip(
            new_rotation[0] + pitch_input * self.turn_speed_deg_s * dt,
            -89.0,
            89.0,
        )
        new_rotation[1] = (
            new_rotation[1] + yaw_input * self.turn_speed_deg_s * dt + 180.0
        ) % 360.0 - 180.0

        local_motion = np.asarray([forward, right, up], dtype=np.float64)
        motion_norm = np.linalg.norm(local_motion)
        new_location = location_ue_cm.astype(np.float64, copy=True)
        if motion_norm > 0.0:
            local_motion /= max(motion_norm, 1.0)
            # Navigation translation follows yaw but remains level; PageUp and
            # PageDown always move along world Z for predictable fly controls.
            yaw_rotation = unreal_rotation_matrix(np.asarray([0.0, new_rotation[1], 0.0]))
            horizontal_world = np.asarray([local_motion[0], local_motion[1], 0.0]) @ yaw_rotation
            horizontal_world[2] = local_motion[2]
            new_location += horizontal_world * self.move_speed_m_s * boost * dt * 100.0
        return new_location, new_rotation


def set_camera_pose_async(
    client: Any,
    camera_id: int,
    location_ue_cm: np.ndarray,
    rotation_pyr_deg: np.ndarray,
) -> None:
    x, y, z = location_ue_cm
    pitch, yaw, roll = rotation_pyr_deg
    client.request(
        [
            f"vset /camera/{camera_id}/location {x:.4f} {y:.4f} {z:.4f}",
            f"vset /camera/{camera_id}/rotation {pitch:.4f} {yaw:.4f} {roll:.4f}",
        ],
        -1,
    )


class KeyboardMappingFigure:
    """Current first-person scan beside a clean top-down accumulated map."""

    CONTROLS = (
        "W/S forward/back  A/D strafe  PageUp/PageDown vertical  "
        "Arrows look  Shift boost  C clear map  P pause mapping  Esc quit"
    )
    EXTERNAL_CONTROL_HINT = (
        "Camera control: Unreal/PIE window | Python reads synchronized pose + LiDAR only"
    )

    def __init__(
        self,
        max_points: int,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
        near_m: float,
        far_m: float,
        slice_height_m: float,
        control_hint: str,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LogNorm

        # Avoid toolbar shortcuts consuming camera-control keys.
        for keymap_name in (
            "keymap.back",
            "keymap.forward",
            "keymap.pan",
            "keymap.quit",
            "keymap.home",
            "keymap.zoom",
        ):
            plt.rcParams[keymap_name] = []

        self.plt = plt
        self.max_points = max_points
        self.horizontal_fov_deg = horizontal_fov_deg
        self.vertical_fov_deg = vertical_fov_deg
        self.near_m = near_m
        self.far_m = far_m
        self.slice_height_m = slice_height_m
        self.depth_norm = LogNorm(vmin=max(near_m, 0.05), vmax=far_m)
        self.map_origin_m: np.ndarray | None = None
        self.map_bounds: tuple[float, float, float, float] | None = None
        self.figure, (self.scan_axis, self.map_axis) = plt.subplots(
            1,
            2,
            figsize=(16, 7.5),
            gridspec_kw={"width_ratios": [1.15, 1.0]},
        )
        self.figure.canvas.manager.set_window_title("UnrealZoo LiDAR Voxel Mapping")
        self.figure.subplots_adjust(bottom=0.12, wspace=0.16)
        self.figure.text(0.5, 0.035, control_hint, ha="center", fontsize=10)

        self.scan_axis.set_facecolor("#05070a")
        self.scan_axis.set_xlim(-horizontal_fov_deg / 2.0, horizontal_fov_deg / 2.0)
        self.scan_axis.set_ylim(-vertical_fov_deg / 2.0, vertical_fov_deg / 2.0)
        self.scan_axis.set_aspect("equal", adjustable="box")
        self.scan_axis.set_xlabel("horizontal angle (deg)")
        self.scan_axis.set_ylabel("vertical angle (deg)")
        self.scan_axis.axhline(0.0, color="white", linewidth=0.5, alpha=0.3)
        self.scan_axis.axvline(0.0, color="white", linewidth=0.5, alpha=0.3)

        self.map_axis.set_facecolor("#081016")
        self.map_axis.set_aspect("equal", adjustable="box")
        self.map_axis.set_xlabel("X from start (m)")
        self.map_axis.set_ylabel("Y from start (m)")
        self.map_axis.grid(True, color="white", alpha=0.10, linewidth=0.5)

        depth_scale = ScalarMappable(norm=self.depth_norm, cmap="turbo_r")
        depth_scale.set_array([])
        self.figure.colorbar(
            depth_scale,
            ax=self.scan_axis,
            label="forward depth (m)",
            shrink=0.78,
            pad=0.03,
        )
        self.scan_points = self.scan_axis.scatter(
            [], [], c=[], s=[], cmap="turbo_r", norm=self.depth_norm, linewidths=0, alpha=0.9
        )
        self.map_points = self.map_axis.scatter(
            [], [], c="#9be7ff", s=0.8, linewidths=0, alpha=0.65
        )
        (self.trajectory_line,) = self.map_axis.plot(
            [], [], color="#ff5252", linewidth=1.6, label="trajectory"
        )
        self.sensor_point = self.map_axis.scatter(
            [], [], c="#ff3d00", marker="^", s=65, label="camera"
        )
        (self.heading_line,) = self.map_axis.plot(
            [], [], color="#ffc107", linewidth=2.5, label="forward"
        )
        self.map_axis.legend(loc="upper right")

    def reset_map_view(self) -> None:
        self.map_origin_m = None
        self.map_bounds = None
        self.map_points.set_offsets(np.empty((0, 2)))
        self.trajectory_line.set_data([], [])
        self.sensor_point.set_offsets(np.empty((0, 2)))
        self.heading_line.set_data([], [])

    def _project_current_scan(self, scan_local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xyz = scan_local[:, :3]
        forward = xyz[:, 0]
        horizontal = np.rad2deg(np.arctan2(xyz[:, 1], forward))
        vertical = np.rad2deg(np.arctan2(xyz[:, 2], np.hypot(forward, xyz[:, 1])))
        visible = (
            (forward >= self.near_m)
            & (forward <= self.far_m)
            & (np.abs(horizontal) <= self.horizontal_fov_deg / 2.0)
            & (np.abs(vertical) <= self.vertical_fov_deg / 2.0)
        )
        horizontal, vertical, forward = horizontal[visible], vertical[visible], forward[visible]
        if forward.size > self.max_points:
            indices = np.linspace(0, forward.size - 1, self.max_points, dtype=np.int64)
            horizontal, vertical, forward = horizontal[indices], vertical[indices], forward[indices]
        order = np.argsort(forward)[::-1]
        return horizontal[order], vertical[order], forward[order]

    def _expand_map_limits(self, xy: np.ndarray) -> None:
        if xy.size == 0:
            return
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        if self.map_bounds is None:
            self.map_bounds = (xmin, xmax, ymin, ymax)
        else:
            old_xmin, old_xmax, old_ymin, old_ymax = self.map_bounds
            self.map_bounds = (
                min(old_xmin, xmin),
                max(old_xmax, xmax),
                min(old_ymin, ymin),
                max(old_ymax, ymax),
            )
        xmin, xmax, ymin, ymax = self.map_bounds
        x_center, y_center = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        span = max(xmax - xmin, ymax - ymin, 8.0) + 3.0
        self.map_axis.set_xlim(x_center - span / 2.0, x_center + span / 2.0)
        self.map_axis.set_ylim(y_center - span / 2.0, y_center + span / 2.0)

    def update(
        self,
        scan_local: np.ndarray,
        map_world: np.ndarray,
        tick: int,
        tick_ms: float,
        batch_ms: float,
        location_ue_cm: np.ndarray,
        rotation_pyr_deg: np.ndarray,
        trajectory_m: np.ndarray,
        paused: bool,
        linear_speed_m_s: float,
        angular_speed_deg_s: float,
        pose_span_cm: float,
        pose_span_deg: float,
        frame_accepted: bool,
    ) -> None:
        position_m = location_ue_cm / 100.0
        if self.map_origin_m is None:
            self.map_origin_m = position_m.copy()

        horizontal, vertical, depth = self._project_current_scan(scan_local)
        self.scan_points.set_offsets(np.column_stack((horizontal, vertical)))
        self.scan_points.set_array(depth)
        self.scan_points.set_sizes(np.clip(16.0 / np.maximum(depth, self.near_m), 0.3, 7.0))
        self.scan_axis.set_title(
            f"Current LiDAR frame | visible {depth.size:,}/{scan_local.shape[0]:,}\n"
            f"v={linear_speed_m_s:.2f} m/s | w={angular_speed_deg_s:.1f} deg/s | "
            f"batch {batch_ms:.1f} ms"
        )

        slice_center_z = self.map_origin_m[2]
        in_slice = np.abs(map_world[:, 2] - slice_center_z) <= self.slice_height_m / 2.0
        map_slice = map_world[in_slice]
        map_slice = downsample(map_slice, self.max_points)
        map_xy = map_slice[:, :2] - self.map_origin_m[:2]
        current_xy = position_m[:2] - self.map_origin_m[:2]
        trajectory_xy = trajectory_m[:, :2] - self.map_origin_m[:2]
        self.map_points.set_offsets(map_xy)
        self.trajectory_line.set_data(trajectory_xy[:, 0], trajectory_xy[:, 1])
        self.sensor_point.set_offsets(current_xy[None, :])
        forward_xy = unreal_rotation_matrix(rotation_pyr_deg)[0, :2]
        self.heading_line.set_data(
            [current_xy[0], current_xy[0] + forward_xy[0]],
            [current_xy[1], current_xy[1] + forward_xy[1]],
        )
        limit_points = np.concatenate((map_xy, trajectory_xy, current_xy[None, :]), axis=0)
        self._expand_map_limits(limit_points)
        if not frame_accepted:
            state = "POSE SYNC REJECTED"
        elif paused:
            state = "PAUSED"
        else:
            state = "mapping"
        self.map_axis.set_title(
            f"Top-down voxel map | {state} | slice {self.slice_height_m:.2f} m\n"
            f"shown {map_slice.shape[0]:,}/{map_world.shape[0]:,} voxels | "
            f"sync span={pose_span_cm:.3f} cm/{pose_span_deg:.3f} deg | "
            f"tick {tick:,} | update {tick_ms:.1f} ms"
        )
        self.figure.canvas.draw_idle()


def summarize(
    source: str,
    scan_local: np.ndarray,
    map_world: np.ndarray,
    location_ue_cm: np.ndarray,
    rotation_pyr_deg: np.ndarray,
    latency_ms: float,
) -> None:
    ranges = np.linalg.norm(scan_local[:, :3], axis=1)
    location_m = location_ue_cm / 100.0
    print(
        f"{source}: scan={scan_local.shape[0]:,}, map={map_world.shape[0]:,}, "
        f"range={ranges.min():.2f}..{ranges.max():.2f} m, "
        f"pose_m={location_m.round(3).tolist()}, "
        f"pyr_deg={rotation_pyr_deg.round(2).tolist()}, batch={latency_ms:.1f} ms"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="UnrealCV server address")
    parser.add_argument("--port", type=int, default=9000, help="UnrealCV server port")
    parser.add_argument("--camera-id", type=int, default=1, help="Fusion camera ID (default: 1)")
    parser.add_argument(
        "--transport",
        choices=("npy", "shared", "both"),
        default="shared",
        help="LiDAR transfer mode to map (default: shared)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Extra delay after each live capture in seconds (default: 0, no delay)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture one tick and stop instead of continuously updating",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50_000,
        help="Maximum rendered map points per panel (default: 50000)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.10,
        help="Accumulated map voxel size in metres (default: 0.10)",
    )
    parser.add_argument(
        "--max-map-points",
        type=int,
        default=250_000,
        help="Maximum retained voxel points per map (default: 250000)",
    )
    parser.add_argument(
        "--view",
        choices=("mapping", "first-person", "world"),
        default="mapping",
        help="Visualization layout (default: mapping)",
    )
    parser.add_argument(
        "--horizontal-fov",
        type=float,
        default=100.0,
        help="First-person horizontal field of view in degrees (default: 100)",
    )
    parser.add_argument(
        "--vertical-fov",
        type=float,
        default=70.0,
        help="First-person vertical field of view in degrees (default: 70)",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.10,
        help="First-person near clipping distance in metres (default: 0.10)",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=50.0,
        help="First-person far clipping distance in metres (default: 50)",
    )
    parser.add_argument(
        "--slice-height",
        type=float,
        default=0.20,
        help="Vertical thickness of the top-down map slice in metres (default: 0.20)",
    )
    parser.add_argument(
        "--move-speed",
        type=float,
        default=2.0,
        help="Keyboard camera translation speed in metres/second (default: 2.0)",
    )
    parser.add_argument(
        "--turn-speed",
        type=float,
        default=75.0,
        help="Keyboard camera look speed in degrees/second (default: 75)",
    )
    parser.add_argument(
        "--control-mode",
        choices=("external", "plot"),
        default="external",
        help=(
            "Camera control source: Unreal/PIE keyboard or this plot window "
            "(default: external)"
        ),
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable plot-window camera controls (legacy alias)",
    )
    parser.add_argument(
        "--max-pose-span-cm",
        type=float,
        default=0.1,
        help="Reject a scan if same-tick before/after position differs more than this (default: 0.1)",
    )
    parser.add_argument(
        "--max-pose-span-deg",
        type=float,
        default=0.1,
        help="Reject a scan if same-tick before/after rotation differs more than this (default: 0.1)",
    )
    parser.add_argument("--save", type=Path, help="Also save the latest figure as a PNG")
    parser.add_argument("--no-show", action="store_true", help="Do not open a Matplotlib window")
    parser.add_argument(
        "--print-metadata",
        action="store_true",
        help="Print lidar_shared JSON metadata",
    )
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    if args.max_points <= 0:
        parser.error("--max-points must be positive")
    if args.voxel_size <= 0:
        parser.error("--voxel-size must be positive")
    if args.max_map_points <= 0:
        parser.error("--max-map-points must be positive")
    if not 0 < args.horizontal_fov < 360:
        parser.error("--horizontal-fov must be between 0 and 360 degrees")
    if not 0 < args.vertical_fov < 180:
        parser.error("--vertical-fov must be between 0 and 180 degrees")
    if args.near < 0 or args.far <= args.near:
        parser.error("--far must be greater than a non-negative --near")
    if args.slice_height <= 0:
        parser.error("--slice-height must be positive")
    if args.move_speed <= 0 or args.turn_speed <= 0:
        parser.error("--move-speed and --turn-speed must be positive")
    if args.max_pose_span_cm < 0 or args.max_pose_span_deg < 0:
        parser.error("pose-span thresholds must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    source_names = {
        "npy": ["TCP NPY"],
        "shared": ["Shared memory"],
        "both": ["TCP NPY", "Shared memory"],
    }[args.transport]
    if args.no_show and args.save is None:
        figure = None
    elif args.view == "mapping":
        control_hint = (
            KeyboardMappingFigure.CONTROLS
            if args.control_mode == "plot" and not args.no_keyboard
            else KeyboardMappingFigure.EXTERNAL_CONTROL_HINT
        )
        figure = KeyboardMappingFigure(
            args.max_points,
            args.horizontal_fov,
            args.vertical_fov,
            args.near,
            args.far,
            args.slice_height,
            control_hint,
        )
    elif args.view == "first-person":
        figure = FirstPersonLidarFigure(
            source_names,
            args.max_points,
            args.horizontal_fov,
            args.vertical_fov,
            args.near,
            args.far,
        )
    else:
        figure = LidarFigure(source_names, args.max_points)
    controller = None
    if (
        figure is not None
        and args.view == "mapping"
        and not args.no_show
        and not args.once
        and not args.no_keyboard
        and args.control_mode == "plot"
    ):
        controller = KeyboardCameraController(
            figure.figure,
            args.move_speed,
            args.turn_speed,
        )
    point_maps = {
        source_name: VoxelPointCloudMap(args.voxel_size, args.max_map_points)
        for source_name in source_names
    }
    trajectory_positions_m: list[np.ndarray] = []
    client = connect(args.host, args.port)
    print(f"Connected to UnrealCV at {args.host}:{args.port} (camera {args.camera_id})")
    if args.once:
        print("Single-tick capture started")
    else:
        print("Live capture started; close the figure or press Ctrl+C to stop")
    if controller is not None:
        print(KeyboardMappingFigure.CONTROLS)
    elif args.control_mode == "external":
        print(KeyboardMappingFigure.EXTERNAL_CONTROL_HINT)

    try:
        tick = 0
        last_control_time = time.perf_counter()
        motion_estimator = PoseMotionEstimator()
        while True:
            tick += 1
            tick_started = time.perf_counter()
            captures: dict[str, tuple[np.ndarray, float]] = {}
            (
                scans_local,
                location,
                rotation,
                metadata,
                batch_ms,
                pose_span_cm,
                pose_span_deg,
            ) = request_pose_conditioned_scans(
                client, args.camera_id, source_names, args.timeout
            )
            linear_speed_m_s, angular_speed_deg_s = motion_estimator.update(
                location, rotation, time.perf_counter()
            )
            frame_accepted = (
                pose_span_cm <= args.max_pose_span_cm
                and pose_span_deg <= args.max_pose_span_deg
            )
            if not frame_accepted:
                print(
                    f"Rejected tick {tick}: same-tick pose span "
                    f"{pose_span_cm:.4f} cm / {pose_span_deg:.4f} deg"
                )
            if controller is not None and controller.consume_clear():
                for point_map in point_maps.values():
                    point_map.clear()
                trajectory_positions_m.clear()
                if isinstance(figure, KeyboardMappingFigure):
                    figure.reset_map_view()
                print("Point-cloud map cleared")
            trajectory_positions_m.append(location / 100.0)
            trajectory_m = np.asarray(trajectory_positions_m, dtype=np.float64)

            for source_name, scan_local in scans_local.items():
                scan_world = transform_cloud_to_world(scan_local, location, rotation)
                if not frame_accepted or (controller is not None and controller.paused):
                    map_world = point_maps[source_name].cloud
                else:
                    map_world = point_maps[source_name].update(scan_world)
                captures[source_name] = (map_world, batch_ms)
                if tick == 1 or tick % 30 == 0:
                    summarize(
                        source_name,
                        scan_local,
                        map_world,
                        location,
                        rotation,
                        batch_ms,
                    )
            if args.print_metadata and tick == 1 and metadata is not None:
                print(json.dumps(metadata, indent=2, ensure_ascii=False))

            tick_ms = (time.perf_counter() - tick_started) * 1000.0
            if figure is not None:
                if args.view == "mapping":
                    mapping_source = (
                        "Shared memory" if "Shared memory" in source_names else source_names[0]
                    )
                    figure.update(
                        scans_local[mapping_source],
                        captures[mapping_source][0],
                        tick,
                        tick_ms,
                        batch_ms,
                        location,
                        rotation,
                        trajectory_m,
                        controller.paused if controller is not None else False,
                        linear_speed_m_s,
                        angular_speed_deg_s,
                        pose_span_cm,
                        pose_span_deg,
                        frame_accepted,
                    )
                else:
                    figure.update(
                        captures,
                        tick,
                        tick_ms,
                        location,
                        rotation,
                        trajectory_m,
                    )
                if args.save:
                    args.save.parent.mkdir(parents=True, exist_ok=True)
                    figure.figure.savefig(args.save, dpi=160, bbox_inches="tight")
                    if tick == 1 or tick % 30 == 0:
                        print(f"Saved latest visualization to {args.save.resolve()}")

            if args.once:
                if figure is not None and not args.no_show:
                    figure.plt.show()
                break
            if figure is not None and not args.no_show:
                figure.plt.show(block=False)
                # Even with no requested delay, Matplotlib needs a short event
                # pump so the window can repaint and process close events.
                figure.plt.pause(max(args.interval, 0.001))
                if not figure.plt.fignum_exists(figure.figure.number):
                    break
            elif args.interval > 0.0:
                time.sleep(args.interval)

            now = time.perf_counter()
            if controller is not None:
                if controller.quit_requested:
                    break
                next_pose = controller.next_pose(location, rotation, now - last_control_time)
                if next_pose is not None:
                    set_camera_pose_async(client, args.camera_id, *next_pose)
            last_control_time = now
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
