"""GPU-instanced four-view renderer for live UnrealCV scene occupancy."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque

import glfw
import moderngl
import numpy as np

from realtime_scene_occupancy import (
    make_command,
    open_shared_memory,
    parse_vector,
    unrealcv,
)


VERTEX_SHADER = """
#version 330
uniform mat4 projection;
uniform mat4 view;
uniform vec3 voxel_size;
in vec3 in_position;
in vec3 in_normal;
in vec3 in_offset;
in float in_height;
out vec3 normal;
out float height_value;
void main() {
    vec3 world_position = in_offset + in_position * voxel_size;
    gl_Position = projection * view * vec4(world_position, 1.0);
    normal = in_normal;
    height_value = in_height;
}
"""


FRAGMENT_SHADER = """
#version 330
in vec3 normal;
in float height_value;
out vec4 frag_color;
vec3 height_color(float t) {
    return clamp(vec3(
        1.5 - abs(4.0 * t - 3.0),
        1.5 - abs(4.0 * t - 2.0),
        1.5 - abs(4.0 * t - 1.0)
    ), 0.0, 1.0);
}
void main() {
    vec3 light_direction = normalize(vec3(0.35, 0.85, 0.45));
    float diffuse = 0.38 + 0.62 * abs(dot(normalize(normal), light_direction));
    frag_color = vec4(height_color(height_value) * diffuse, 1.0);
}
"""


VIEW_DIRECTIONS = (
    np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
    np.asarray((-1.0, 0.0, 0.0), dtype=np.float32),
    np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
    np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU live mesh occupancy viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--camera-id", default="0")
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--extent-m", type=float, default=10.0)
    parser.add_argument("--voxel-size-m", type=float, default=0.3)
    parser.add_argument("--method", choices=("mesh", "bounds"), default="mesh")
    parser.add_argument("--include-dynamic", action="store_true")
    parser.add_argument("--origin-cm", nargs=3, type=float)
    parser.add_argument("--yaw-degrees", type=float)
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=1000)
    parser.add_argument("--surface-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def cube_vertices() -> np.ndarray:
    faces = (
        ((1, 0, 0), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
        ((-1, 0, 0), ((-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, -1, -1))),
        ((0, 1, 0), ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
        ((0, -1, 0), ((-1, -1, 1), (-1, -1, -1), (1, -1, -1), (1, -1, 1))),
        ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
        ((0, 0, -1), ((1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1))),
    )
    result = []
    for normal, corners in faces:
        for corner_index in (0, 1, 2, 0, 2, 3):
            position = np.asarray(corners[corner_index], dtype=np.float32) * 0.5
            result.append((*position, *normal))
    return np.asarray(result, dtype=np.float32)


def perspective(field_of_view: float, aspect: float, near: float, far: float) -> np.ndarray:
    scale = 1.0 / np.tan(np.deg2rad(field_of_view) * 0.5)
    return np.asarray(
        (
            (scale / aspect, 0, 0, 0),
            (0, scale, 0, 0),
            (0, 0, (far + near) / (near - far), 2 * far * near / (near - far)),
            (0, 0, -1, 0),
        ),
        dtype=np.float32,
    )


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    result = np.eye(4, dtype=np.float32)
    result[0, :3] = right
    result[1, :3] = up
    result[2, :3] = -forward
    result[0, 3] = -np.dot(right, eye)
    result[1, 3] = -np.dot(up, eye)
    result[2, 3] = np.dot(forward, eye)
    return result


def occupancy_instances(metadata: dict, shared, surface_only: bool) -> tuple[np.ndarray, np.ndarray, int]:
    shape = tuple(int(value) for value in metadata["shape"])
    grid = np.ndarray(
        shape,
        dtype=np.bool_,
        buffer=shared,
        offset=int(metadata["offset_bytes"]),
        order="C",
    ).copy()
    occupied_count = int(np.count_nonzero(grid))
    if surface_only and all(size > 2 for size in shape):
        interior = (
            grid[1:-1, 1:-1, 1:-1]
            & grid[:-2, 1:-1, 1:-1]
            & grid[2:, 1:-1, 1:-1]
            & grid[1:-1, :-2, 1:-1]
            & grid[1:-1, 2:, 1:-1]
            & grid[1:-1, 1:-1, :-2]
            & grid[1:-1, 1:-1, 2:]
        )
        grid[1:-1, 1:-1, 1:-1] &= ~interior
    indices = np.argwhere(grid).astype(np.float32)
    minimum = np.asarray(metadata["min_meters"], dtype=np.float32)
    maximum = np.asarray(metadata["max_meters"], dtype=np.float32)
    voxel_size = (maximum - minimum) / np.asarray(shape, dtype=np.float32)
    positions = minimum + (indices + 0.5) * voxel_size
    heights = np.clip((positions[:, 1] - minimum[1]) / (maximum[1] - minimum[1]), 0, 1)
    instances = np.column_stack((positions, heights)).astype(np.float32)
    return instances, voxel_size, occupied_count


def matrix_bytes(matrix: np.ndarray) -> bytes:
    return np.ascontiguousarray(matrix.T, dtype=np.float32).tobytes()


def main() -> int:
    args = parse_args()
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(
        args.window_width, args.window_height, "UnrealCV GPU Occupancy", None, None
    )
    if not window:
        glfw.terminate()
        raise RuntimeError("Unable to create the OpenGL window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    context = moderngl.create_context()
    context.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    program = context.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    vertex_buffer = context.buffer(cube_vertices().tobytes())
    max_instances = int(np.ceil(args.extent_m / args.voxel_size_m)) ** 3
    instance_buffer = context.buffer(reserve=max_instances * 4 * 4, dynamic=True)
    vertex_array = context.vertex_array(
        program,
        (
            (vertex_buffer, "3f 3f", "in_position", "in_normal"),
            (instance_buffer, "3f 1f /i", "in_offset", "in_height"),
        ),
    )

    client = unrealcv.Client((args.host, args.port))
    if not client.connect():
        raise ConnectionError(f"Unable to connect to UnrealCV at {args.host}:{args.port}")
    frame_times: deque[float] = deque(maxlen=60)
    period = 1.0 / args.target_fps
    frame_index = 0
    try:
        while not glfw.window_should_close(window):
            frame_start = time.perf_counter()
            origin_cm = tuple(args.origin_cm) if args.origin_cm else parse_vector(
                client.request(f"vget /camera/{args.camera_id}/location", timeout=10),
                "location",
            )
            if args.yaw_degrees is None:
                rotation = parse_vector(
                    client.request(f"vget /camera/{args.camera_id}/rotation", timeout=10),
                    "rotation",
                )
                yaw_degrees = rotation[1]
            else:
                yaw_degrees = args.yaw_degrees
            response = client.request(make_command(args, origin_cm, yaw_degrees), timeout=30)
            metadata = json.loads(str(response))
            shared = open_shared_memory(metadata)
            try:
                instances, voxel_size, occupied_count = occupancy_instances(
                    metadata, shared, args.surface_only
                )
            finally:
                shared.close()
            instance_buffer.write(instances.tobytes())
            program["voxel_size"].value = tuple(float(value) for value in voxel_size)

            width, height = glfw.get_framebuffer_size(window)
            gap = 3
            panel_width, panel_height = (width - gap) // 2, (height - gap) // 2
            projection = perspective(48.0, panel_width / panel_height, 0.1, 100.0)
            program["projection"].write(matrix_bytes(projection))
            actual_extent = float(max(np.asarray(metadata["max_meters"]) - np.asarray(metadata["min_meters"])))
            center = np.zeros(3, dtype=np.float32)
            elevation = np.deg2rad(20.0)
            viewports = (
                (0, panel_height + gap, panel_width, panel_height),
                (panel_width + gap, panel_height + gap, panel_width, panel_height),
                (0, 0, panel_width, panel_height),
                (panel_width + gap, 0, panel_width, panel_height),
            )
            context.clear(0.08, 0.08, 0.09, 1.0)
            for viewport, horizontal in zip(viewports, VIEW_DIRECTIONS):
                direction = horizontal * np.cos(elevation)
                direction = direction.copy()
                direction[1] = -np.sin(elevation)
                eye = center - direction * (actual_extent * 1.65)
                program["view"].write(matrix_bytes(look_at(eye, center)))
                context.viewport = viewport
                context.scissor = viewport
                context.clear(0.96, 0.96, 0.97, 1.0, depth=1.0)
                vertex_array.render(instances=len(instances))
            context.scissor = None
            glfw.swap_buffers(window)
            glfw.poll_events()

            elapsed = time.perf_counter() - frame_start
            if elapsed < period:
                time.sleep(period - elapsed)
            frame_times.append(time.perf_counter() - frame_start)
            fps = len(frame_times) / sum(frame_times)
            frame_index += 1
            glfw.set_window_title(
                window,
                f"UnrealCV GPU Occupancy | {fps:.1f}/{args.target_fps:g} FPS | "
                f"occupied {occupied_count:,} | rendered {len(instances):,} | "
                "views: forward, backward, left, right",
            )
            if frame_index == 1 or frame_index % 30 == 0 or (
                args.max_frames and frame_index >= args.max_frames
            ):
                print(
                    f"frame={frame_index} loop_fps={fps:.2f} "
                    f"occupied={occupied_count} rendered={len(instances)}",
                    flush=True,
                )
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        client.disconnect()
        glfw.destroy_window(window)
        glfw.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
