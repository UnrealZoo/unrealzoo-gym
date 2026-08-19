#!/usr/bin/env python3
"""Record editor-connected UnrealZoo features as compact GitHub-ready GIFs.

Start PIE in an UnrealCV-enabled Unreal Editor before recording. The recorder
can connect directly to an editor camera, capture a visible editor/dashboard
window, or capture a screen region. It does not launch or require a packaged
binary.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import mmap
import os
from pathlib import Path
import sys
import time
from typing import Callable

from PIL import Image, ImageDraw, ImageFont, ImageGrab


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def connect_unrealcv(host: str, port: int):
    import unrealcv

    client = unrealcv.Client((host, port))
    client.connect()
    if not client.isconnected():
        raise RuntimeError(f"Failed to connect UnrealCV client to {host}:{port}")
    return client


def unrealcv_capture(client, camera_id: str, mode: str) -> Image.Image:
    from unrealcv.util import read_png

    command = f"vget /camera/{camera_id}/{mode} png"
    response = client.request(command)
    if isinstance(response, str) or not response:
        raise RuntimeError(f"{command} failed: {response!r}")
    array = read_png(response)
    if array is None:
        raise RuntimeError(f"{command} returned an invalid PNG payload")
    return Image.fromarray(array).convert("RGB")


def unrealcv_capture_shared(client, camera_id: str) -> Image.Image:
    """Capture BGRA8 through UnrealCV shared memory without PNG serialization."""
    command = f"vget /camera/{camera_id}/lit_shared"
    response = client.request(command)
    if not response or str(response).casefold().startswith("error"):
        raise RuntimeError(f"{command} failed: {response!r}")
    try:
        meta = json.loads(str(response))
        width = int(meta["width"])
        height = int(meta["height"])
        num_bytes = int(meta["num_bytes"])
        offset = int(meta.get("offset_bytes", 0))
        name = str(meta["name"])
        transport = str(meta["transport"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid shared lit metadata: {response!r}") from exc
    expected_bytes = width * height * 4
    if (
        width <= 0
        or height <= 0
        or num_bytes != expected_bytes
        or meta.get("dtype") != "uint8"
        or meta.get("layout") != "HWC"
        or meta.get("channel_order") != "BGRA"
    ):
        raise RuntimeError(f"Unexpected shared lit layout: {meta!r}")

    if transport == "windows_shared_memory" and os.name == "nt":
        shared = mmap.mmap(-1, offset + num_bytes, tagname=name, access=mmap.ACCESS_READ)
    elif transport == "posix_shared_memory" and sys.platform.startswith("linux"):
        descriptor = os.open(Path("/dev/shm") / name.lstrip("/"), os.O_RDONLY)
        try:
            shared = mmap.mmap(descriptor, offset + num_bytes, access=mmap.ACCESS_READ)
        finally:
            os.close(descriptor)
    else:
        raise RuntimeError(f"Unsupported shared memory transport: {transport!r}")
    try:
        payload = bytes(shared[offset : offset + num_bytes])
    finally:
        shared.close()
    return Image.frombytes("RGBA", (width, height), payload, "raw", "BGRA").convert("RGB")


def parse_vector3(response, operation: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(value) for value in str(response).strip().split())
    except ValueError as exc:
        raise RuntimeError(f"Failed to {operation}: {response!r}") from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Failed to {operation}: {response!r}")
    return values


class UnrealCvImageSource:
    """Own an optional temporary camera and keep it behind a target actor."""

    def __init__(self, client, args):
        self.client = client
        self.args = args
        self.camera_id = ""
        self.spawned_camera_actor = ""
        self.follow_actor = ""
        self.original: dict[str, str] = {}
        self.smoothed_location: tuple[float, float, float] | None = None

    @staticmethod
    def _mapping_id(mapping) -> str:
        for key in ("camera_id", "id", "cid", "CID"):
            if key in mapping:
                return str(mapping[key])
        return ""

    @staticmethod
    def _mapping_actor(mapping) -> str:
        for key in ("actor_name", "name", "actor", "object_name"):
            if key in mapping:
                return str(mapping[key])
        return ""

    def _request_text(self, command: str) -> str:
        response = self.client.request(command)
        text = str(response).strip()
        if not text or text.casefold().startswith("error"):
            raise RuntimeError(f"{command} failed: {response!r}")
        return text

    def _camera_mappings(self) -> list[dict]:
        response = self._request_text("vget /cameras/ids")
        try:
            mappings = json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid camera mapping response: {response!r}") from exc
        if not isinstance(mappings, list):
            raise RuntimeError(f"Expected camera mapping list, got: {response!r}")
        return mappings

    def _spawn_camera(self) -> str:
        before_ids = {self._mapping_id(item) for item in self._camera_mappings()}
        self.spawned_camera_actor = self._request_text("vset /cameras/spawn")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            mappings = self._camera_mappings()
            candidates = [
                item for item in mappings if self._mapping_id(item) not in before_ids
            ]
            if not candidates:
                candidates = [
                    item
                    for item in mappings
                    if self._mapping_actor(item) == self.spawned_camera_actor
                ]
            if candidates:
                return self._mapping_id(candidates[-1])
            time.sleep(0.05)
        raise RuntimeError(
            f"Camera actor {self.spawned_camera_actor!r} did not register a camera ID"
        )

    def _resolve_follow_actor(self, requested: str) -> str:
        objects = set(self._request_text("vget /objects").split())
        if requested not in objects:
            raise RuntimeError(f"Third-person target {requested!r} was not found")
        return requested

    def _set(self, property_name: str, value: str) -> None:
        self._request_text(f"vset /camera/{self.camera_id}/{property_name} {value}")

    def open(self) -> None:
        if self.args.third_person_actor:
            self.follow_actor = self._resolve_follow_actor(self.args.third_person_actor)

        if str(self.args.camera_id).casefold() == "auto":
            self.camera_id = self._spawn_camera()
        else:
            self.camera_id = str(self.args.camera_id)

        if self.args.third_person_actor:
            if not self.spawned_camera_actor:
                self.original = {
                    name: self._request_text(f"vget /camera/{self.camera_id}/{name}")
                    for name in ("location", "rotation", "fov", "size")
                }
            self._set("projection_type", "perspective")
            self._set("size", f"{self.args.camera_width} {self.args.camera_height}")
            self._set("fov", f"{self.args.camera_fov:.6f}")
            self.update_follow_pose()
            print(
                f"THIRD_PERSON|actor={self.follow_actor}|camera={self.camera_id}|"
                f"distance_m={self.args.follow_distance:.2f}|"
                f"side_m={self.args.follow_side:.2f}|height_m={self.args.follow_height:.2f}"
            )
        else:
            print(f"CAMERA|id={self.camera_id}|mode={self.args.camera_mode}")

    def update_follow_pose(self) -> None:
        location = parse_vector3(
            self._request_text(f"vget /object/{self.follow_actor}/location"),
            "read third-person target location",
        )
        rotation = parse_vector3(
            self._request_text(f"vget /object/{self.follow_actor}/rotation"),
            "read third-person target rotation",
        )
        yaw_radians = math.radians(rotation[1])
        local_x = -self.args.follow_distance * 100.0
        local_y = self.args.follow_side * 100.0
        desired = (
            location[0] + math.cos(yaw_radians) * local_x - math.sin(yaw_radians) * local_y,
            location[1] + math.sin(yaw_radians) * local_x + math.cos(yaw_radians) * local_y,
            location[2] + self.args.follow_height * 100.0,
        )
        if self.smoothed_location is None:
            self.smoothed_location = desired
        else:
            alpha = self.args.follow_smoothing
            self.smoothed_location = tuple(
                previous + (target - previous) * alpha
                for previous, target in zip(self.smoothed_location, desired)
            )

        target_z = location[2] + self.args.follow_look_height * 100.0
        delta = (
            location[0] - self.smoothed_location[0],
            location[1] - self.smoothed_location[1],
            target_z - self.smoothed_location[2],
        )
        horizontal = math.hypot(delta[0], delta[1])
        camera_pitch = math.degrees(math.atan2(delta[2], max(horizontal, 1e-6)))
        camera_yaw = math.degrees(math.atan2(delta[1], delta[0]))
        self._set(
            "location", " ".join(f"{value:.6f}" for value in self.smoothed_location)
        )
        self._set("rotation", f"{camera_pitch:.6f} {camera_yaw:.6f} 0.000000")

    def capture(self) -> Image.Image:
        if self.follow_actor:
            self.update_follow_pose()
        return unrealcv_capture(self.client, self.camera_id, self.args.camera_mode)

    def capture_shared(self) -> Image.Image:
        if self.follow_actor:
            self.update_follow_pose()
        return unrealcv_capture_shared(self.client, self.camera_id)

    def close(self) -> None:
        if self.spawned_camera_actor:
            try:
                self.client.request(f"vset /object/{self.spawned_camera_actor}/destroy")
            except Exception:
                pass
        else:
            for property_name in ("size", "fov", "rotation", "location"):
                if property_name not in self.original:
                    continue
                try:
                    self._set(property_name, self.original[property_name])
                except Exception:
                    pass


def find_window_client_rect(title_fragment: str) -> tuple[int, int, int, int]:
    if sys.platform != "win32":
        raise RuntimeError("--source window is currently supported on Windows only")

    user32 = ctypes.windll.user32
    matches: list[tuple[int, str]] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        if title_fragment.casefold() in title.casefold():
            matches.append((int(hwnd), title))
        return True

    callback = callback_type(visit)
    user32.EnumWindows(callback, 0)
    if not matches:
        raise RuntimeError(f"No visible window title contains {title_fragment!r}")
    if len(matches) > 1:
        titles = ", ".join(repr(title) for _, title in matches)
        raise RuntimeError(f"Window title is ambiguous; matches: {titles}")

    hwnd, title = matches[0]
    rect = wintypes.RECT()
    origin = wintypes.POINT(0, 0)
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError(f"Failed to read client area for {title!r}")
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise RuntimeError(f"Failed to locate client area for {title!r}")
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Window is minimized or has an empty client area: {title!r}")
    return origin.x, origin.y, origin.x + width, origin.y + height


def screen_capture(bbox: tuple[int, int, int, int] | None) -> Image.Image:
    return ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB")


def crop_region(image: Image.Image, region: list[int] | None) -> Image.Image:
    if region is None:
        return image
    left, top, width, height = region
    return image.crop((left, top, left + width, top + height))


def resize_width(image: Image.Image, width: int) -> Image.Image:
    if width <= 0 or image.width == width:
        return image
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def add_title(image: Image.Image, title: str) -> Image.Image:
    if not title:
        return image
    frame = image.convert("RGBA")
    draw = ImageDraw.Draw(frame, "RGBA")
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        # Pillow versions before 10.1 do not accept a size argument here.
        font = ImageFont.load_default()
    padding = 12
    text_box = draw.textbbox((0, 0), title, font=font)
    bar_height = text_box[3] - text_box[1] + padding * 2
    draw.rectangle((0, 0, frame.width, bar_height), fill=(0, 0, 0, 170))
    draw.text((padding, padding), title, font=font, fill=(255, 255, 255, 255))
    return frame.convert("RGB")


def save_gif(frames: list[Image.Image], output: Path, fps: float, colors: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    palette_frames = [
        frame.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        for frame in frames
    ]
    palette_frames[0].save(
        output,
        save_all=True,
        append_images=palette_frames[1:],
        duration=round(1000.0 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("unrealcv", "window", "screen"), default="window")
    parser.add_argument("--output", type=Path, default=Path("artifacts/feature.gif"))
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--width", type=int, default=960, help="Output width; 0 keeps source size")
    parser.add_argument("--colors", type=int, default=128, help="GIF palette size")
    parser.add_argument("--title", default="", help="Optional title overlay")
    parser.add_argument("--frames-dir", type=Path, help="Optionally retain captured PNG frames")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--camera-id", default="0", help="Camera ID, or auto to spawn one")
    parser.add_argument("--camera-mode", default="lit", help="UnrealCV image mode, for example lit")
    parser.add_argument(
        "--third-person-actor",
        default="",
        help="Existing actor name to follow",
    )
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--camera-height", type=int, default=540)
    parser.add_argument("--camera-fov", type=float, default=75.0)
    parser.add_argument("--follow-distance", type=float, default=3.0, help="Metres behind")
    parser.add_argument("--follow-side", type=float, default=0.7, help="Metres to actor right")
    parser.add_argument("--follow-height", type=float, default=1.4, help="Metres above")
    parser.add_argument("--follow-look-height", type=float, default=0.15, help="Metres above root")
    parser.add_argument(
        "--follow-smoothing",
        type=float,
        default=0.35,
        help="Per-frame camera position interpolation in (0, 1]",
    )
    parser.add_argument("--window-title", default="", help="Unique window-title fragment")
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Screen bbox, or a crop relative to UnrealCV/window capture",
    )
    args = parser.parse_args()
    if args.duration <= 0 or args.fps <= 0:
        parser.error("--duration and --fps must be positive")
    if args.countdown < 0 or args.width < 0:
        parser.error("--countdown and --width must be non-negative")
    if not 2 <= args.colors <= 256:
        parser.error("--colors must be between 2 and 256")
    if args.region is not None and (args.region[2] <= 0 or args.region[3] <= 0):
        parser.error("--region WIDTH and HEIGHT must be positive")
    if args.source == "window" and not args.window_title:
        parser.error("--source window requires --window-title")
    if args.third_person_actor and args.source != "unrealcv":
        parser.error("--third-person-actor requires --source unrealcv")
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("--camera-width and --camera-height must be positive")
    if args.camera_fov <= 0 or args.follow_distance <= 0 or args.follow_height < 0:
        parser.error("camera FOV/distance must be positive and follow height non-negative")
    if not 0 < args.follow_smoothing <= 1:
        parser.error("--follow-smoothing must be in (0, 1]")
    if args.output.suffix.casefold() != ".gif":
        parser.error("--output must use the .gif extension")
    return args


def main() -> int:
    args = parse_args()
    client = None
    camera_source = None
    capture: Callable[[], Image.Image]

    if args.source == "unrealcv":
        client = connect_unrealcv(args.host, args.port)
        camera_source = UnrealCvImageSource(client, args)
        try:
            camera_source.open()
        except Exception:
            camera_source.close()
            client.disconnect()
            raise
        capture = lambda: crop_region(camera_source.capture(), args.region)
    elif args.source == "window":
        window_bbox = find_window_client_rect(args.window_title)
        capture = lambda: crop_region(screen_capture(window_bbox), args.region)
    else:
        bbox = None
        if args.region is not None:
            left, top, width, height = args.region
            bbox = (left, top, left + width, top + height)
        capture = lambda: screen_capture(bbox)

    try:
        for remaining in range(args.countdown, 0, -1):
            print(f"COUNTDOWN|{remaining}")
            time.sleep(1.0)

        frame_count = max(1, round(args.duration * args.fps))
        interval = 1.0 / args.fps
        frames: list[Image.Image] = []
        start = time.perf_counter()
        for index in range(frame_count):
            deadline = start + index * interval
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            frame = add_title(resize_width(capture(), args.width), args.title)
            frames.append(frame)
            if args.frames_dir is not None:
                args.frames_dir.mkdir(parents=True, exist_ok=True)
                frame.save(args.frames_dir / f"frame_{index:04d}.png")
            print(f"CAPTURE|{index + 1}/{frame_count}", end="\r", flush=True)
        print()
        save_gif(frames, args.output, args.fps, args.colors)
        size_mb = args.output.stat().st_size / (1024 * 1024)
        print(
            f"DONE|output={args.output.resolve()}|frames={len(frames)}|"
            f"fps={args.fps:.2f}|size_mb={size_mb:.2f}"
        )
    finally:
        if camera_source is not None:
            camera_source.close()
        if client is not None:
            client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
