"""Record a MetaHuman crowd wardrobe demo from prebuilt outfit Blueprints."""

from __future__ import annotations

import argparse
import io
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from imageio_ffmpeg import get_ffmpeg_exe
from unrealcv import Client


@dataclass(frozen=True)
class WardrobePreset:
    title: str
    female_path: str
    male_path: str


PRESETS = (
    WardrobePreset(
        "Everyday layers",
        "/Game/MetaHumans/human_Ada_dress0/BP_human_Ada_dress0.BP_human_Ada_dress0",
        "/Game/MetaHumans/human_1_dress1_m-med/BP_human_1_dress1_m-med.BP_human_1_dress1_m-med",
    ),
    WardrobePreset(
        "Street wardrobe",
        "/Game/MetaHumans/human_Ada_dress1_opt/BP_human_Ada_dress1_opt.BP_human_Ada_dress1_opt",
        "/Game/MetaHumans/human_2_dress1_f-fat/BP_human_2_dress1_f-fat.BP_human_2_dress1_f-fat",
    ),
    WardrobePreset(
        "Warm-weather wardrobe",
        "/Game/MetaHumans/human_2_dress0_f-fat/BP_human_2_dress0_f-fat.BP_human_2_dress0_f-fat",
        "/Game/MetaHumans/human_1_dress1_m-thin/BP_human_1_dress1_m-thin.BP_human_1_dress1_m-thin",
    ),
    WardrobePreset(
        "Casual wardrobe",
        "/Game/MetaHumans/human_2_dress2_f-thin/BP_human_2_dress2_f-thin.BP_human_2_dress2_f-thin",
        "/Game/MetaHumans/human_1_m-thin/BP_human_1_m-thin.BP_human_1_m-thin",
    ),
    WardrobePreset(
        "Alternate styling",
        "/Game/MetaHumans/human_Ada_dress0/BP_human_Ada_dress0.BP_human_Ada_dress0",
        "/Game/MetaHumans/human_Ada_dress1_opt/BP_human_Ada_dress1_opt.BP_human_Ada_dress1_opt",
    ),
)


def request_ok(client: Client, command: str) -> bytes | str:
    response = client.request(command)
    if isinstance(response, str) and response.lower().startswith("error"):
        raise RuntimeError(f"{command}: {response}")
    return response


def request_best_effort(client: Client, command: str) -> None:
    response = client.request(command)
    if isinstance(response, str) and response.lower().startswith("error"):
        return


def spawn_pair(client: Client, preset: WardrobePreset, index: int, x: float, y: float, z: float) -> None:
    request_ok(
        client,
        f"vset /objects/spawn_from_path {preset.female_path} WardrobeFemale{index} {x - 70:.1f} {y:.1f} {z:.1f}",
    )
    request_ok(
        client,
        f"vset /objects/spawn_from_path {preset.male_path} WardrobeMale{index} {x + 70:.1f} {y:.1f} {z:.1f}",
    )


def capture(client: Client) -> np.ndarray:
    payload = request_ok(client, "vget /camera/0/lit png")
    if not isinstance(payload, bytes):
        raise RuntimeError(f"Expected PNG bytes, got {payload!r}")
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def label(frame: np.ndarray, preset: WardrobePreset, index: int) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 58), fill=(8, 12, 18, 220))
    draw.text((16, 9), "MetaHuman crowd wardrobe", fill=(248, 250, 255, 255))
    draw.text(
        (16, 32),
        f"{preset.title}  |  preset {index + 1}/{len(PRESETS)}",
        fill=(171, 205, 231, 255),
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def encode(frame_dir: Path, output: Path, fps: int, gif_width: int) -> tuple[Path, Path]:
    ffmpeg = get_ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%05d.png"),
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    gif = output.with_suffix(".gif")
    palette = frame_dir / "palette.png"
    scale = f"fps={fps},scale={gif_width}:-1:flags=lanczos"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(output), "-vf", f"{scale},palettegen", str(palette)],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(output),
            "-i",
            str(palette),
            "-lavfi",
            f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
            str(gif),
        ],
        check=True,
    )
    return output, gif


def record(args: argparse.Namespace) -> tuple[Path, Path]:
    output = Path(args.output).resolve()
    client = Client((args.host, args.port))
    client.connect()
    try:
        for command in (
            f"vset /camera/0/location {args.camera_location}",
            f"vset /camera/0/rotation {args.camera_rotation}",
            f"vset /camera/0/fov {args.fov}",
            "vset /object/BP_Female_Preview_C_1/destroy",
            "vset /object/BP_Male_Preview_C_0/destroy",
        ):
            if command.endswith("/destroy"):
                request_best_effort(client, command)
            else:
                request_ok(client, command)

        frames_per_preset = max(1, round(args.seconds_per_preset * args.fps))
        with tempfile.TemporaryDirectory(prefix="unrealzoo_wardrobe_") as temp_dir:
            frame_dir = Path(temp_dir)
            frame_index = 0
            for index, preset in enumerate(PRESETS):
                if index:
                    request_ok(client, f"vset /object/WardrobeFemale{index - 1}/destroy")
                    request_ok(client, f"vset /object/WardrobeMale{index - 1}/destroy")
                spawn_pair(client, preset, index, args.spawn_x, args.spawn_y, args.spawn_z)
                time.sleep(args.refresh_delay)
                # Let skeletal meshes, materials, lighting, and temporal history settle.
                for _ in range(args.warmup_frames):
                    capture(client)
                    time.sleep(args.warmup_interval)
                frame = label(capture(client), preset, index)
                for _ in range(frames_per_preset):
                    cv2.imwrite(str(frame_dir / f"frame_{frame_index:05d}.png"), frame)
                    frame_index += 1
            return encode(frame_dir, output, args.fps, args.gif_width)
    finally:
        client.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--camera-location", default="0 -600 100")
    parser.add_argument("--camera-rotation", default="0 90 0")
    parser.add_argument("--spawn-x", type=float, default=-70)
    parser.add_argument("--spawn-y", type=float, default=0)
    parser.add_argument("--spawn-z", type=float, default=0)
    parser.add_argument("--fov", type=float, default=45)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seconds-per-preset", type=float, default=2.0)
    parser.add_argument("--refresh-delay", type=float, default=0.5)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--warmup-interval", type=float, default=0.08)
    parser.add_argument("--gif-width", type=int, default=640)
    parser.add_argument(
        "--output",
        default="doc/figs/new_features/metahuman/metahuman_crowd_wardrobe.mp4",
    )
    args = parser.parse_args()
    for path in record(args):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
