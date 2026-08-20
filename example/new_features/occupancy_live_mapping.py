"""Record a live RGB and occupancy observation dashboard from UnrealCV Plus."""

from __future__ import annotations

import argparse
import io
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from imageio_ffmpeg import get_ffmpeg_exe

import sys

PLUGIN_CLIENT = Path(__file__).resolve().parents[3] / "UnrealZoo_UE5_7" / "Plugins" / "unrealcv" / "client" / "python"
if PLUGIN_CLIENT.exists():
    sys.path.insert(0, str(PLUGIN_CLIENT))

from unrealcv import Client  # noqa: E402


def _image(payload: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _occupancy_view(payload: bytes, size: tuple[int, int]) -> np.ndarray:
    grid = np.load(io.BytesIO(payload), allow_pickle=False)
    # Project the x/y_up/z volume onto the ground plane. Flip z for a readable map.
    projection = np.any(grid, axis=1).T.astype(np.uint8) * 255
    projection = cv2.resize(projection, size, interpolation=cv2.INTER_NEAREST)
    colored = np.zeros((*projection.shape, 3), dtype=np.uint8)
    occupied = projection > 0
    # Use occupancy density as intensity so sparse geometry remains visible
    # instead of collapsing into a flat yellow panel.
    intensity = np.clip(120 + projection.astype(np.int16) // 3, 0, 255).astype(np.uint8)
    colored[occupied] = np.stack((np.full(np.count_nonzero(occupied), 40, np.uint8), intensity[occupied], np.full(np.count_nonzero(occupied), 80, np.uint8)), axis=1)
    colored[~occupied] = (18, 24, 32)
    return colored


def _label(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 48), fill=(12, 18, 28))
    draw.text((16, 8), title, fill=(245, 248, 255))
    draw.text((16, 28), subtitle, fill=(168, 196, 220))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def record(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    client = Client((args.host, args.port))
    client.connect()
    width, height = args.width, args.height
    panel_size = (width, height)
    with tempfile.TemporaryDirectory(prefix="unrealzoo_occ_") as temp_dir:
        frame_dir = Path(temp_dir)
        for _ in range(args.warmup_frames):
            client.request("vget /camera/0/lit png")
            client.request(
                f"vget /scene/occupancy npy {args.profile} {args.method} "
                f"{args.start_x:.3f} {args.start_y:.3f} {args.origin_z:.3f} {args.origin_yaw:.3f} 0",
                timeout=args.timeout,
            )
        for index in range(args.frames):
            progress = index / max(1, args.frames - 1)
            y = args.start_y + (args.end_y - args.start_y) * progress
            x = args.start_x + math.sin(progress * math.pi) * args.side_swing
            client.request(f"vset /camera/0/location {x:.3f} {y:.3f} {args.z:.3f}")
            client.request(f"vset /camera/0/rotation {args.pitch:.3f} {args.yaw:.3f} 0")
            rgb = _image(client.request(f"vget /camera/0/lit png"))
            occ_payload = client.request(
                f"vget /scene/occupancy npy {args.profile} {args.method} "
                f"{x:.3f} {y:.3f} {args.origin_z:.3f} {args.origin_yaw:.3f} 0",
                timeout=args.timeout,
            )
            occ = _occupancy_view(occ_payload, panel_size)
            rgb = cv2.resize(rgb, panel_size, interpolation=cv2.INTER_AREA)
            dashboard = np.hstack((
                _label(rgb, "UnrealZoo RGB observation", f"frame {index + 1:02d}/{args.frames}"),
                _label(occ, "Live occupancy projection", f"profile={args.profile} method={args.method}"),
            ))
            cv2.imwrite(str(frame_dir / f"frame_{index:05d}.png"), dashboard)
        client.disconnect()
        command = [
            get_ffmpeg_exe(), "-y", "-framerate", str(args.fps),
            "-i", str(frame_dir / "frame_%05d.png"),
            "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "20", "-movflags", "+faststart",
            str(output),
        ]
        subprocess.run(command, check=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--output", default="artifacts/occupancy_live_mapping.mp4")
    parser.add_argument("--profile", default="lingo_train", choices=("lingo_train", "lingo_vis"))
    parser.add_argument("--method", default="mesh", choices=("bounds", "mesh"))
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--start-x", type=float, default=-4500)
    parser.add_argument("--start-y", type=float, default=-900)
    parser.add_argument("--end-y", type=float, default=350)
    parser.add_argument("--side-swing", type=float, default=80)
    parser.add_argument("--z", type=float, default=220)
    parser.add_argument("--pitch", type=float, default=0)
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument("--origin-z", type=float, default=0)
    parser.add_argument("--origin-yaw", type=float, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--warmup-frames", type=int, default=5)
    args = parser.parse_args()
    print(record(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
