#!/usr/bin/env python3
"""Record a supplied Go1 policy using real UE camera frames and 50 Hz physics.

Default recording is 18 simulated seconds: stand, forward, turn, backward, stop.
An existing rendered UE process is required unless --ue-binary is supplied.
Camera capture never advances physics; every 20 ms step and capture is checked.
Original PNGs and a JSON trace are retained beside the annotated MP4.
"""
import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from play_go1 import (
    CONTROL_PERIOD, PROFILE, ROOT, environment_class, finite_vector, launch_owned, load_policy,
    prepare_policy_observation, reset_with_policy_calibration, stop_owned,
    summarize, validate_state, wait_owned,
)


DEFAULT_SEQUENCE = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("forward", 6.0, (0.5, 0.0, 0.0)),
    ("turn", 4.0, (0.2, 0.0, 0.8)),
    ("backward", 4.0, (-0.3, 0.0, 0.0)),
    ("stop", 2.0, (0.0, 0.0, 0.0)),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="Torch .pt inference device; ONNX uses CPU")
    parser.add_argument("--threads", type=int, default=1, help="Torch .pt CPU thread count")
    parser.add_argument("--output", type=Path, required=True, help="New .mp4 path; existing output is refused")
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "YAW"),
                        help="Use one fixed command instead of the demo sequence")
    parser.add_argument("--seconds", type=float, default=18.0, help="Duration for --command, or truncate the sequence")
    parser.add_argument("--fps", type=int, choices=(10, 25), default=10)
    parser.add_argument("--size", nargs=2, type=int, default=(640, 480), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19026)
    parser.add_argument("--actor", default="")
    parser.add_argument("--spawn", type=float, nargs=3, metavar=("X_CM", "Y_CM", "Z_CM"))
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--keep-actor", action="store_true")
    parser.add_argument("--compat-v310", action="store_true")
    parser.add_argument("--camera-id", default="0", help="Temporarily reposition this existing camera")
    parser.add_argument("--camera-distance", type=float, default=2.5, help="Initial rear offset, meters")
    parser.add_argument("--camera-side", type=float, default=2.0, help="Initial side offset, meters")
    parser.add_argument("--camera-height", type=float, default=1.4, help="Height above robot, meters")
    parser.add_argument("--camera-fov", type=float, default=60.0)
    parser.add_argument("--camera-follow-seconds", type=float, default=0.4,
                        help="Camera position lag in simulated seconds; world view direction stays fixed")
    parser.add_argument("--ue-binary", type=Path)
    parser.add_argument("--render-offscreen", action="store_true", help="Launch own process with RenderOffScreen")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--fall-upright", type=float, default=math.cos(math.radians(70.0)))
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.checkpoint.is_file() or args.checkpoint.suffix.lower() not in (".pt", ".onnx"):
        parser.error("--checkpoint must name an existing .pt or .onnx file")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must end in .mp4")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if not args.command and args.seconds > 18:
        parser.error("The default sequence lasts 18 seconds; use --command for longer recording")
    if not math.isclose(args.seconds * args.fps, round(args.seconds * args.fps), abs_tol=1e-8):
        parser.error("--seconds must contain a whole number of video frames")
    if any(value < 64 or value % 2 for value in args.size):
        parser.error("--size values must be even and at least 64")
    if not -1 <= args.fall_upright <= 1:
        parser.error("--fall-upright must be in [-1, 1]")
    try:
        finite_vector(args.command or (0, 0, 0), 3, "command")
        finite_vector([args.camera_distance, args.camera_side, args.camera_height], 3, "camera offset")
        if args.spawn:
            finite_vector(args.spawn, 3, "spawn")
    except ValueError as error:
        parser.error(str(error))
    if not math.isfinite(args.camera_follow_seconds) or args.camera_follow_seconds < 0:
        parser.error("--camera-follow-seconds must be non-negative and finite")
    if not 1 <= args.camera_fov < 179:
        parser.error("--camera-fov must be in [1, 179)")
    if args.ue_binary and (not sys.platform.startswith("linux") or args.host not in ("127.0.0.1", "localhost")):
        parser.error("--ue-binary launches only a local Linux process")
    if args.render_offscreen and not args.ue_binary:
        parser.error("--render-offscreen controls only --ue-binary; existing process must already render")
    args.render = True
    args.offscreen = args.render_offscreen
    return args


def sequence_command(step, fixed):
    if fixed is not None:
        return "fixed", list(fixed)
    boundary = 0
    for name, seconds, command in DEFAULT_SEQUENCE:
        boundary += int(round(seconds / CONTROL_PERIOD))
        if step < boundary:
            return name, list(command)
    return "stop", [0.0, 0.0, 0.0]


def find_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError("Recording requires an existing ffmpeg or imageio_ffmpeg installation") from error


def wait_camera_ready(env, args, proc):
    """A listening UnrealCV socket can precede the map's camera registration."""
    command = "vget /camera/{}/location".format(args.camera_id)
    started = time.monotonic()
    deadline = started + args.startup_timeout
    last_error = "camera location has not been requested"
    original_timeout = env.request_timeout
    attempts = 0
    try:
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError("Owned UE exited with code {} while waiting for camera; last response: {}".format(
                    proc.returncode, last_error))
            env.request_timeout = min(original_timeout, max(0.1, deadline - time.monotonic()))
            attempts += 1
            try:
                response = env.request(command)
            except RuntimeError as error:
                # Retry only the known startup condition, not transport errors,
                # unsupported endpoints, or other rejected commands.
                if "invalid sensor id" not in str(error).lower():
                    raise
                response = str(error)
            if "invalid sensor id" in response.lower():
                last_error = response
            else:
                location = finite_vector(response.replace(",", " ").split(), 3, "camera location")
                return {"attempts": attempts, "wall_seconds": time.monotonic() - started,
                        "location_ue_cm": location, "last_startup_error": last_error if attempts > 1 else None}
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        raise TimeoutError("Camera {} did not register within {}s; last response: {}".format(
            args.camera_id, args.startup_timeout, last_error))
    finally:
        env.request_timeout = original_timeout


class FollowCamera:
    """Follow translation with lag while retaining a stable world viewing angle."""

    def __init__(self, env, args):
        self.env, self.args = env, args
        self.original = {}
        self.location = None
        self.offset = None

    def get(self, name):
        return self.env.request("vget /camera/{}/{}".format(self.args.camera_id, name))

    def set(self, name, value):
        self.env.request("vset /camera/{}/{} {}".format(self.args.camera_id, name, value))

    def open(self):
        for name in ("location", "rotation", "fov", "size"):
            self.original[name] = self.get(name)
        self.set("size", "{} {}".format(*self.args.size))
        self.set("fov", str(self.args.camera_fov))
        self.update()

    def update(self):
        position = finite_vector(self.env.request(
            "vget /object/{}/location".format(self.env.actor_name)
        ).replace(",", " ").split(), 3, "actor location")
        rotation = finite_vector(self.env.request(
            "vget /object/{}/rotation".format(self.env.actor_name)
        ).replace(",", " ").split(), 3, "actor rotation")
        if self.offset is None:
            yaw = math.radians(rotation[1])
            x, y = -100 * self.args.camera_distance, 100 * self.args.camera_side
            self.offset = (math.cos(yaw) * x - math.sin(yaw) * y,
                           math.sin(yaw) * x + math.cos(yaw) * y,
                           100 * self.args.camera_height)
        desired = [p + delta for p, delta in zip(position, self.offset)]
        alpha = 1.0 if self.args.camera_follow_seconds == 0 else (
            1 - math.exp(-1 / (self.args.fps * self.args.camera_follow_seconds))
        )
        self.location = desired if self.location is None else [
            old + alpha * (new - old) for old, new in zip(self.location, desired)
        ]
        delta = [target - camera for target, camera in zip(position, self.location)]
        pitch = math.degrees(math.atan2(delta[2], max(math.hypot(*delta[:2]), 1e-6)))
        yaw = math.degrees(math.atan2(delta[1], delta[0]))
        self.set("location", " ".join("{:.6f}".format(value) for value in self.location))
        self.set("rotation", "{:.6f} {:.6f} 0".format(pitch, yaw))
        return {"actor_location_ue_cm": position, "actor_rotation_degrees": rotation,
                "camera_location_ue_cm": list(self.location), "camera_rotation_degrees": [pitch, yaw, 0]}

    def capture(self, destination):
        from PIL import Image
        pose = self.update()
        command = "vget /camera/{}/lit png".format(self.args.camera_id)
        # env.request converts all responses to text, so use the same client's
        # binary request path for PNG transport.
        payload = self.env.client.request(command, timeout=self.args.request_timeout)
        if not isinstance(payload, bytes) or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("UE did not return a PNG: {!r}".format(str(payload)[:200]))
        with Image.open(io.BytesIO(payload)) as frame:
            frame.load()
            if frame.size != tuple(self.args.size):
                raise RuntimeError("UE camera size {} differs from requested {}".format(frame.size, self.args.size))
        destination.write_bytes(payload)
        return pose

    def close(self):
        for name in ("size", "fov", "rotation", "location"):
            if name in self.original:
                self.set(name, self.original[name])


def encode_video(ffmpeg, frames, args, checkpoint_sha, run_name):
    from PIL import Image, ImageDraw, ImageFont
    log = args.output.with_suffix(".ffmpeg.log")
    font = ImageFont.load_default()
    for filename in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if Path(filename).is_file():
            font = ImageFont.truetype(filename, max(14, int(args.size[0] / 45)))
            break
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", "{}x{}".format(*args.size), "-r", str(args.fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    with log.open("wb") as stderr:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)
        try:
            for frame in frames:
                with Image.open(frame["path"]) as source:
                    picture = source.convert("RGB")
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, args.size[0], 67), fill=(15, 19, 26))
                vx, vy, yaw = frame["command"]
                lines = (
                    "Go1 | {} | {} | {}".format(run_name, checkpoint_sha[:10], frame["phase"]),
                    "vx={:+.2f} vy={:+.2f} m/s  yaw={:+.2f} rad/s  t={:.2f}s".format(vx, vy, yaw, frame["elapsed_sim_seconds"]),
                    "UE camera | 50 Hz physics | {} fps video | playback at simulation speed".format(args.fps),
                )
                for index, line in enumerate(lines):
                    draw.text((10, 4 + 20 * index), line, font=font, fill=(238, 242, 246))
                encoder.stdin.write(picture.tobytes())
            encoder.stdin.close()
            code = encoder.wait(timeout=60)
            if code:
                raise RuntimeError("ffmpeg exited {}; see {}".format(code, log))
        finally:
            if encoder.poll() is None:
                encoder.terminate()
                try:
                    encoder.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
    return command


def main(argv=None):
    args = parse_args(argv)
    ffmpeg = find_ffmpeg()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "example" / "mujoco"))
    import numpy as np
    policy, metadata = load_policy(args.checkpoint, args.device, args.threads)
    output_json = args.output.with_suffix(".json")
    frame_dir = args.output.with_name(args.output.stem + ".frames")
    for path in (args.output, output_json, frame_dir):
        if path.exists():
            raise FileExistsError("Refusing to overwrite {}".format(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir()
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha,
              "policy_metadata": metadata, "compat_v310": args.compat_v310,
              "checkpoint_format": args.checkpoint.suffix.lower(), "inference_device": args.device,
              "inference_threads": args.threads if args.checkpoint.suffix.lower() == ".pt" else None,
              "control_period_seconds": CONTROL_PERIOD, "video_fps": args.fps,
              "video_resolution": args.size, "video": str(args.output),
              "raw_frames": str(frame_dir), "complete": False,
              "recording": "Real UE lit PNGs; overlays added only to MP4; playback follows simulation time",
              "camera_mode": "Stable world viewing angle; smooth translation following actual actor position"}
    proc, env, camera = None, None, None
    samples, frames = [], []
    rollout_start = None
    try:
        if args.ue_binary:
            proc, result["ue_launch_command"] = launch_owned(args, args.output.parent)
            result["owned_ue_pid"] = proc.pid
            wait_owned(proc, args.host, args.port, args.startup_timeout)
        env = environment_class(args.compat_v310)(
            host=args.host, port=args.port, actor_name=args.actor,
            spawn_location=args.spawn, spawn_yaw_offset=args.spawn_yaw,
            spawn_camera_id=args.camera_id,
            keep_actor=args.keep_actor, launch=False, request_timeout=args.request_timeout,
        )
        result["camera_ready"] = wait_camera_ready(env, args, proc)
        env.set_command(np.zeros(3, dtype=np.float32))
        observation, result["observation_calibration"] = reset_with_policy_calibration(
            env, metadata, args.startup_timeout, proc
        )
        _, previous_time = validate_state(env.state)
        result["initial_sim_time"] = previous_time
        result["physics_config"] = copy.deepcopy(env.effective_physics_config)
        result["actor"] = env.actor_name
        camera = FollowCamera(env, args)
        camera.open()
        policy.reset()
        steps = int(round(args.seconds / CONTROL_PERIOD))
        capture_every = int(round(1 / CONTROL_PERIOD / args.fps))
        result["requested_steps"] = steps
        rollout_start = time.perf_counter()
        for step in range(steps):
            phase, command = sequence_command(step, args.command)
            command_array = np.asarray(command, dtype=np.float32)
            env.set_command(command_array)
            policy_observation = prepare_policy_observation(
                observation, command_array, metadata, PROFILE["default_joint_pos"]
            )
            action = policy.act(policy_observation, command_array)
            finite_vector(action, 12, "policy action")
            observation, _, _, info = env.step(action)
            state, previous_time = validate_state(info, previous_time)
            sample = {"step": step + 1, "sim_time": previous_time, "phase": phase,
                      "command": command, "linvel": state[:3], "gyro": state[3:6],
                      "upright": -state[8], "control_clip_count": int(info.get("control_clip_count", 0))}
            samples.append(sample)
            fallen = sample["upright"] < args.fall_upright
            if (step + 1) % capture_every == 0 or fallen:
                frame_path = frame_dir / "frame_{:06d}.png".format(len(frames))
                pose = camera.capture(frame_path)
                after_capture = json.loads(env.request(
                    "vget /object/{}/mujoco_go1_policy_obs".format(env.actor_name)
                ))
                _, captured_time = validate_state(after_capture)
                if not math.isclose(captured_time, previous_time, rel_tol=0, abs_tol=1e-6):
                    raise RuntimeError("Physics advanced during image capture: {} -> {}".format(previous_time, captured_time))
                frames.append({"frame": len(frames), "path": str(frame_path),
                               "step": step + 1, "sim_time": captured_time,
                               "elapsed_sim_seconds": (step + 1) * CONTROL_PERIOD,
                               "phase": phase, "command": command, **pose})
            if (step + 1) % 100 == 0:
                print("RECORD|step={}|phase={}|frames={}|sim_time={:.3f}".format(step + 1, phase, len(frames), previous_time), flush=True)
            if fallen:
                result["stop_reason"] = "fall"
                break
        result.setdefault("stop_reason", "steps_completed")
        result["rollout_complete"] = len(samples) == steps
    except KeyboardInterrupt:
        result["stop_reason"] = "keyboard_interrupt"
    except Exception as error:
        result["error"] = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        elapsed = time.perf_counter() - rollout_start if rollout_start is not None else 0
        result["metrics"] = summarize(samples, 25, elapsed, args.fall_upright)
        result["samples"], result["frames"] = samples, frames
        for name, resource in (("camera", camera), ("env", env)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    result[name + "_cleanup_error"] = str(error)
        if proc is not None:
            stop_owned(proc)
            result["owned_ue_exit_code"] = proc.returncode
        output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("TRACE|{}".format(output_json), flush=True)
    try:
        if frames:
            result["ffmpeg_command"] = encode_video(
                ffmpeg, frames, args, sha, metadata.get("run_path", args.checkpoint.parent.name)
            )
            result["video_frames"] = len(frames)
            result["video_seconds"] = len(frames) / args.fps
            result["complete"] = bool(result.get("rollout_complete"))
    except Exception as error:
        result["encoding_error"] = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("VIDEO|{}".format(args.output), flush=True)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
