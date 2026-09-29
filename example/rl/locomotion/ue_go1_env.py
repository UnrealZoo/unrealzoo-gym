"""UE Go1 workers for static-scene RL in owned binaries or a local PIE editor.

Physics always runs in UE. World pause suppresses free
running Actor ticks; explicit policy_step still advances each MuJoCo instance.
Capability checks select legacy or keyboard-v2 telemetry and reset contracts.
Physics profiles are explicit; full native domain randomization is not implemented.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import hashlib
import math
import os
from pathlib import Path
from queue import Empty
import signal
import socket
import struct
import shutil
import uuid
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np

CONTROL_DT = 0.02
BLUEPRINT = "/Game/robot-dog-unitree-go1/BP_UnitreeGo1.BP_UnitreeGo1"
MODEL_BUILD_CONCURRENCY = 4
RESET_MODES = ("auto", "rebuild")
RESET_RESULTS = ("initial_model_build", "rebuilt", "reused")
STEP_MODES = ("auto", "single", "batch_serial", "batch_parallel", "fast")
EXECUTION_MODES = STEP_MODES[1:]
MAX_BATCH_ROBOTS = 4096
BATCH_TIMING_FIELDS = ("validation", "prepare", "physics", "finalize_and_report")
TRAIN_MAGIC = b"UZG1TRN1"
TRAIN_COLUMNS = 89
TRAIN_V2_MAGIC = b"UZG1TRN2"
TRAIN_V2_COLUMNS = 117
TRAINING_TELEMETRY_MODES = ("disabled", "keyboard_v2")
TRAIN_V2_FIELDS = {
    "foot_heights": slice(89, 93),
    "foot_current_air_time": slice(93, 97),
    "foot_current_contact_time": slice(97, 101),
    "foot_contact_forces_world": slice(101, 113),
    "root_quat_wxyz": slice(113, 117),
}
TRAIN_V2_PHYSICS_FIELDS = (
    "training_physics_profile", "training_model_path", "actuator_stiffness",
    "actuator_damping", "actuator_force_limits", "actuator_ctrl_limited",
    "actuator_control_ranges", "joint_damping", "joint_frictionloss", "joint_armature",
    "joint_hard_limits", "soft_joint_pos_limits", "training_default_joint_positions",
    "training_default_root_height",
)
TRAIN_DTYPE = "<f8"
TRAIN_PREFIX = struct.Struct("<8sIII")
MAX_TRAIN_HEADER_BYTES = 2 * 1024 * 1024


def _reject_json_constant(value: str):
    raise ValueError(f"Non-finite number in UE JSON response: {value}")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite number in UE JSON response: {value}")
    return number


def checked(response, command: str, *, decode_json: bool = False):
    if response is None:
        raise RuntimeError(f"No response to {command}")
    text = response.decode() if isinstance(response, bytes) else str(response)
    text = text.strip()
    if not text or text.lower().startswith(("error", "failed", "unknown command")):
        raise RuntimeError(f"UE rejected {command}: {text[:500]}")
    if decode_json or text.startswith("{"):
        # Batch replies stay decoded through all per-actor validation. Reject
        # every non-finite JSON number here, including nested diagnostic fields
        # and valid JSON tokens such as 1e999 that overflow Python's float.
        data = (json.loads(text, parse_constant=_reject_json_constant, parse_float=_finite_json_float)
                if decode_json else json.loads(text))
        if isinstance(data, dict) and (data.get("error") or data.get("success") is False):
            raise RuntimeError(f"UE error for {command}: {text[:500]}")
        if decode_json:
            return data
    return text


def state_from(response: str | dict, previous_time: float | None = None) -> dict:
    # A dict comes from the batch request's strict JSON decoder; avoid a
    # serialization round trip for every robot. Single-actor RPCs remain text.
    state = response if isinstance(response, dict) else json.loads(response)
    if not isinstance(state, dict):
        raise RuntimeError("UE Go1 state must be a JSON object")
    observation = np.asarray(state["obs"], dtype=np.float64)
    if observation.shape != (48,) or not np.isfinite(observation).all():
        raise RuntimeError("Non-finite or incompatible Go1 observation")
    if state.get("policy_profile") != "velocity" or state.get("synchronous") is not True:
        raise RuntimeError("UE must use the synchronous Go1 velocity profile")
    now = float(state["sim_time"])
    if not math.isfinite(now) or now < 0:
        raise RuntimeError("Invalid UE simulation time")
    if previous_time is not None and not math.isclose(
        now - previous_time, CONTROL_DT, rel_tol=0, abs_tol=1e-6
    ):
        raise RuntimeError(f"Unexpected physics advance {now - previous_time}; expected 0.02")
    return state


def vector(text: str) -> np.ndarray:
    value = np.asarray(text.replace(",", " ").split(), dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise RuntimeError(f"Invalid UE transform: {text}")
    return value


def numbers(value) -> str:
    return " ".join(f"{float(item):.9f}" for item in value)


def reset_arguments(indices, count, spawn_locations, spawn_rotations, reset_mode):
    """Validate the whole request before any worker can mutate its UE actors."""
    if reset_mode not in RESET_MODES:
        raise ValueError("reset_mode must be auto or rebuild")
    indices = list(indices)
    if any(isinstance(index, (bool, np.bool_)) for index in indices):
        raise ValueError("Reset indices must be integers, not booleans")
    indices = np.asarray(indices)
    if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
        raise ValueError("Reset indices must be a one-dimensional integer sequence")
    if (indices < 0).any() or (indices >= count).any():
        raise IndexError("Reset index is out of range")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Reset indices must be unique")
    poses = []
    for name, value in (("spawn_locations", spawn_locations), ("spawn_rotations", spawn_rotations)):
        if value is not None:
            value = np.asarray(value, dtype=np.float64)
            if value.shape != (len(indices), 3) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape [len(indices), 3]")
            value = value.copy()
        poses.append(value)
    return indices.astype(np.int64), *poses


def validate_reset_randomization(training_telemetry, randomize_reset_pose, seed):
    if type(randomize_reset_pose) is not bool:
        raise ValueError("randomize_reset_pose must be Boolean")
    if randomize_reset_pose and training_telemetry != "keyboard_v2":
        raise ValueError("Random reset poses require keyboard_v2 telemetry and physics")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Reset seed must be a nonnegative integer")


def supports_reset_api(state: dict) -> bool:
    version = state.get("reset_api_version")
    if isinstance(version, bool) or not isinstance(version, (int, float)) or version != 2:
        return False
    modes = state.get("reset_modes")
    if not isinstance(modes, list) or len(modes) != 2 or not all(mode in modes for mode in RESET_MODES):
        raise RuntimeError("Invalid reset API modes")
    return True


def supported_batch_modes(state: dict) -> tuple[str, ...]:
    version_key, modes_key = "policy_step_batch_api_version", "policy_step_batch_modes"
    if version_key not in state and modes_key not in state:
        return ()
    version, modes = state.get(version_key), state.get(modes_key)
    if (isinstance(version, bool) or not isinstance(version, (int, float)) or version != 1
            or not isinstance(modes, list) or len(modes) != 2
            or "serial" not in modes or "parallel" not in modes):
        raise RuntimeError("Invalid or unsupported UE policy step batch capability")
    return ("serial", "parallel")


def supported_train_modes(state: dict, version: int = 1) -> tuple[str, ...]:
    if version not in (1, 2):
        raise ValueError("Training transport version must be 1 or 2")
    prefix = "policy_step_train" if version == 1 else "policy_step_train_v2"
    keys = tuple(f"{prefix}_{name}" for name in ("api_version", "modes", "columns", "dtype"))
    if not any(key in state for key in keys):
        return ()
    advertised, modes, columns, dtype = (state.get(key) for key in keys)
    if (isinstance(advertised, bool) or not isinstance(advertised, (int, float)) or advertised != version
            or not isinstance(modes, list) or len(modes) != 2
            or "serial" not in modes or "parallel" not in modes
            or isinstance(columns, bool) or not isinstance(columns, (int, float))
            or columns != (TRAIN_COLUMNS if version == 1 else TRAIN_V2_COLUMNS) or dtype != TRAIN_DTYPE):
        raise RuntimeError("Invalid or unsupported UE training step capability/schema")
    return ("serial", "parallel")


def validate_training_telemetry(state: dict, previous: dict | None = None, *, reset=False) -> None:
    """Require the source keyboard task's telemetry; never synthesize missing inputs."""
    if (state.get("training_telemetry_mode") != "keyboard_v2"
            or type(state.get("training_telemetry_version")) is not int
            or state["training_telemetry_version"] != 2):
        raise RuntimeError("UE keyboard_v2 training telemetry is missing or incompatible")
    arrays = {}
    for key, section in TRAIN_V2_FIELDS.items():
        value = np.asarray(state.get(key), dtype=np.float64)
        if value.shape != (section.stop - section.start,) or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid {key} in UE keyboard_v2 telemetry")
        arrays[key] = value
    limits = np.asarray(state.get("soft_joint_pos_limits"), dtype=np.float64)
    if (limits.shape != (24,) or not np.isfinite(limits).all()
            or not (limits[::2] < limits[1::2]).all()):
        raise RuntimeError("Invalid soft_joint_pos_limits in UE keyboard_v2 telemetry")
    quaternion = arrays["root_quat_wxyz"]
    if not math.isclose(float(np.linalg.norm(quaternion)), 1.0, rel_tol=0, abs_tol=1e-4):
        raise RuntimeError("Invalid root_quat_wxyz unit quaternion in UE keyboard_v2 telemetry")
    air, contact = arrays["foot_current_air_time"], arrays["foot_current_contact_time"]
    now = float(state["sim_time"])
    if ((air < 0).any() or (contact < 0).any() or ((air > 0) & (contact > 0)).any()
            or (air > now + 1e-6).any() or (contact > now + 1e-6).any()):
        raise RuntimeError("Invalid foot air/contact clocks in UE keyboard_v2 telemetry")
    if reset:
        if (air != 0).any() or (contact != 0).any():
            raise RuntimeError("UE keyboard_v2 reset did not clear foot clocks")
    else:
        contacts = np.asarray(state["foot_contacts"])
        if (not np.array_equal(contact > 0, contacts.astype(bool))
                or not np.array_equal(air > 0, ~contacts.astype(bool))):
            raise RuntimeError("UE keyboard_v2 foot clocks disagree with terrain contact state")
    if previous is not None:
        for key in ("foot_current_air_time", "foot_current_contact_time"):
            old = np.asarray(previous[key], dtype=np.float64)
            if (arrays[key] > old + CONTROL_DT + 1e-6).any():
                raise RuntimeError("UE keyboard_v2 foot clock advanced faster than physics")
        if not np.array_equal(limits, np.asarray(previous["soft_joint_pos_limits"])):
            raise RuntimeError("UE keyboard_v2 joint limits changed during sampling")


def validate_training_physics(state: dict) -> dict:
    """Verify actual opt-in model parameters against the pinned keyboard asset."""
    if state.get("training_physics_profile") != "keyboard_v2":
        raise RuntimeError("UE did not retain the keyboard_v2 training physics profile")
    hard = np.asarray([[-.863, .863], [-.686, 4.501], [-2.818, -.888]] * 4)
    centers, half_ranges = hard.mean(axis=1), np.diff(hard, axis=1)[:, 0] * .45
    expected = {
        "actuator_stiffness": [15.8952426532, 15.8952426532, 35.7642959698] * 4,
        "actuator_damping": [1.0119225760, 1.0119225760, 2.2768257960] * 4,
        "actuator_force_limits": [-23.7, 23.7, -23.7, 23.7, -35.55, 35.55] * 4,
        "joint_damping": [0.] * 12, "joint_frictionloss": [0.] * 12,
        "joint_armature": [.004026312, .004026312, .009059202] * 4,
        "joint_hard_limits": hard.ravel(),
        "soft_joint_pos_limits": np.column_stack((centers - half_ranges, centers + half_ranges)).ravel(),
        "training_default_joint_positions": [.1, .9, -1.8, -.1, .9, -1.8] * 2,
    }
    for key, reference in expected.items():
        value = np.asarray(state.get(key), dtype=np.float64)
        reference = np.asarray(reference)
        if (value.shape != reference.shape or not np.isfinite(value).all()
                or not np.allclose(value, reference, rtol=1e-6, atol=1e-8)):
            raise RuntimeError(f"UE keyboard_v2 source physics mismatch: {key}")
    limited = state.get("actuator_ctrl_limited")
    if (not isinstance(limited, list) or len(limited) != 12
            or any(type(value) is not bool or value for value in limited)):
        raise RuntimeError("UE keyboard_v2 requires unlimited position actuator controls")
    ranges = np.asarray(state.get("actuator_control_ranges"), dtype=np.float64)
    if ranges.shape != (24,) or not np.isfinite(ranges).all():
        raise RuntimeError("Invalid UE keyboard_v2 actuator_control_ranges")
    height = state.get("training_default_root_height")
    if (isinstance(height, bool) or not isinstance(height, (float, int))
            or not math.isclose(height, .278, rel_tol=0, abs_tol=1e-8)):
        raise RuntimeError("UE keyboard_v2 source root height mismatch")
    if not isinstance(state.get("training_model_path"), str) or not state["training_model_path"]:
        raise RuntimeError("UE keyboard_v2 missing training_model_path")
    return {key: state[key] for key in TRAIN_V2_PHYSICS_FIELDS}


def validated_batch_timings(timings) -> dict:
    if (not isinstance(timings, dict) or set(timings) != set(BATCH_TIMING_FIELDS)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in timings.values())):
        raise RuntimeError("Invalid UE batch wall-clock timings_ms")
    return timings


def training_batch_from(response, names, previous_times, actions, commands, *, version=1):
    """Decode and validate one binary physics response before any local commit.

    UnrealCV may decode an all-UTF8 binary payload into a str; encoding that
    exact string recovers its original bytes. Numeric rows remain read-only
    NumPy views, including when the JSON header leaves an unaligned offset.
    """
    if version not in (1, 2):
        raise ValueError("Training transport version must be 1 or 2")
    expected_magic = TRAIN_MAGIC if version == 1 else TRAIN_V2_MAGIC
    expected_columns = TRAIN_COLUMNS if version == 1 else TRAIN_V2_COLUMNS
    if isinstance(response, str):
        response = response.encode("utf-8")
    if not isinstance(response, (bytes, bytearray)):
        raise RuntimeError("UE training step did not return a binary response")
    response = bytes(response)
    if not response.startswith(expected_magic):
        raise RuntimeError(f"UE training step returned an error or invalid binary magic: {response[:200]!r}")
    if len(response) < TRAIN_PREFIX.size:
        raise RuntimeError("Truncated UE training batch prefix")
    _, header_bytes, rows, columns = TRAIN_PREFIX.unpack_from(response)
    if (not 0 < header_bytes <= MAX_TRAIN_HEADER_BYTES or rows != len(names)
            or not 1 <= rows <= MAX_BATCH_ROBOTS or columns != expected_columns):
        raise RuntimeError("Invalid UE training batch dimensions/header length")
    offset = TRAIN_PREFIX.size + header_bytes
    if len(response) != offset + rows * columns * np.dtype(TRAIN_DTYPE).itemsize:
        raise RuntimeError("Incomplete or oversized UE training batch payload")
    header = checked(response[TRAIN_PREFIX.size:offset], "training batch header", decode_json=True)
    advertised = header.get("api_version") if isinstance(header, dict) else None
    if (isinstance(advertised, bool) or not isinstance(advertised, (int, float)) or advertised != version
            or header.get("mode") != "parallel" or header.get("pose_sync") is not False
            or header.get("runtime_diagnostics_enabled") is not False
            or header.get("actors") != names):
        raise RuntimeError("UE training batch schema, identity/order or execution flags differ")
    timings = validated_batch_timings(header.get("timings_ms"))
    values = np.frombuffer(response, dtype=TRAIN_DTYPE, count=rows * columns, offset=offset).reshape(rows, columns)
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite numeric state in UE training batch")
    if ((values[:, 0] < 0).any()
            or not np.allclose(values[:, 0] - previous_times, CONTROL_DT, rtol=0, atol=1e-6)):
        raise RuntimeError("Unexpected physics advance in UE training batch; expected 0.02")
    if (not np.allclose(values[:, 34:46], actions, rtol=1e-5, atol=1e-6)
            or not np.allclose(values[:, 46:49], commands, rtol=1e-5, atol=1e-6)):
        raise RuntimeError("UE training batch does not match the submitted action/command")
    if not np.isin(values[:, 85:89], (0, 1)).all():
        raise RuntimeError("Invalid foot_contacts in UE training batch")
    return values, timings


def step_mode_summary(robots, requested: str, training_telemetry: str = "disabled") -> dict:
    initialized = [robot for robot in robots if robot.state and robot.asset_contract]
    capable = sum(bool(getattr(robot, "batch_step_modes", ())) for robot in initialized)
    mode_field = "train_v2_step_modes" if training_telemetry == "keyboard_v2" else "train_step_modes"
    train_capable = sum("parallel" in getattr(robot, mode_field, ()) for robot in initialized)
    selected = None
    reason = None
    if initialized:
        if training_telemetry == "keyboard_v2" and train_capable != len(initialized):
            reason = "Requested keyboard_v2 telemetry is unsupported; no legacy fallback"
        elif requested == "single":
            selected = "single"
        elif requested == "fast":
            if train_capable != len(initialized):
                reason = "Requested fast mode is unsupported by one or more replicas"
            elif any(robot.runtime_diagnostics_enabled is not False for robot in initialized):
                reason = "Fast mode requires confirmed disabled runtime diagnostics on every replica"
            else:
                selected = "fast"
        elif capable == len(initialized):
            selected = "batch_parallel" if requested == "auto" else requested
        elif requested == "auto":
            selected = "single"
            reason = "One or more replicas do not advertise batch API v1; auto selects legacy single-actor RPCs"
        else:
            reason = "Requested batch mode is unsupported by one or more replicas"
    return {"requested": requested, "selected": selected,
            "initialized_replicas": len(initialized), "batch_api_capable_replicas": capable,
            "train_api_capable_replicas": train_capable,
            "training_telemetry": training_telemetry,
            "state_format": ("train_binary_v2" if training_telemetry == "keyboard_v2" else "train_binary_v1") if selected == "fast" else "full_json" if selected else None,
            "actor_pose_sync_during_step": False if selected == "fast" else True if selected else None,
            "unavailable_reason": reason}


def batch_timing_summary(workers) -> dict:
    result = {
        "source": "UE server wall-clock milliseconds; not CPU time or core utilization",
        "scope": "physics includes dispatch/join; timings exclude outer JSON serialization, transport and queue wait",
        "by_mode": {},
    }
    for mode in ("batch_serial", "batch_parallel", "fast"):
        count = sum(getattr(worker, "batch_timing_counts", {}).get(mode, 0) for worker in workers)
        sums = {field: sum(getattr(worker, "batch_timing_sums_ms", {}).get(mode, {}).get(field, 0.0)
                           for worker in workers) for field in BATCH_TIMING_FIELDS}
        result["by_mode"][mode] = {
            "measured_responses": count, "sum_ms": sums,
            "mean_ms": {field: value / count if count else None for field, value in sums.items()},
        }
    return result


def diagnostics_state(state: dict) -> bool | None:
    """Absent means old server/unknown; only an explicit Boolean is evidence."""
    if "runtime_diagnostics_enabled" not in state:
        return None
    value = state["runtime_diagnostics_enabled"]
    if type(value) is not bool:
        raise RuntimeError("runtime_diagnostics_enabled must be a Boolean")
    return value


def diagnostics_summary(robots, requested: bool) -> dict:
    initialized = [robot for robot in robots if robot.state and robot.asset_contract]
    enabled = sum(getattr(robot, "runtime_diagnostics_enabled", None) is True for robot in initialized)
    disabled = sum(getattr(robot, "runtime_diagnostics_enabled", None) is False for robot in initialized)
    unknown = len(initialized) - enabled - disabled
    matched = enabled if requested else disabled
    return {
        "requested": "enabled" if requested else "disabled",
        "initialized_replicas": len(initialized),
        "control_supported_replicas": enabled + disabled,
        "confirmed_enabled_replicas": enabled, "confirmed_disabled_replicas": disabled,
        "unavailable_replicas": unknown,
        "request_verified_for_all_initialized": bool(initialized) and matched == len(initialized),
        "unavailable_reason": (
            "Server lacks runtime_diagnostics_enabled; diagnostic writes cannot be controlled or verified"
            if unknown else None
        ),
        "scope": "Per-step runtime diagnostic writes only; initialization exports remain enabled",
    }


def same_rotation(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare orientations, including UE's equivalent Euler representations."""
    def quaternion(rotation):
        # FRotator::Quaternion: UE pitch/yaw/roll degrees to XYZW quaternion.
        pitch, yaw, roll = np.deg2rad(np.remainder(rotation, 360.0)) / 2
        sp, sy, sr = np.sin([pitch, yaw, roll])
        cp, cy, cr = np.cos([pitch, yaw, roll])
        return np.array([cr * sp * sy - sr * cp * cy,
                         -cr * sp * cy - sr * cp * sy,
                         cr * cp * sy - sr * sp * cy,
                         cr * cp * cy + sr * sp * sy])

    a, b = quaternion(left), quaternion(right)
    return min(np.linalg.norm(a - b), np.linalg.norm(a + b)) <= math.radians(1e-3) / 2


def parse_connect(value: str) -> tuple[str, int]:
    """Only same-machine attachment can validate UE's exported runtime files."""
    if not isinstance(value, str) or value != value.strip():
        raise ValueError("--connect must be localhost:PORT or 127.0.0.1:PORT")
    if value.startswith("["):
        host, separator, port_text = value[1:].partition("]:")
    else:
        host, separator, port_text = value.rpartition(":")
    if not separator or host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("--connect requires a local host (localhost, 127.0.0.1 or [::1]) and port")
    if not port_text.isascii() or not port_text.isdecimal() or not 1 <= int(port_text) <= 65535:
        raise ValueError("--connect port must be an integer between 1 and 65535")
    # UnrealCV's current Python transport creates AF_INET sockets.
    if host == "::1":
        raise ValueError("UnrealCV Python transport requires localhost or 127.0.0.1 (IPv4)")
    return host, int(port_text)


def host_resources(disk_path: Path | None = None) -> dict:
    memory = {}
    source = None
    reason = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0]) * 1024
        source = "proc_meminfo"
    else:
        try:
            import psutil
            virtual, swap = psutil.virtual_memory(), psutil.swap_memory()
            memory = {"MemAvailable": virtual.available, "MemTotal": virtual.total, "SwapFree": swap.free}
            source = "psutil"
        except (ImportError, OSError) as error:
            reason = f"Host memory metrics unavailable: {error}"
    disk_path = Path(disk_path or Path.cwd()).expanduser().resolve()
    while not disk_path.exists():
        disk_path = disk_path.parent
    disk = shutil.disk_usage(disk_path)
    return {
        "available_ram_bytes": memory.get("MemAvailable"),
        "total_ram_bytes": memory.get("MemTotal"),
        "swap_free_bytes": memory.get("SwapFree"),
        "memory_metrics_source": source,
        "memory_metrics_unavailable_reason": reason,
        "free_disk_bytes": disk.free,
        "disk_checked_path": str(disk_path),
    }


def check_resources(disk_path: Path) -> None:
    resources = host_resources(disk_path)
    available_ram = resources["available_ram_bytes"]
    if available_ram is not None and available_ram < 8 * 1024**3:
        raise RuntimeError("Resource guard: less than 8 GiB available host RAM")
    if resources["free_disk_bytes"] < 12 * 1024**3:
        raise RuntimeError("Resource guard: less than 12 GiB free disk")


def bounded_batch_request(client, commands: list[str], timeout: float) -> list:
    """Use UnrealCV 1.3.2's batched wire format with one response deadline.

    That version's request(list, timeout=...) discards its timeout and blocks on
    recv_data_q.get(). Preserve its IDs and receive thread, but bound both sends
    and response waits. The caller must discard the connection after failure.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Batch timeout must be positive and finite")
    deadline = time.monotonic() + timeout
    sock = client.sock
    previous_timeout = sock.gettimeout()

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError(f"UE batch exceeded {timeout:g}s deadline")
        return value

    try:
        for command in commands:
            sock.settimeout(remaining())
            payload = f"{client.send_message_id}:{command}".encode("utf-8")
            if not client.send(payload):
                raise ConnectionError("Failed to send UE batch request")
            client.send_message_id += 1
        client.recv_num_q.put(-len(commands))
        replies = []
        for _ in commands:
            try:
                reply = client.recv_data_q.get(timeout=remaining())
            except Empty as error:
                raise TimeoutError(f"UE batch exceeded {timeout:g}s deadline") from error
            if isinstance(reply, BaseException):
                raise reply
            if reply is None:
                raise ConnectionError("UE closed during a batch response")
            replies.append(reply)
        return replies
    finally:
        try:
            sock.settimeout(previous_timeout)
        except OSError:
            pass


def write_failure(path: Path, error: BaseException, details: dict) -> None:
    """Keep startup evidence even when construction never returns a pool."""
    details = {**details, "error": f"{type(error).__name__}: {error}"}
    try:
        details["host"] = host_resources(path.parent)
    except Exception as resource_error:
        details["resource_error"] = str(resource_error)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(details, indent=2) + "\n")
    except OSError:
        # A full disk or revoked directory must not prevent process cleanup or
        # replace the original startup failure with a diagnostics failure.
        pass


def failure_resources(path: Path) -> dict:
    try:
        return host_resources(path)
    except Exception as error:
        return {"resource_read_error": str(error)}


@dataclass
class Robot:
    name: str
    location: np.ndarray
    rotation: np.ndarray
    state: dict = field(default_factory=dict)
    command: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    asset_contract: dict = field(default_factory=dict)
    reset_api_supported: bool = False
    runtime_diagnostics_enabled: bool | None = None
    batch_step_modes: tuple[str, ...] = ()
    train_step_modes: tuple[str, ...] = ()
    train_v2_step_modes: tuple[str, ...] = ()
    reset_metadata: dict = field(default_factory=dict)


class UEWorker:
    def __init__(
        self, binary: Path, port: int, directory: Path, timeout: float,
        reset_mode: str = "auto", runtime_diagnostics: bool = False,
        step_mode: str = "auto", training_telemetry: str = "disabled",
        randomize_reset_pose: bool = False, seed: int = 42,
    ):
        import unrealcv

        self._initialize(
            directory, timeout, reset_mode, port, True, ("127.0.0.1", port),
            runtime_diagnostics, step_mode, training_telemetry, randomize_reset_pose, seed,
        )
        directory = self.directory
        with socket.socket() as listener:
            # A recently closed owned UE connection may leave TIME_WAIT here.
            # Reuse permits that state but still rejects an active listener.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))
        self.launch_command = [
            str(binary), binary.parents[2].name, "-NullRHI", "-nosound",
            "-unattended", "-nosplash", "-NoVSync", "-NoSaveConfig",
            f"-UnrealCVPort={port}", f"-UserDir={directory / 'user'}",
            f"-abslog={directory / 'ue.log'}",
            "-ini:Engine:[/Script/Engine.Engine]:bUseFixedFrameRate=False,"
            "[/Script/Engine.Engine]:bSmoothFrameRate=False",
        ]
        started = time.perf_counter()
        try:
            with (directory / "stdout.log").open("wb") as stream:
                self.proc = subprocess.Popen(
                    self.launch_command, cwd=binary.parents[3], stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, "TMPDIR": str(self.tmpdir)},
                )
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"UE {self.proc.pid} exited {self.proc.returncode}")
                listing = subprocess.run(
                    ["ss", "-ltnp"], capture_output=True, text=True, check=True
                ).stdout
                if any(
                    f":{port} " in row and f"pid={self.proc.pid}," in row
                    for row in listing.splitlines()
                ):
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError(f"Owned UE did not listen on {port}")
            self.client = unrealcv.Client(("127.0.0.1", port))
            self.client.connect()
            if not self.client.isconnected():
                raise RuntimeError(f"Cannot connect to owned UE on {port}")
            while time.monotonic() < deadline:
                try:
                    self.camera_location = vector(self.request("vget /camera/0/location"))
                    self.camera_rotation = vector(self.request("vget /camera/0/rotation"))
                    break
                except RuntimeError as error:
                    if "invalid sensor id" not in str(error).lower():
                        raise
                    time.sleep(0.2)
            else:
                raise TimeoutError("UE camera registration did not complete")
            self.request("vset /action/game/pause")
            if self.request("vget /action/game/is_paused").lower() != "true":
                raise RuntimeError("UE pause required for deterministic static training")
            self.start_seconds = time.perf_counter() - started
        except BaseException as error:
            resources_at_failure = failure_resources(directory)
            self.close()
            write_failure(directory / "startup_failure.json", error, {
                "port": port,
                "pid": self.proc.pid if self.proc is not None else None,
                "exit_code": self.proc.returncode if self.proc is not None else None,
                "robots": len(self.robots),
                "tmpdir": str(self.tmpdir),
                "cleanup_errors": self.cleanup_errors,
                "host_at_failure": resources_at_failure,
            })
            raise

    def _initialize(
        self, directory, timeout, reset_mode, port, owns_process, endpoint,
        runtime_diagnostics=False, step_mode="auto", training_telemetry="disabled",
        randomize_reset_pose=False, seed=42,
    ):
        reset_arguments([], 0, None, None, reset_mode)
        if step_mode not in STEP_MODES:
            raise ValueError(f"step_mode must be one of {STEP_MODES}")
        self.step_mode = step_mode
        if training_telemetry not in TRAINING_TELEMETRY_MODES:
            raise ValueError(f"training_telemetry must be one of {TRAINING_TELEMETRY_MODES}")
        self.training_telemetry = training_telemetry
        validate_reset_randomization(training_telemetry, randomize_reset_pose, seed)
        self.randomize_reset_pose = randomize_reset_pose
        self.seed = seed
        self.reset_pose_rng = np.random.default_rng(seed)
        if step_mode == "fast" and runtime_diagnostics:
            raise ValueError("Fast mode requires runtime_diagnostics=False")
        self.step_mode_counts = dict.fromkeys(EXECUTION_MODES, 0)
        self.batch_timing_counts = {}
        self.batch_timing_sums_ms = {}
        if type(runtime_diagnostics) is not bool:
            raise ValueError("runtime_diagnostics must be Boolean")
        self.runtime_diagnostics = runtime_diagnostics
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Request timeout must be positive and finite")
        self.reset_mode = reset_mode
        directory = Path(directory).expanduser().resolve()
        self.owns_process = owns_process
        self.endpoint = endpoint
        self.original_world_paused = None
        self.world_context = None
        self.pause_restore_needed = False
        self.owned_actor_names = []
        self.actor_name_prefix = "uzgym_" + uuid.uuid4().hex
        self.proc = None
        self.client = None
        self.closed = False
        self.failed = False
        self.cleanup_errors: list[str] = []
        self.robots: list[Robot] = []
        self.timeout = timeout
        self.port = port
        self.directory = directory
        self.start_seconds = 0.0
        self.reset_seconds = 0.0
        self.reset_count = 0
        self.reset_result_counts = dict.fromkeys(RESET_RESULTS, 0)
        self.reset_request_counts = dict.fromkeys(RESET_MODES, 0)
        self.request_seconds = 0.0
        directory.mkdir(parents=True, exist_ok=False)
        self.tmpdir = directory / "tmp"
        if owns_process:
            self.tmpdir.mkdir()

    @classmethod
    def attach(
        cls, connect: str, directory: Path, timeout: float, reset_mode: str = "auto",
        runtime_diagnostics: bool = False, step_mode: str = "auto",
        training_telemetry: str = "disabled", randomize_reset_pose: bool = False, seed: int = 42,
    ):
        """Connect to an existing local PIE session without owning its process."""
        import unrealcv

        endpoint = parse_connect(connect)
        worker = cls.__new__(cls)
        worker._initialize(
            directory, timeout, reset_mode, endpoint[1], False, endpoint,
            runtime_diagnostics, step_mode, training_telemetry, randomize_reset_pose, seed,
        )
        started = time.perf_counter()
        try:
            worker.client = unrealcv.Client(endpoint)
            worker.client.connect()
            if not worker.client.isconnected():
                raise RuntimeError(f"Cannot connect to UE editor at {connect}; start PIE and enable UnrealCV")
            worker.world_context = worker._read_game_context(worker.client)
            worker.camera_location = vector(worker.request("vget /camera/0/location"))
            worker.camera_rotation = vector(worker.request("vget /camera/0/rotation"))
            paused = worker.request("vget /action/game/is_paused").lower()
            if paused not in ("true", "false"):
                raise RuntimeError("UE did not report a valid game pause state")
            worker.original_world_paused = paused == "true"
            worker.pause_restore_needed = True
            if not worker.original_world_paused:
                worker.request("vset /action/game/pause")
            if worker.request("vget /action/game/is_paused").lower() != "true":
                raise RuntimeError("UE pause required for deterministic static training")
            worker.start_seconds = time.perf_counter() - started
            return worker
        except BaseException as error:
            worker.close()
            write_failure(worker.directory / "startup_failure.json", error, {
                "connection_mode": "external_editor", "endpoint": connect,
                "owns_process": False, "cleanup_errors": worker.cleanup_errors,
            })
            raise

    def _read_game_context(self, client):
        command = "vget /action/game/context"
        try:
            context = json.loads(checked(client.request(command, timeout=self.timeout), command))
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(
                "Editor attach requires the game/context command and active PIE; "
                "build the updated UnrealCV plugin and start Play before connecting"
            ) from error
        if (not isinstance(context, dict)
                or context.get("is_game_world") is not True or context.get("has_player_controller") is not True
                or not isinstance(context.get("world_name"), str)
                or isinstance(context.get("world_id"), bool)
                or not isinstance(context.get("world_id"), (int, float))
                or not math.isfinite(context["world_id"])):
            raise RuntimeError("No active playable PIE/game world; start Play in UE before connecting")
        return context

    def request(self, command: str, *, decode_json: bool = False, raw: bool = False):
        if decode_json and raw:
            raise ValueError("Select either decoded JSON or a raw response")
        if self.closed or self.failed:
            raise RuntimeError("UE worker connection is closed or failed")
        started = time.perf_counter()
        try:
            response = self.client.request(command, timeout=self.timeout)
            return response if raw else checked(response, command, decode_json=decode_json)
        finally:
            self.request_seconds += time.perf_counter() - started

    def request_many(self, commands: list[str]) -> list[str]:
        if not commands:
            return []
        if self.closed or self.failed:
            raise RuntimeError("UE worker connection is closed or failed")
        started = time.perf_counter()
        try:
            replies = bounded_batch_request(self.client, commands, self.timeout)
            if not isinstance(replies, list) or len(replies) != len(commands):
                raise RuntimeError("UE batch reply length mismatch")
            return [checked(reply, command) for reply, command in zip(replies, commands)]
        except BaseException:
            # Partial batches leave IDs and replies unsynchronized. Never retry
            # physics actions or reuse this connection after such a failure.
            self.failed = True
            raise
        finally:
            self.request_seconds += time.perf_counter() - started

    def add_robot(self) -> dict:
        # Co-located independent models give every replica the same static road.
        # Isolation is checked by the benchmark before reporting training capacity.
        yaw = math.radians(float(self.camera_rotation[1]))
        location = self.camera_location + np.array([300 * math.cos(yaw), 300 * math.sin(yaw), 100])
        if getattr(self, "owns_process", True):
            name = self.request(f"vset /objects/spawn_from_path {BLUEPRINT} {numbers(location)}")
        else:
            name = f"{self.actor_name_prefix}_{len(self.owned_actor_names)}"
            self.owned_actor_names.append(name)
            returned_name = self.request(
                f"vset /objects/spawn_from_path {BLUEPRINT} {name} {numbers(location)}"
            )
            if returned_name != name:
                raise RuntimeError("UE did not preserve the requested owned actor name")
        if self.robots:
            # Reuse the saved initial pose, not a peer's current rendered pose.
            # Repeating a generic settle trace after peers have stepped can hit
            # their displayed bodies and place new replicas above the ground.
            reference = self.robots[0]
            self.request(f"vset /object/{name}/rotation {numbers(reference.rotation)}")
            self.request(f"vset /object/{name}/location {numbers(reference.location)}")
        else:
            self.request(f"vset /object/{name}/rotation 0 {self.camera_rotation[1]} 0")
            self.request(f"vset /object/{name}/settle_to_ground simple 100 5000 35")
        robot = Robot(
            name, vector(self.request(f"vget /object/{name}/location")),
            vector(self.request(f"vget /object/{name}/rotation")),
        )
        if self.robots and (
            not np.allclose(robot.location, reference.location, atol=1e-3, rtol=0)
            or not np.allclose(robot.rotation, reference.rotation, atol=1e-3, rtol=0)
        ):
            raise RuntimeError("New replica did not retain the shared initial pose")
        self.robots.append(robot)
        return self.reset_one(len(self.robots) - 1, initial=True)

    def reset_one(
        self, index: int, initial: bool = False, *, spawn_location=None,
        spawn_rotation=None, reset_mode: str | None = None,
    ) -> dict:
        mode = self.reset_mode if reset_mode is None else reset_mode
        indices, locations, rotations = reset_arguments(
            [index], len(self.robots),
            None if spawn_location is None else [spawn_location],
            None if spawn_rotation is None else [spawn_rotation], mode,
        )
        robot = self.robots[int(indices[0])]
        location = robot.location.copy() if locations is None else locations[0]
        rotation = robot.rotation.copy() if rotations is None else rotations[0]
        use_api = not initial and robot.reset_api_supported
        keyboard_v2 = getattr(self, "training_telemetry", "disabled") == "keyboard_v2"
        randomize_pose = getattr(self, "randomize_reset_pose", False)
        # reset_pose performs its own cached auto reset, but preserves the
        # preceding result enum. Skip the redundant reset only after a verified
        # reuse at the same saved spawn; initialization/rebuild results and
        # changed spawns still need the ordinary reset RPC first.
        pose_only = (
            use_api and keyboard_v2 and randomize_pose and mode == "auto"
            and robot.reset_metadata.get("last_reset_result") == "reused"
            and np.array_equal(location, robot.location)
            and np.array_equal(rotation, robot.rotation)
        )
        started = time.perf_counter()
        try:
            offset = self.reset_pose_rng.uniform(
                [-.5, -.5, .01, -math.pi], [.5, .5, .05, math.pi],
            ) if randomize_pose else None
            if initial and keyboard_v2:
                # The physical profile must be selected before preview/start
                # builds the MuJoCo model. Unsupported servers fail explicitly.
                self.request(f"vset /object/{robot.name}/mujoco_go1_training_profile keyboard_v2")
            if pose_only:
                state = state_from(self.request(
                    f"vset /object/{robot.name}/mujoco_go1_policy_reset_pose {numbers(offset)}"
                ))
                if diagnostics_state(state) is not robot.runtime_diagnostics_enabled:
                    raise RuntimeError("UE randomized reset changed runtime diagnostics")
            elif use_api:
                state = state_from(self.request(
                    f"vset /object/{robot.name}/mujoco_go1_policy_reset {mode} "
                    f"{numbers(location)} {numbers(rotation)}"
                ))
            else:
                if not initial:
                    self.request(f"vset /object/{robot.name}/mujoco_quadruped_pose_preview/stop")
                    self.request(f"vset /object/{robot.name}/location {numbers(location)}")
                    self.request(f"vset /object/{robot.name}/rotation {numbers(rotation)}")
                self.request(f"vset /object/{robot.name}/mujoco_quadruped_pose_preview/start go1")
                self.request(f"vset /object/{robot.name}/mujoco_go1_policy_command 0 0 0")
                state = state_from(self.request(
                    f"vset /object/{robot.name}/mujoco_go1_policy_sync/start"
                ))
            if keyboard_v2:
                if "parallel" not in supported_train_modes(state, 2):
                    raise RuntimeError("keyboard_v2 requires UE training API v2; no legacy fallback")
                if initial:
                    state = state_from(self.request(
                        f"vset /object/{robot.name}/mujoco_go1_training_telemetry keyboard_v2"
                    ))
                if state.get("training_physics_profile") != "keyboard_v2":
                    raise RuntimeError("UE did not retain the keyboard_v2 training physics profile")
            actual_diagnostics = diagnostics_state(state)
            if actual_diagnostics is None:
                if robot.runtime_diagnostics_enabled is not None:
                    raise RuntimeError("UE reset lost its runtime diagnostics capability")
            else:
                requested_diagnostics = getattr(self, "runtime_diagnostics", False)
                # Configure every new capable actor explicitly, even if its
                # current setting already matches. Older servers get no RPC.
                if initial or robot.runtime_diagnostics_enabled is None or actual_diagnostics != requested_diagnostics:
                    setting = "enabled" if requested_diagnostics else "disabled"
                    state = state_from(self.request(
                        f"vset /object/{robot.name}/mujoco_go1_diagnostics {setting}"
                    ))
                    actual_diagnostics = diagnostics_state(state)
                if actual_diagnostics is not requested_diagnostics:
                    raise RuntimeError("UE did not apply the requested runtime diagnostics setting")
            if randomize_pose:
                if not pose_only:
                    state = state_from(self.request(
                        f"vset /object/{robot.name}/mujoco_go1_policy_reset_pose {numbers(offset)}"
                    ))
                applied = np.asarray(state.get("training_reset_pose_offset"), dtype=np.float64)
                if (applied.shape != (4,) or not np.isfinite(applied).all()
                        or not np.allclose(applied, offset, rtol=0, atol=1e-6)):
                    raise RuntimeError("UE randomized root pose does not match the requested cached-spawn offset")
                if diagnostics_state(state) is not actual_diagnostics:
                    raise RuntimeError("UE randomized reset changed runtime diagnostics")
            if abs(float(state["sim_time"])) > 1e-8:
                raise RuntimeError(f"Reset advanced physics: {state['sim_time']}")
            targets = np.asarray(state["control_targets"], dtype=np.float64)
            if targets.shape != (12,) or not np.isfinite(targets).all():
                raise RuntimeError("Invalid reset joint reference")
            if not np.allclose(state["obs"][33:48], 0.0, rtol=0, atol=1e-8):
                raise RuntimeError("Reset did not clear the previous action and command")
            if robot.asset_contract and not np.allclose(
                targets, robot.asset_contract["reset_joint_reference"], rtol=0, atol=1e-6
            ):
                raise RuntimeError("Reset changed the initial joint reference")
            supports_api = supports_reset_api(state)
            batch_modes = supported_batch_modes(state)
            train_modes = supported_train_modes(state)
            train_v2_modes = supported_train_modes(state, 2)
            if keyboard_v2:
                validate_training_telemetry(state, reset=True)
                physics_metadata = validate_training_physics(state)
                if not np.allclose(targets, state["training_default_joint_positions"], rtol=0, atol=1e-6):
                    raise RuntimeError("UE keyboard_v2 reset joints differ from the source default pose")
                if "parallel" not in train_v2_modes:
                    raise RuntimeError("UE reset lost keyboard_v2 training capability")
            if not initial and robot.batch_step_modes and batch_modes != robot.batch_step_modes:
                raise RuntimeError("UE reset lost its batch step capability")
            if not initial and robot.train_step_modes and train_modes != robot.train_step_modes:
                raise RuntimeError("UE reset lost its training step capability")
            requested_step_mode = getattr(self, "step_mode", "auto")
            if requested_step_mode.startswith("batch_") and requested_step_mode[6:] not in batch_modes:
                raise RuntimeError(f"Requested step mode {requested_step_mode} requires UE batch API v1")
            if requested_step_mode == "fast" and ("parallel" not in (train_v2_modes if keyboard_v2 else train_modes) or actual_diagnostics is not False):
                raise RuntimeError(f"Requested fast mode requires UE training API v{2 if keyboard_v2 else 1} and confirmed disabled diagnostics")
            if use_api and not supports_api:
                raise RuntimeError("Reset response no longer advertises its API capability")
            result = "initial_model_build" if initial else "rebuilt"
            if supports_api:
                actual_location = np.asarray(state.get("reset_spawn_location"), dtype=np.float64)
                actual_rotation = np.asarray(state.get("reset_spawn_rotation"), dtype=np.float64)
                if (actual_location.shape != (3,) or actual_rotation.shape != (3,)
                        or not np.isfinite(actual_location).all() or not np.isfinite(actual_rotation).all()):
                    raise RuntimeError("Invalid reset spawn in UE response")
                if (not np.allclose(actual_location, location, rtol=0, atol=1e-3)
                        or not same_rotation(actual_rotation, rotation)):
                    raise RuntimeError("UE reset spawn does not match the requested pose")
                result = state.get("last_reset_result")
                allowed_results = ("initial_model_build",) if initial else ("reused", "rebuilt")
                if result not in allowed_results or (mode == "rebuild" and not initial and result != "rebuilt"):
                    raise RuntimeError("Unexpected UE reset result")
                location, rotation = actual_location.copy(), actual_rotation.copy()
            asset_contract = robot.asset_contract
            if result != "reused" or not asset_contract:
                # A changed spawn or forced terrain rebuild can overwrite the
                # same XML pathname. Hash and validate the new file each time.
                report = Path(state["environment_collision_report_path"])
                model_path = report.with_name(report.name.replace("_collision.csv", ".xml"))
                if getattr(self, "owns_process", True):
                    if not model_path.resolve().is_relative_to(self.directory.resolve()):
                        raise RuntimeError("Runtime model must belong to this owned UE worker")
                else:
                    expected = f"go1_runtime_000_{robot.name}.xml"
                    if (not model_path.is_absolute() or model_path.name != expected
                            or model_path.resolve().name != expected
                            or tuple(model_path.resolve().parent.parts[-3:]) != ("Saved", "UnrealCV_MuJoCo", "MJCF")):
                        raise RuntimeError("Attached runtime model must be this actor's local UE Saved/MJCF export")
                model_bytes = model_path.read_bytes()
                root = ET.fromstring(model_bytes)
                imu = root.find(".//site[@name='imu']")
                if imu is None:
                    raise RuntimeError("UE runtime MJCF has no IMU site")
                offset = vector(imu.attrib["pos"])
                if not np.allclose(offset, [-0.01592, -0.06659, -0.00617], rtol=0, atol=1e-9):
                    raise RuntimeError(f"UE IMU offset does not match the task reward: {offset}")
                asset_contract = {
                    "runtime_mjcf": str(model_path),
                    "runtime_mjcf_sha256": hashlib.sha256(model_bytes).hexdigest(),
                    "imu_offset_m": offset.tolist(),
                    "reset_joint_reference": targets.tolist(),
                    "environment_geom_count": state["environment_geom_count"],
                }
            if keyboard_v2:
                asset_contract = {**asset_contract,
                    "training_physics": physics_metadata,
                }
                if Path(state["training_model_path"]).resolve() != Path(asset_contract["runtime_mjcf"]).resolve():
                    raise RuntimeError("UE keyboard_v2 physics metadata does not identify the validated runtime model")
            reset_metadata = {key: value for key, value in state.items()
                              if key.startswith(("reset_", "last_reset_", "policy_step_", "environment_", "training_"))
                              or key in TRAIN_V2_PHYSICS_FIELDS}
        except BaseException:
            # A reset may already have changed physics or queued a late reply.
            # Never retry or conceal it through the legacy rebuild path.
            self.failed = True
            raise
        robot.state = state
        robot.asset_contract = asset_contract
        robot.reset_api_supported = supports_api
        robot.runtime_diagnostics_enabled = actual_diagnostics
        robot.batch_step_modes = batch_modes
        robot.train_step_modes = train_modes
        robot.train_v2_step_modes = train_v2_modes
        robot.reset_metadata = reset_metadata
        robot.location, robot.rotation = location.copy(), rotation.copy()
        robot.command = np.zeros(3)
        self.reset_seconds += time.perf_counter() - started
        self.reset_count += 1
        self.reset_result_counts[result] += 1
        if not initial:
            self.reset_request_counts[mode] += 1
        return robot.state

    def reset_indices(
        self, indices, *, spawn_locations=None, spawn_rotations=None, reset_mode=None,
    ) -> None:
        mode = self.reset_mode if reset_mode is None else reset_mode
        indices, locations, rotations = reset_arguments(
            indices, len(self.robots), spawn_locations, spawn_rotations, mode,
        )
        for offset, index in enumerate(indices):
            check_resources(self.directory)
            self.reset_one(
                int(index), reset_mode=mode,
                spawn_location=None if locations is None else locations[offset],
                spawn_rotation=None if rotations is None else rotations[offset],
            )

    def _validate_step_state(self, robot: Robot, reply: str | dict, action, command) -> dict:
        state = state_from(reply, float(robot.state["sim_time"]))
        if diagnostics_state(state) is not robot.runtime_diagnostics_enabled:
            raise RuntimeError("UE runtime diagnostics setting/capability changed during sampling")
        if supported_batch_modes(state) != robot.batch_step_modes:
            raise RuntimeError("UE batch step capability changed during sampling")
        if supported_train_modes(state) != robot.train_step_modes:
            raise RuntimeError("UE training step capability changed during sampling")
        if getattr(self, "training_telemetry", "disabled") == "keyboard_v2":
            if (supported_train_modes(state, 2) != robot.train_v2_step_modes
                    or state.get("training_physics_profile") != "keyboard_v2"):
                raise RuntimeError("UE keyboard_v2 capability/physics profile changed during sampling")
            validate_training_telemetry(state, robot.state)
            if any(state.get(key) != robot.reset_metadata.get(key) for key in TRAIN_V2_PHYSICS_FIELDS):
                raise RuntimeError("UE keyboard_v2 physics metadata changed during sampling")
        observation = np.asarray(state["obs"], dtype=np.float64)
        if (not np.allclose(observation[33:45], action, rtol=1e-5, atol=1e-6)
                or not np.allclose(observation[45:48], command, rtol=1e-5, atol=1e-6)):
            raise RuntimeError("UE step response does not match the submitted action/command")
        for name, size in (("control_targets", 12), ("foot_positions", 12),
                           ("foot_velocities", 12), ("foot_contacts", 4)):
            values = np.asarray(state.get(name), dtype=np.float64)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise RuntimeError(f"Invalid {name} in UE step response")
            if name == "foot_contacts" and not np.isin(values, (0, 1)).all():
                raise RuntimeError("Invalid foot_contacts in UE step response")
        return state

    def step(self, actions: np.ndarray, commands: np.ndarray) -> list[dict]:
        if self.closed or self.failed:
            raise RuntimeError("UE worker connection is closed or failed; never retry a physics step")
        # Validate the complete action+command batch before any RPC. Pool.step
        # also validates across workers, before dispatching to their UE worlds.
        actions = np.asarray(actions, dtype=np.float32)
        commands = np.asarray(commands, dtype=np.float32)
        if (not self.robots or actions.shape != (len(self.robots), 12)
                or commands.shape != (len(self.robots), 3)
                or not np.isfinite(actions).all() or not np.isfinite(commands).all()):
            raise ValueError("Worker requires finite actions[N,12] and commands[N,3]")
        names = [robot.name for robot in self.robots]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Worker robot names must be unique nonempty strings")
        telemetry = getattr(self, "training_telemetry", "disabled")
        selection = step_mode_summary(self.robots, getattr(self, "step_mode", "auto"), telemetry)
        mode = selection["selected"]
        if mode is None:
            raise RuntimeError(selection["unavailable_reason"] or "Worker robots are not initialized")
        if mode != "single" and len(self.robots) > MAX_BATCH_ROBOTS:
            raise ValueError(f"UE batch API v1 supports at most {MAX_BATCH_ROBOTS} robots per request")
        timings = None
        try:
            if mode == "single":
                changed = [i for i, robot in enumerate(self.robots) if not np.array_equal(robot.command, commands[i])]
                self.request_many([
                    f"vset /object/{self.robots[i].name}/mujoco_go1_policy_command {numbers(commands[i])}"
                    for i in changed
                ])
                replies = self.request_many([
                    f"vset /object/{robot.name}/mujoco_go1_policy_step {numbers(action)}"
                    for robot, action in zip(self.robots, actions)
                ])
                if len(replies) != len(self.robots):
                    raise RuntimeError("Incomplete single-actor UE step responses")
            elif mode == "fast":
                version = 2 if telemetry == "keyboard_v2" else 1
                payload = {"pose_sync": False, "robots": [
                    {"actor": robot.name, "actions": action.tolist(), "command": command.tolist()}
                    for robot, action, command in zip(self.robots, actions, commands)
                ]}
                response = self.request(
                    f"vset /mujoco/go1/policy_step_train{'_v2' if version == 2 else ''} parallel "
                    + json.dumps(payload, separators=(",", ":"), allow_nan=False), raw=True,
                )
                rows, timings = training_batch_from(
                    response, names, np.asarray([robot.state["sim_time"] for robot in self.robots]),
                    actions, commands, version=version,
                )
                # Keep the existing task interface with compact dictionaries of
                # array views. No per-robot JSON or stale full-state diagnostics
                # are produced. Full evaluation/display stays on the old modes.
                states = [{
                    **(robot.reset_metadata if version == 2 else {}),
                    "state_format": f"train_binary_v{version}", "actor_pose_synchronized": False,
                    "policy_profile": "velocity", "synchronous": True,
                    "runtime_diagnostics_enabled": False,
                    **({"policy_step_batch_api_version": 1,
                        "policy_step_batch_modes": list(robot.batch_step_modes)} if robot.batch_step_modes else {}),
                    "policy_step_train_api_version": 1,
                    "policy_step_train_modes": list(robot.train_step_modes),
                    "policy_step_train_columns": TRAIN_COLUMNS,
                    "policy_step_train_dtype": TRAIN_DTYPE,
                    "sim_time": float(row[0]), "obs": row[1:49],
                    "control_targets": row[49:61], "foot_positions": row[61:73],
                    "foot_velocities": row[73:85], "foot_contacts": row[85:89],
                    **({key: row[section] for key, section in TRAIN_V2_FIELDS.items()} if version == 2 else {}),
                } for robot, row in zip(self.robots, rows)]
                if version == 2:
                    for robot, state in zip(self.robots, states):
                        validate_training_telemetry(state, robot.state)
            else:
                batch_mode = mode[6:]
                payload = {"robots": [
                    {"actor": robot.name, "actions": action.tolist(), "command": command.tolist()}
                    for robot, action, command in zip(self.robots, actions, commands)
                ]}
                response = self.request(
                    f"vset /mujoco/go1/policy_step_batch {batch_mode} "
                    + json.dumps(payload, separators=(",", ":"), allow_nan=False),
                    decode_json=True,
                )
                version = response.get("policy_step_batch_api_version") if isinstance(response, dict) else None
                if (isinstance(version, bool) or not isinstance(version, (int, float)) or version != 1
                        or response.get("mode") != batch_mode):
                    raise RuntimeError("UE batch response has the wrong API version or execution mode")
                entries = response.get("robots")
                if (not isinstance(entries, list) or len(entries) != len(self.robots)
                        or any(not isinstance(entry, dict) for entry in entries)
                        or [entry.get("actor") for entry in entries] != names):
                    raise RuntimeError("UE batch response is incomplete or actor order/identity differs")
                if "timings_ms" in response:
                    timings = validated_batch_timings(response["timings_ms"])
                replies = [entry.get("state") for entry in entries]
            if mode != "fast":
                states = [self._validate_step_state(robot, reply, action, command)
                          for robot, reply, action, command in zip(self.robots, replies, actions, commands)]
        except BaseException:
            # Physics may have stepped; never replay the request or fall back
            # to per-actor RPCs. Commit local state only after every row validates.
            self.failed = True
            raise
        for robot, state, command in zip(self.robots, states, commands):
            robot.state, robot.command = state, command.copy()
        if not hasattr(self, "step_mode_counts"):
            self.step_mode_counts = dict.fromkeys(EXECUTION_MODES, 0)
        self.step_mode_counts[mode] += 1
        if timings is not None:
            if not hasattr(self, "batch_timing_counts"):
                self.batch_timing_counts, self.batch_timing_sums_ms = {}, {}
            self.batch_timing_counts[mode] = self.batch_timing_counts.get(mode, 0) + 1
            sums = self.batch_timing_sums_ms.setdefault(mode, dict.fromkeys(BATCH_TIMING_FIELDS, 0.0))
            for field in BATCH_TIMING_FIELDS:
                sums[field] += timings[field]
        return [robot.state for robot in self.robots]

    def close(self) -> None:
        # All actors belong to our process; exiting avoids hundreds of stop RPCs.
        if self.closed:
            return
        self.closed = True
        if not getattr(self, "owns_process", True):
            self._close_attached()
            return
        client, self.client = self.client, None
        if client is not None:
            try:
                client.recv_data_q.put(ConnectionError("Owned UE worker is closing"))
            except Exception as error:
                self.cleanup_errors.append(f"Wake pending request: {error}")
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                self.cleanup_errors.append(f"Terminate owned process group: {error}")
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    self.cleanup_errors.append(f"Kill owned process group: {error}")
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired as error:
                    self.cleanup_errors.append(str(error))
        if client is not None:
            try:
                client.disconnect()
            except Exception as error:
                self.cleanup_errors.append(f"Disconnect UnrealCV: {error}")

    def _close_attached(self):
        client, self.client = self.client, None
        if client is not None:
            try:
                client.recv_data_q.put(ConnectionError("Attached UE worker is closing"))
                client.disconnect()
            except Exception as error:
                self.cleanup_errors.append(f"Disconnect external UE: {error}")
        if not self.owned_actor_names and not self.pause_restore_needed:
            return
        cleanup_client = None
        try:
            import unrealcv
            cleanup_client = unrealcv.Client(self.endpoint)
            cleanup_client.connect()
            if not cleanup_client.isconnected():
                raise RuntimeError("Cannot reconnect for actor cleanup and pause restoration")
            context = self._read_game_context(cleanup_client)
            if self.world_context is None or context["world_id"] != self.world_context["world_id"]:
                raise RuntimeError("PIE world changed; skipped actor cleanup and pause restoration")
            def cleanup_request(command):
                return checked(cleanup_client.request(command, timeout=self.timeout), command)
            for name in self.owned_actor_names:
                try:
                    # Destroy calls component EndPlay/StopSimulation as needed.
                    cleanup_request(f"vset /object/{name}/destroy")
                except Exception as error:
                    self.cleanup_errors.append(f"Destroy owned actor {name}: {error}")
            if self.pause_restore_needed:
                action = "pause" if self.original_world_paused else "resume"
                cleanup_request(f"vset /action/game/{action}")
                actual = cleanup_request("vget /action/game/is_paused").lower()
                if actual != str(self.original_world_paused).lower():
                    raise RuntimeError("Failed to restore the original UE pause state")
                self.pause_restore_needed = False
        except Exception as error:
            self.cleanup_errors.append(f"External UE cleanup: {error}")
        finally:
            if cleanup_client is not None:
                try:
                    cleanup_client.disconnect()
                except Exception as error:
                    self.cleanup_errors.append(f"Disconnect cleanup client: {error}")


class UEGo1Pool:
    def __init__(
        self, ue_binary: Path | None, num_processes: int, agents_per_process: int,
        base_port: int, output_dir: Path, request_timeout: float = 120.0,
        reset_mode: str = "auto", connect: str | None = None,
        runtime_diagnostics: bool = False,
        step_mode: str = "auto",
        training_telemetry: str = "disabled",
        randomize_reset_pose: bool = False,
        seed: int = 42,
    ):
        reset_arguments([], 0, None, None, reset_mode)
        if step_mode not in STEP_MODES:
            raise ValueError(f"step_mode must be one of {STEP_MODES}")
        self.step_mode = step_mode
        if training_telemetry not in TRAINING_TELEMETRY_MODES:
            raise ValueError(f"training_telemetry must be one of {TRAINING_TELEMETRY_MODES}")
        self.training_telemetry = training_telemetry
        validate_reset_randomization(training_telemetry, randomize_reset_pose, seed)
        self.randomize_reset_pose = randomize_reset_pose
        self.seed = seed
        if type(runtime_diagnostics) is not bool:
            raise ValueError("runtime_diagnostics must be Boolean")
        if step_mode == "fast" and runtime_diagnostics:
            raise ValueError("Fast mode requires runtime_diagnostics=False")
        self.runtime_diagnostics = runtime_diagnostics
        self.reset_mode = reset_mode
        if (ue_binary is None) == (connect is None):
            raise ValueError("Specify exactly one of ue_binary or connect")
        self.connect = connect
        if connect is not None:
            parse_connect(connect)
            if num_processes != 1:
                raise ValueError("Editor attachment supports exactly one connection/process")
        self.binary = Path(ue_binary).expanduser().resolve(strict=True) if ue_binary is not None else None
        if num_processes < 1 or agents_per_process < 1:
            raise ValueError("Process and replica counts must be positive")
        self.workers: list[UEWorker] = []
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.requested_processes = num_processes
        self.agents_per_process = agents_per_process
        self.executor = ThreadPoolExecutor(max_workers=num_processes)
        self.closed = False
        self.cleanup_errors: list[str] = []
        self.initialized = False
        self.step_seconds = 0.0
        self.vector_steps = 0
        self.reset_wall_seconds = 0.0
        self.created_at = time.perf_counter()
        try:
            for index in range(num_processes):
                self.check_resources()
                if connect is not None:
                    self.workers.append(UEWorker.attach(
                        connect, self.output_dir / "ue-attached", request_timeout, reset_mode=reset_mode,
                        runtime_diagnostics=runtime_diagnostics,
                        step_mode=step_mode,
                        training_telemetry=training_telemetry,
                        randomize_reset_pose=randomize_reset_pose, seed=int(seed) + index,
                    ))
                else:
                    self.workers.append(UEWorker(
                        self.binary, base_port + index,
                        self.output_dir / f"ue-{base_port + index}", request_timeout, reset_mode=reset_mode,
                        runtime_diagnostics=runtime_diagnostics,
                        step_mode=step_mode,
                        training_telemetry=training_telemetry,
                        randomize_reset_pose=randomize_reset_pose, seed=int(seed) + index,
                    ))
        except BaseException as error:
            resources_at_failure = failure_resources(self.output_dir)
            self.close()
            failed_directory = self.output_dir / (
                "ue-attached" if connect is not None else f"ue-{base_port + len(self.workers)}"
            )
            write_failure(self.output_dir / "startup_failure.json", error, {
                "requested_processes": num_processes,
                "requested_agents_per_process": agents_per_process,
                "fully_initialized_workers": len(self.workers),
                "actual_robots": sum(len(worker.robots) for worker in self.workers),
                "attempted_worker_directory": str(failed_directory),
                "attempted_worker_failure_report": str(failed_directory / "startup_failure.json"),
                "workers": [
                    {"pid": worker.proc.pid if worker.proc else None, "port": worker.port,
                     "robots": len(worker.robots), "exit_code": worker.proc.returncode if worker.proc else None}
                    for worker in self.workers
                ],
                "cleanup_errors": self.cleanup_errors,
                "host_at_failure": resources_at_failure,
            })
            raise

    @property
    def num_envs(self) -> int:
        return len(self.workers) * self.agents_per_process

    @property
    def states(self) -> list[dict]:
        return [robot.state for worker in self.workers for robot in worker.robots]

    def check_resources(self) -> None:
        check_resources(self.output_dir)

    def _require_ready(self) -> None:
        if self.closed or any(len(worker.robots) != self.agents_per_process for worker in self.workers):
            raise RuntimeError("UE pool is closed or has incompletely initialized replicas")
        if any(getattr(worker, "failed", False) for worker in self.workers):
            raise RuntimeError("UE pool has a failed worker; physics steps must not be retried")

    def reset(self) -> list[dict]:
        if not self.initialized:
            started = time.perf_counter()
            self.grow(self.agents_per_process)
            self.initialized = True
            self.reset_wall_seconds += time.perf_counter() - started
        else:
            self.reset_indices(list(range(self.num_envs)))
        return self.states

    def grow(self, target_per_process: int) -> list[dict]:
        if self.closed:
            raise RuntimeError("Cannot grow a closed UE pool")
        if not isinstance(target_per_process, int) or target_per_process < 1:
            raise ValueError("Target replica count must be a positive integer")
        if any(len(worker.robots) > target_per_process for worker in self.workers):
            raise ValueError("Pool can only grow")
        while any(len(worker.robots) < target_per_process for worker in self.workers):
            active = [worker for worker in self.workers if len(worker.robots) < target_per_process]
            for offset in range(0, len(active), MODEL_BUILD_CONCURRENCY):
                self.check_resources()
                wave = active[offset:offset + MODEL_BUILD_CONCURRENCY]
                pending = [self.executor.submit(worker.add_robot) for worker in wave]
                for future in pending:
                    future.result()
            print(f"UE_INIT|robots={sum(len(worker.robots) for worker in self.workers)}", flush=True)
        self.agents_per_process = target_per_process
        return self.states

    def step(self, actions, commands) -> list[dict]:
        self._require_ready()
        actions = np.asarray(actions, dtype=np.float32)
        commands = np.asarray(commands, dtype=np.float32)
        if actions.shape != (self.num_envs, 12) or commands.shape != (self.num_envs, 3):
            raise ValueError("Pool requires actions[N,12] and commands[N,3]")
        if not np.isfinite(actions).all() or not np.isfinite(commands).all():
            raise ValueError("Non-finite policy action or command")
        started = time.perf_counter()
        futures = []
        for index, worker in enumerate(self.workers):
            sl = slice(index * self.agents_per_process, (index + 1) * self.agents_per_process)
            futures.append(self.executor.submit(worker.step, actions[sl], commands[sl]))
        for future in futures:
            future.result()
        self.step_seconds += time.perf_counter() - started
        self.vector_steps += 1
        return self.states

    def reset_indices(
        self, indices, *, spawn_locations=None, spawn_rotations=None, reset_mode=None,
    ) -> list[dict]:
        """Reset selected replicas at saved or explicitly supplied UE spawn poses.

        Spawn rows correspond to indices in their supplied order. Locations use
        UE centimeters; rotations use degrees in pitch, yaw, roll order.
        """
        self._require_ready()
        mode = self.reset_mode if reset_mode is None else reset_mode
        indices, locations, rotations = reset_arguments(
            indices, self.num_envs, spawn_locations, spawn_rotations, mode,
        )
        if not indices.size:
            return self.states
        started = time.perf_counter()
        groups = [[] for _ in self.workers]
        rows = [[] for _ in self.workers]
        for offset, index in enumerate(indices):
            worker_id, local_id = divmod(int(index), self.agents_per_process)
            groups[worker_id].append(local_id)
            rows[worker_id].append(offset)
        active = [(worker, group, selected) for worker, group, selected
                  in zip(self.workers, groups, rows) if group]
        for offset in range(0, len(active), MODEL_BUILD_CONCURRENCY):
            self.check_resources()
            wave = active[offset:offset + MODEL_BUILD_CONCURRENCY]
            futures = [self.executor.submit(
                worker.reset_indices, group, reset_mode=mode,
                spawn_locations=None if locations is None else locations[selected],
                spawn_rotations=None if rotations is None else rotations[selected],
            ) for worker, group, selected in wave]
            for future in futures:
                future.result()
        self.reset_wall_seconds += time.perf_counter() - started
        return self.states

    def metrics(self) -> dict:
        processes = []
        for worker in self.workers:
            pid = worker.proc.pid if worker.proc else None
            values = {"VmRSS": None, "VmHWM": None, "Threads": None,
                      "cpu_seconds": None, "wchar": None, "write_bytes": None}
            if pid is not None:
                status = Path(f"/proc/{pid}/status")
                try:
                    if status.exists():
                        for line in status.read_text().splitlines():
                            if line.startswith(("VmRSS:", "VmHWM:", "Threads:")):
                                key, value = line.split(":", 1)
                                values[key] = int(value.split()[0])
                        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
                        values["cpu_seconds"] = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
                    else:
                        values["resource_read_error"] = "Process /proc metrics unavailable"
                except (OSError, ValueError, IndexError) as error:
                    values["resource_read_error"] = str(error)
                try:
                    for line in Path(f"/proc/{pid}/io").read_text().splitlines():
                        key, value = line.split(":", 1)
                        if key in ("wchar", "write_bytes"):
                            values[key] = int(value.strip())
                except (OSError, ValueError) as error:
                    values["io_read_error"] = str(error)
            else:
                values["resource_read_error"] = "External editor process is not owned; process metrics are not measured"
            processes.append({
                "pid": pid, "port": worker.port, "robots": len(worker.robots),
                "owns_process": getattr(worker, "owns_process", True),
                "world_context": getattr(worker, "world_context", None),
                "reset_result_counts": dict(worker.reset_result_counts),
                "reset_request_counts": dict(worker.reset_request_counts),
                "runtime_diagnostics": diagnostics_summary(worker.robots, getattr(self, "runtime_diagnostics", False)),
                "step_execution": {
                    **step_mode_summary(worker.robots, getattr(self, "step_mode", "auto"),
                                        getattr(self, "training_telemetry", "disabled")),
                    "completed_vector_steps_by_mode": dict(getattr(worker, "step_mode_counts", {})),
                },
                "batch_timings": batch_timing_summary([worker]),
                "tmpdir": str(worker.tmpdir) if getattr(worker, "owns_process", True) else None,
                **values,
            })
        reset_result_counts = {
            result: sum(worker.reset_result_counts[result] for worker in self.workers)
            for result in RESET_RESULTS
        }
        reset_request_counts = {
            mode: sum(worker.reset_request_counts[mode] for worker in self.workers)
            for mode in RESET_MODES
        }
        return {
            "backend": "ue_external_editor" if getattr(self, "connect", None) else "ue_v310_nullrhi",
            "connection_mode": "attach" if getattr(self, "connect", None) else "launch",
            "render_mode": "existing_editor_settings" if getattr(self, "connect", None) else "NullRHI",
            "world_paused": True,
            "static_scene_only": True,
            "num_envs": sum(len(worker.robots) for worker in self.workers),
            "initialized_replicas": sum(
                bool(robot.state) and bool(robot.asset_contract)
                for worker in self.workers for robot in worker.robots
            ),
            "configured_num_envs": self.num_envs,
            "uniform_ready": all(len(worker.robots) == self.agents_per_process for worker in self.workers),
            "num_processes": len(self.workers), "agents_per_process": self.agents_per_process,
            "model_build_concurrency_limit": MODEL_BUILD_CONCURRENCY,
            "vector_steps": self.vector_steps, "step_wall_seconds": self.step_seconds,
            "reset_wall_seconds": self.reset_wall_seconds,
            "robot_resets": sum(worker.reset_count for worker in self.workers),
            "reset_result_counts": reset_result_counts,
            "reset_request_counts": reset_request_counts,
            "reset_api_v2_capable_replicas": sum(
                robot.reset_api_supported
                for worker in self.workers for robot in worker.robots
            ),
            "lifetime_wall_seconds": time.perf_counter() - self.created_at,
            "physics_control_dt": CONTROL_DT, "full_physics_config_verified": False,
            "reset_mode": self.reset_mode,
            "training_telemetry": getattr(self, "training_telemetry", "disabled"),
            "randomize_reset_pose": getattr(self, "randomize_reset_pose", False),
            "reset_pose_seed": getattr(self, "seed", None),
            "runtime_diagnostics": diagnostics_summary(
                [robot for worker in self.workers for robot in worker.robots],
                getattr(self, "runtime_diagnostics", False),
            ),
            "step_execution": {
                "requested": getattr(self, "step_mode", "auto"),
                "selected_modes_by_worker": [process["step_execution"]["selected"] for process in processes],
                "batch_api_capable_replicas": sum(process["step_execution"]["batch_api_capable_replicas"] for process in processes),
                "train_api_capable_replicas": sum(process["step_execution"]["train_api_capable_replicas"] for process in processes),
                "state_formats_by_worker": [process["step_execution"]["state_format"] for process in processes],
                "actor_pose_sync_during_step_by_worker": [process["step_execution"]["actor_pose_sync_during_step"] for process in processes],
                "unavailable_reasons_by_worker": [process["step_execution"]["unavailable_reason"] for process in processes],
                "completed_worker_vector_steps_by_mode": {
                    mode: sum(getattr(worker, "step_mode_counts", {}).get(mode, 0) for worker in self.workers)
                    for mode in EXECUTION_MODES
                },
            },
            "batch_timings": batch_timing_summary(self.workers),
            "reset_validation": "synchronous velocity, sim_time=0, initial joint reference, cleared action/command",
            "replica_layout": "separate MuJoCo models, saved per-replica spawns; isolation benchmark required",
            "verified_runtime_assets": [
                robot.asset_contract for worker in self.workers for robot in worker.robots
            ],
            "processes": processes, "host": host_resources(self.output_dir),
            "cleanup_errors": self.cleanup_errors,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        # End processes before joining workers, so a blocked request cannot keep
        # the UE instance alive after an exception or interrupted benchmark.
        # Use a separate executor: physics workers can all be blocked in RPC.
        # Shutdown deadlines must overlap instead of costing N * 5 seconds.
        with ThreadPoolExecutor(max_workers=max(1, len(self.workers))) as cleaners:
            closing = [(worker, cleaners.submit(worker.close)) for worker in self.workers]
            for worker, future in closing:
                try:
                    future.result()
                except Exception as error:
                    self.cleanup_errors.append(f"Worker {worker.port}: {error}")
                self.cleanup_errors.extend(getattr(worker, "cleanup_errors", []))
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
        except Exception as error:
            self.cleanup_errors.append(f"Worker executor: {error}")
