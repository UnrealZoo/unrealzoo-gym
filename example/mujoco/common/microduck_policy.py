#!/usr/bin/env python3
"""Pollen Robotics Microduck 61-D ONNX policy wrapper."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ACTION_DIM = 14
OBSERVATION_DIM = 61
COMMAND_DIM = 13
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
DEFAULT_POSE = np.asarray(
    [
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,
        0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530,
    ],
    dtype=np.float32,
)


def _vector(state: Mapping[str, object], key: str, length: int) -> np.ndarray:
    value = np.asarray(state.get(key, []), dtype=np.float32).reshape(-1)
    if value.size != length:
        raise ValueError(f"State field {key!r} has {value.size} values; expected {length}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"State field {key!r} contains non-finite values")
    return value


class _OnnxPolicy:
    def __init__(self, path: str | Path):
        import onnxruntime as ort

        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.session = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"{self.path.name} must have one input and one output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        if list(inputs[0].shape) != [1, OBSERVATION_DIM]:
            raise ValueError(
                f"{self.path.name} input is {inputs[0].shape}; expected [1, {OBSERVATION_DIM}]"
            )
        if list(outputs[0].shape) != [1, ACTION_DIM]:
            raise ValueError(
                f"{self.path.name} output is {outputs[0].shape}; expected [1, {ACTION_DIM}]"
            )
        print(
            f"POLICY|file={self.path.name}|input={self.input_name}[1,{OBSERVATION_DIM}]|"
            f"output={self.output_name}[1,{ACTION_DIM}]"
        )

    def infer(self, observation: np.ndarray) -> np.ndarray:
        action = self.session.run(
            [self.output_name], {self.input_name: observation.reshape(1, -1)}
        )[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != ACTION_DIM or not np.all(np.isfinite(action)):
            raise RuntimeError(f"Invalid ONNX action shape/value: {action}")
        return action


class MicroDuckPolicy:
    """Walking/standing policy switch with the upstream observation contract."""

    def __init__(
        self,
        walking_path: str | Path,
        standing_path: str | Path,
        *,
        walking_action_scale: float = 0.9,
        standing_action_scale: float = 1.0,
        head_lowpass: float = 0.5,
        legs_lowpass: float = 0.7,
        switch_threshold: float = 0.05,
    ):
        self.walking = _OnnxPolicy(walking_path)
        self.standing = _OnnxPolicy(standing_path)
        self.walking_action_scale = float(walking_action_scale)
        self.standing_action_scale = float(standing_action_scale)
        self.head_lowpass = float(head_lowpass)
        self.legs_lowpass = float(legs_lowpass)
        for name, value in (
            ("head_lowpass", self.head_lowpass),
            ("legs_lowpass", self.legs_lowpass),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        self.switch_threshold = float(switch_threshold)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.previous_targets: np.ndarray | None = None
        self.active_name = "standing"

    def reset(self) -> None:
        self.last_action.fill(0.0)
        self.previous_targets = None
        self.active_name = "standing"

    def build_observation(
        self,
        state: Mapping[str, object],
        velocity_command: Sequence[float],
        *,
        head_command: Sequence[float] | None = None,
        body_command: Sequence[float] | None = None,
    ) -> np.ndarray:
        velocity = np.asarray(velocity_command, dtype=np.float32).reshape(-1)
        if velocity.size != 3:
            raise ValueError("velocity_command must contain vx, vy, yaw_rate")
        head = np.zeros(4, dtype=np.float32) if head_command is None else np.asarray(head_command, dtype=np.float32)
        body = np.zeros(6, dtype=np.float32) if body_command is None else np.asarray(body_command, dtype=np.float32)
        if head.size != 4 or body.size != 6:
            raise ValueError("head_command/body_command must contain 4/6 values")

        command = np.concatenate((velocity, head.reshape(-1), body.reshape(-1)))
        observation = np.concatenate(
            (
                _vector(state, "base_angular_velocity", 3),
                _vector(state, "projected_gravity", 3),
                _vector(state, "joint_positions", ACTION_DIM) - DEFAULT_POSE,
                _vector(state, "joint_velocities", ACTION_DIM),
                self.last_action,
                command,
            )
        ).astype(np.float32)
        if observation.size != OBSERVATION_DIM:
            raise AssertionError(f"Built {observation.size}-D observation")
        return observation

    def infer(
        self, state: Mapping[str, object], velocity_command: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, str]:
        command = np.asarray(velocity_command, dtype=np.float32).reshape(3)
        moving = float(np.linalg.norm(command)) >= self.switch_threshold
        selected = self.walking if moving else self.standing
        active_name = "walking" if moving else "standing"
        policy_command = command if moving else np.zeros(3, dtype=np.float32)
        observation = self.build_observation(state, policy_command)
        action = selected.infer(observation)
        action_scale = (
            self.walking_action_scale if moving else self.standing_action_scale
        )
        targets = DEFAULT_POSE + action * action_scale
        if self.previous_targets is not None:
            # Match robotd's trained first-order filters. Policy slots 5..8 are
            # the head; all remaining slots are leg joints (the mouth is not
            # part of the 14-D policy/action bridge).
            targets[5:9] = (
                self.head_lowpass * targets[5:9]
                + (1.0 - self.head_lowpass) * self.previous_targets[5:9]
            )
            leg_indices = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13))
            targets[leg_indices] = (
                self.legs_lowpass * targets[leg_indices]
                + (1.0 - self.legs_lowpass) * self.previous_targets[leg_indices]
            )
        self.previous_targets = targets.copy()
        # The observation feeds back the raw network output, not scaled or
        # filtered targets. This is shared across standing/walking policies.
        self.last_action = action.copy()
        self.active_name = active_name
        return targets.astype(np.float32), observation, active_name


class MicroDuckPolicySet:
    """Hot-swap official 61-D policies while sharing action/filter history."""

    def __init__(
        self,
        policy_paths: Mapping[str, str | Path],
        *,
        head_lowpass: float = 0.5,
        legs_lowpass: float = 0.7,
    ):
        if not policy_paths:
            raise ValueError("policy_paths cannot be empty")
        self.policies = {
            str(name): _OnnxPolicy(path) for name, path in policy_paths.items()
        }
        self.head_lowpass = float(head_lowpass)
        self.legs_lowpass = float(legs_lowpass)
        for name, value in (
            ("head_lowpass", self.head_lowpass),
            ("legs_lowpass", self.legs_lowpass),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.previous_targets: np.ndarray | None = None
        self.active_name = next(iter(self.policies))

    def reset(self) -> None:
        self.last_action.fill(0.0)
        self.previous_targets = None

    def infer(
        self,
        state: Mapping[str, object],
        policy_name: str,
        command: Sequence[float],
        *,
        action_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if policy_name not in self.policies:
            raise KeyError(f"Unknown Microduck policy {policy_name!r}")
        command_vector = np.asarray(command, dtype=np.float32).reshape(-1)
        if command_vector.size != COMMAND_DIM or not np.all(np.isfinite(command_vector)):
            raise ValueError(f"command must contain {COMMAND_DIM} finite values")
        observation = np.concatenate(
            (
                _vector(state, "base_angular_velocity", 3),
                _vector(state, "projected_gravity", 3),
                _vector(state, "joint_positions", ACTION_DIM) - DEFAULT_POSE,
                _vector(state, "joint_velocities", ACTION_DIM),
                self.last_action,
                command_vector,
            )
        ).astype(np.float32)
        action = self.policies[policy_name].infer(observation)
        targets = DEFAULT_POSE + action * float(action_scale)
        if self.previous_targets is not None:
            targets[5:9] = (
                self.head_lowpass * targets[5:9]
                + (1.0 - self.head_lowpass) * self.previous_targets[5:9]
            )
            leg_indices = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13))
            targets[leg_indices] = (
                self.legs_lowpass * targets[leg_indices]
                + (1.0 - self.legs_lowpass) * self.previous_targets[leg_indices]
            )
        self.previous_targets = targets.copy()
        self.last_action = action.copy()
        self.active_name = policy_name
        return targets.astype(np.float32), observation, policy_name


def neutral_state() -> dict[str, object]:
    return {
        "base_angular_velocity": [0.0, 0.0, 0.0],
        "projected_gravity": [0.0, 0.0, -1.0],
        "joint_positions": DEFAULT_POSE.tolist(),
        "joint_velocities": [0.0] * ACTION_DIM,
    }


def main() -> int:
    mujoco_dir = Path(__file__).resolve().parents[1]
    default_dir = mujoco_dir / "policies" / "microduck"
    parser = argparse.ArgumentParser(description="Offline-check Microduck ONNX policies")
    parser.add_argument("--walking", default=str(default_dir / "alpha_walking.onnx"))
    parser.add_argument("--standing", default=str(default_dir / "alpha_stand.onnx"))
    args = parser.parse_args()
    policy = MicroDuckPolicy(args.walking, args.standing)
    for command in ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)):
        targets, observation, active = policy.infer(neutral_state(), command)
        print(
            f"OFFLINE_OK|mode={active}|obs={observation.size}|targets={targets.size}|"
            f"target_rms={math.sqrt(float(np.mean(targets * targets))):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
