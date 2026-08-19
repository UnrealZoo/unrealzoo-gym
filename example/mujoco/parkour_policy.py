#!/usr/bin/env python3
"""Robot Parkour Learning policy adapter for the UnrealCV Go1 MuJoCo bridge."""
from __future__ import annotations

import json
import math
import sys
from contextlib import redirect_stdout
from collections import OrderedDict
from io import StringIO
from pathlib import Path
from typing import Sequence


ACTION_DIM = 12
PROPRIO_DIM = 48

# UnrealCV's MuJoCo bridge uses FR, FL, RR, RL. Robot Parkour Learning uses
# FL, FR, RL, RR. Each entry expands to hip, thigh, calf.
BRIDGE_TO_POLICY = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
POLICY_TO_BRIDGE = BRIDGE_TO_POLICY
BRIDGE_OBSERVATION_DEFAULT_JOINT_POS = (
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
)
# StepSynchronousPolicy converts its normalized input with these C++ affine
# offsets.  They intentionally differ from the MuJoCo home-keyframe values
# above, so observation reconstruction and action conversion must not share a
# single "bridge default" array.
BRIDGE_ACTION_DEFAULT_JOINT_POS = (
    -0.1, 0.8, -1.5,
    0.1, 0.8, -1.5,
    -0.1, 1.0, -1.5,
    0.1, 1.0, -1.5,
)
BRIDGE_SYNC_ACTION_SCALE = (
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
)
POLICY_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)


def _latest_checkpoint(policy_dir: Path) -> Path:
    checkpoints = list(policy_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in {policy_dir}")

    def iteration(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(checkpoints, key=iteration)


def _as_action_scale(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != ACTION_DIM:
            raise RuntimeError(f"Expected {ACTION_DIM} action scales, got {len(value)}")
        return [float(item) for item in value]
    return [float(value)] * ACTION_DIM


class RobotParkourPolicy:
    """Loads the official visual encoder + GRU actor and maps bridge I/O."""

    def __init__(
        self,
        policy_dir: str | Path,
        checkpoint: str | Path | None = None,
        runtime_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        self.policy_dir = Path(policy_dir).expanduser().resolve()
        self.config_path = self.policy_dir / "config.json"
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Robot Parkour config not found: {self.config_path}")
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

        runtime_root = (
            Path(runtime_dir).expanduser().resolve()
            if runtime_dir
            else self.policy_dir / "rsl_rl"
        )
        if not (runtime_root / "rsl_rl" / "modules").is_dir():
            raise FileNotFoundError(
                f"Official rsl_rl runtime not found under {runtime_root}. "
                "Expected <runtime>/rsl_rl/modules."
            )
        if str(runtime_root) not in sys.path:
            sys.path.insert(0, str(runtime_root))

        try:
            import numpy as np
            import torch
            import torch.nn.functional as functional
            from rsl_rl import modules
        except ImportError as exc:
            raise RuntimeError(
                "Robot Parkour inference requires numpy, torch, and the bundled official rsl_rl runtime"
            ) from exc

        self.np = np
        self.torch = torch
        self.functional = functional
        self.device = torch.device(device)
        self.checkpoint_path = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint
            else _latest_checkpoint(self.policy_dir)
        )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        camera_config = self.config["sensor"]["forward_camera"]
        self.input_height, self.input_width = camera_config.get(
            "output_resolution", camera_config["resolution"]
        )
        self.capture_height, self.capture_width = camera_config["resolution"]
        self.depth_min, self.depth_max = map(float, camera_config["depth_range"])
        self.crop_left, self.crop_right = map(int, camera_config["crop_left_right"])
        self.crop_top, self.crop_bottom = map(int, camera_config["crop_top_bottom"])
        self.depth_refresh_seconds = float(camera_config.get("refresh_duration", 0.1))
        latency_range = camera_config.get("latency_range", [0.0, 0.0])
        self.depth_latency_seconds = 0.5 * (float(latency_range[0]) + float(latency_range[1]))
        fov_range = camera_config.get("horizontal_fov", [86.0, 86.0])
        self.horizontal_fov_degrees = 0.5 * (float(fov_range[0]) + float(fov_range[1]))
        position = camera_config.get("position", {}).get("mean", [0.272, 0.0075, 0.092])
        self.camera_position_m = tuple(float(value) for value in position)
        rotation = camera_config.get("rotation", {})
        rotation_lower = rotation.get("lower", [0.0, 0.5, 0.0])
        rotation_upper = rotation.get("upper", [0.0, 0.54, 0.0])
        self.camera_pitch_degrees = -math.degrees(
            0.5 * (float(rotation_lower[1]) + float(rotation_upper[1]))
        )

        checkpoint_payload = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
        critic_weight = state_dict.get("memory_c.rnn.weight_ih_l0")
        critic_obs_dim = int(critic_weight.shape[1]) if critic_weight is not None else PROPRIO_DIM

        obs_segments = OrderedDict(
            [
                ("proprioception", (PROPRIO_DIM,)),
                ("forward_depth", (1, self.input_height, self.input_width)),
            ]
        )
        raw_actor_obs_dim = PROPRIO_DIM + self.input_height * self.input_width
        policy_class_name = self.config["runner"]["policy_class_name"]
        policy_class = getattr(modules, policy_class_name)
        # The upstream constructors print the full actor/critic architecture.
        # Keep normal demo output compact while preserving the official module.
        with redirect_stdout(StringIO()):
            self.model = policy_class(
                num_actor_obs=raw_actor_obs_dim,
                num_critic_obs=critic_obs_dim,
                num_actions=ACTION_DIM,
                obs_segments=obs_segments,
                privileged_obs_segments=None,
                **self.config["policy"],
            )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        default_angles = self.config["init_state"]["default_joint_angles"]
        self.policy_default_joint_pos = [float(default_angles[name]) for name in POLICY_JOINT_NAMES]
        self.policy_action_scale = _as_action_scale(self.config["control"]["action_scale"])
        normalization = self.config["normalization"]
        self.clip_actions = float(normalization.get("clip_actions", 100.0))
        self.hard_clip = normalization.get("clip_actions_method") == "hard"
        self.clip_actions_low = [float(value) for value in normalization.get(
            "clip_actions_low", [-self.clip_actions] * ACTION_DIM
        )]
        self.clip_actions_high = [float(value) for value in normalization.get(
            "clip_actions_high", [self.clip_actions] * ACTION_DIM
        )]
        obs_scales = normalization["obs_scales"]
        self.angular_velocity_scale = float(obs_scales["ang_vel"])
        self.joint_position_scale = float(obs_scales["dof_pos"])
        self.joint_velocity_scale = float(obs_scales["dof_vel"])
        self.command_scale = float(obs_scales["lin_vel"])
        self.bridge_observation_default_joint_pos = list(
            BRIDGE_OBSERVATION_DEFAULT_JOINT_POS
        )
        self.bridge_action_default_joint_pos = list(BRIDGE_ACTION_DEFAULT_JOINT_POS)
        self.bridge_action_scale = list(BRIDGE_SYNC_ACTION_SCALE)
        self.reset()

    def calibrate_bridge(
        self,
        action_default_joint_targets: Sequence[float],
        action_scales: Sequence[float] | None = None,
        observation_default_joint_positions: Sequence[float] | None = None,
    ) -> None:
        """Use independently measured values from the running UE bridge."""
        if len(action_default_joint_targets) != ACTION_DIM:
            raise ValueError(f"Expected {ACTION_DIM} bridge action default targets")
        defaults = [float(value) for value in action_default_joint_targets]
        if not all(math.isfinite(value) for value in defaults):
            raise ValueError("Bridge action default targets must be finite")
        self.bridge_action_default_joint_pos = defaults
        if action_scales is not None:
            if len(action_scales) != ACTION_DIM:
                raise ValueError(f"Expected {ACTION_DIM} bridge action scales")
            scales = [float(value) for value in action_scales]
            if not all(math.isfinite(value) and abs(value) > 1e-8 for value in scales):
                raise ValueError("Bridge action scales must be finite and non-zero")
            self.bridge_action_scale = scales
        if observation_default_joint_positions is not None:
            if len(observation_default_joint_positions) != ACTION_DIM:
                raise ValueError(f"Expected {ACTION_DIM} bridge observation defaults")
            observation_defaults = [
                float(value) for value in observation_default_joint_positions
            ]
            if not all(math.isfinite(value) for value in observation_defaults):
                raise ValueError("Bridge observation defaults must be finite")
            self.bridge_observation_default_joint_pos = observation_defaults

    def describe(self) -> str:
        return (
            f"RobotParkourLearning checkpoint={self.checkpoint_path.name} "
            f"depth={self.input_height}x{self.input_width}@{1.0 / self.depth_refresh_seconds:.1f}Hz "
            f"latency={self.depth_latency_seconds:.3f}s device={self.device}"
        )

    def reset(self) -> None:
        # The upstream reset implementation performs an in-place write that is
        # incompatible with some newer torch autograd view rules. Clearing the
        # inference hidden states is equivalent and version-independent.
        if hasattr(self.model, "memory_a"):
            self.model.memory_a.hidden_states = None
        if hasattr(self.model, "memory_c"):
            self.model.memory_c.hidden_states = None

    def preprocess_depth(self, depth_m, crop_far_m: float | None = None):
        depth = self.torch.as_tensor(depth_m, dtype=self.torch.float32, device=self.device)
        if depth.ndim != 2:
            raise ValueError(f"Expected an HxW depth image, got shape {tuple(depth.shape)}")
        depth = depth.unsqueeze(0).unsqueeze(0)
        depth = self.torch.nan_to_num(
            depth,
            nan=self.depth_max,
            posinf=self.depth_max,
            neginf=self.depth_min,
        )
        crop_far = self.depth_max if crop_far_m is None else float(crop_far_m)
        depth = self.torch.where(depth > crop_far, self.depth_max, depth)
        depth = self.torch.clamp(depth, self.depth_min, self.depth_max)
        depth = (depth - self.depth_min) / max(self.depth_max - self.depth_min, 1e-6)

        # Match the official Go1 deployment filter, including its one-pixel
        # bottom/right exclusion before adaptive average pooling.
        bottom_stop = -(self.crop_bottom + 1)
        right_stop = -(self.crop_right + 1)
        depth = depth[
            :,
            :,
            self.crop_top:bottom_stop,
            self.crop_left:right_stop,
        ]
        if depth.shape[-2] <= 0 or depth.shape[-1] <= 0:
            raise ValueError(
                f"Depth crop is empty: input={tuple(depth_m.shape)}, "
                f"crop=({self.crop_top},{self.crop_bottom},{self.crop_left},{self.crop_right})"
            )
        return self.functional.adaptive_avg_pool2d(
            depth, (self.input_height, self.input_width)
        )

    def build_proprioception(
        self,
        bridge_observation: Sequence[float],
        command: Sequence[float],
        last_policy_action: Sequence[float],
    ) -> list[float]:
        if len(bridge_observation) != PROPRIO_DIM:
            raise ValueError(f"Expected {PROPRIO_DIM}D bridge observation")
        if len(command) != 3 or len(last_policy_action) != ACTION_DIM:
            raise ValueError("Expected a 3D command and 12D last policy action")

        bridge_qpos = [
            float(bridge_observation[9 + index])
            + self.bridge_observation_default_joint_pos[index]
            for index in range(ACTION_DIM)
        ]
        policy_qpos = [bridge_qpos[index] for index in BRIDGE_TO_POLICY]
        policy_qvel = [float(bridge_observation[21 + index]) for index in BRIDGE_TO_POLICY]
        proprio = (
            [0.0, 0.0, 0.0]
            + [float(value) * self.angular_velocity_scale for value in bridge_observation[3:6]]
            + [float(value) for value in bridge_observation[6:9]]
            + [float(value) * self.command_scale for value in command]
            + [
                (policy_qpos[index] - self.policy_default_joint_pos[index])
                * self.joint_position_scale
                for index in range(ACTION_DIM)
            ]
            + [value * self.joint_velocity_scale for value in policy_qvel]
            + [float(value) for value in last_policy_action]
        )
        if len(proprio) != PROPRIO_DIM:
            raise AssertionError(f"Built invalid proprioception length {len(proprio)}")
        return proprio

    def infer_preprocessed(self, proprioception: Sequence[float], processed_depth) -> list[float]:
        if len(proprioception) != PROPRIO_DIM:
            raise ValueError(f"Expected {PROPRIO_DIM}D proprioception")
        depth = processed_depth.to(device=self.device, dtype=self.torch.float32)
        if tuple(depth.shape) != (1, 1, self.input_height, self.input_width):
            raise ValueError(f"Unexpected processed depth shape {tuple(depth.shape)}")
        proprio = self.torch.as_tensor(
            proprioception, dtype=self.torch.float32, device=self.device
        ).reshape(1, PROPRIO_DIM)
        observation = self.torch.cat((proprio, depth.reshape(1, -1)), dim=1)
        with self.torch.inference_mode():
            action = self.model.act_inference(observation).reshape(-1)
        if action.numel() != ACTION_DIM:
            raise RuntimeError(f"Policy produced {action.numel()} actions")
        return action.detach().cpu().tolist()

    def clip_policy_action(self, action: Sequence[float]) -> list[float]:
        if len(action) != ACTION_DIM:
            raise ValueError(f"Expected {ACTION_DIM}D policy action")
        result = []
        for index, value in enumerate(action):
            value = float(value)
            if not math.isfinite(value):
                raise RuntimeError(f"Policy produced a non-finite action at index {index}")
            value = max(-self.clip_actions, min(self.clip_actions, value))
            if self.hard_clip:
                value = max(self.clip_actions_low[index], min(self.clip_actions_high[index], value))
            result.append(value)
        return result

    def to_bridge_sync_action(self, policy_action: Sequence[float]) -> tuple[list[float], list[float]]:
        clipped_policy_action = self.clip_policy_action(policy_action)
        policy_targets = [
            self.policy_default_joint_pos[index]
            + clipped_policy_action[index] * self.policy_action_scale[index]
            for index in range(ACTION_DIM)
        ]
        bridge_targets = [policy_targets[index] for index in POLICY_TO_BRIDGE]
        bridge_action = [
            (bridge_targets[index] - self.bridge_action_default_joint_pos[index])
            / self.bridge_action_scale[index]
            for index in range(ACTION_DIM)
        ]
        return bridge_action, clipped_policy_action

    def bridge_home_action(self) -> list[float]:
        """Raw bridge action that preserves the untouched MuJoCo home pose."""
        return [
            (
                self.bridge_observation_default_joint_pos[index]
                - self.bridge_action_default_joint_pos[index]
            )
            / self.bridge_action_scale[index]
            for index in range(ACTION_DIM)
        ]
