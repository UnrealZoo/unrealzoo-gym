"""NumPy tasks for real UE MuJoCo transitions, with explicit legacy/v2 profiles.

The legacy ``ue_v310_velocity`` profile is explicitly reduced; neither profile
claims full native simulator equivalence. No simulation or teacher policy runs
here. The caller owns physics and supplies the actual executed action (raw for
legacy, bounded for v2), with its ``commands``. It must reset completed
environments before stepping again. Returned terminal states
are never replaced with reset states by this class.

Reward formulas and configuration are adapted from mujocolab/mjlab commit
e710cead240b4c0f6f52afaa4f4b2a22c734082c (velocity task and envs/mdp/rewards.py).
Modifications: NumPy implementation, explicit UE state contract, and the reduced
observation/reward/event profile documented by ``task_manifest``.
``ue_keyboard_flat_v2`` restores the source flat rewards and 72D critic using
required version-2 UE telemetry; remaining source differences stay explicit.

Copyright 2025, The mjlab Developers
Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from collections import deque
from copy import deepcopy
import math

import numpy as np


SOURCE_COMMIT = "e710cead240b4c0f6f52afaa4f4b2a22c734082c"
TASK_PROFILE = "ue_v310_velocity"
KEYBOARD_TASK_PROFILE = "ue_keyboard_flat_v2"
TASK_PROFILES = (TASK_PROFILE, KEYBOARD_TASK_PROFILE)
KEYBOARD_ACTION_CLIP = 5.0
JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
DEFAULT_JOINT_POS = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2)
ACTION_SCALE = [0.3727530387, 0.3727530387, 0.2485020258] * 4
JOINT_STIFFNESS = [15.8952426532, 15.8952426532, 35.7642959698] * 4
JOINT_DAMPING = [1.0119225760, 1.0119225760, 2.2768257960] * 4
OBSERVATION_NAMES = (
    "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos",
    "joint_vel", "actions", "command",
)
REWARD_WEIGHTS = {
    "track_linear_velocity": 2.0,
    "track_angular_velocity": 2.0,
    "upright": 1.0,
    "pose": 1.0,
    "action_rate_l2": -0.1,
    "foot_slip": -0.1,
}
KEYBOARD_REWARD_WEIGHTS = {
    **REWARD_WEIGHTS,
    "dof_pos_limits": -1.0,
    "foot_clearance": -2.0,
    "foot_swing_height": -0.25,
    "soft_landing": -1e-5,
}
NOISE_AMPLITUDE = np.array(
    [0.5] * 3 + [0.2] * 3 + [0.05] * 3 + [0.01] * 12
    + [1.5] * 12 + [0.0] * 15,
    dtype=np.float32,
)


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and have shape {shape}")
    return array


def _finite_float32(value, name):
    """Reject overflow instead of hiding a corrupted observation/reward by clipping."""
    array = np.asarray(value, dtype=np.float64)
    invalid = ~np.isfinite(array) | (np.abs(array) > np.finfo(np.float32).max)
    if invalid.any():
        index = tuple(np.argwhere(invalid)[0])
        raise FloatingPointError(
            f"{name} outside finite float32 range at {index}: {array[index]!r}"
        )
    result = array.astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"Nonfinite {name} after float32 conversion")
    return result


class Go1Task:
    """Stateful reward, command, observation and episode bookkeeping.

    ``reset(indices, states)`` accepts either N states or the selected states in
    index order. Each reset state must come directly from a fresh UE reset,
    before any synchronous action: its ``control_targets`` are then the actual
    joint reference subtracted by UE's observation builder. Do not infer this
    reference from the policy's target pose, which differs in v3.1.0.

    The IMU translation defaults to the UE Go1 XML's trunk-relative site offset.
    The caller must verify it against the runtime asset; angular velocity and
    IMU linear velocity give v_root = v_imu - omega cross r_imu.
    """

    def __init__(
        self, num_envs, seed=42, control_dt=0.02, observation_noise=True,
        episode_seconds=20.0, imu_offset=(-0.01592, -0.06659, -0.00617),
        profile=TASK_PROFILE,
        curriculum_step_scale=1,
        command_sampling="source",
        linear_tracking_reward="source",
    ):
        if not isinstance(num_envs, (int, np.integer)) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not math.isclose(control_dt, 0.02, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("ue_v310_velocity requires the verified 20 ms control step")
        if not math.isfinite(episode_seconds) or episode_seconds <= 0:
            raise ValueError("episode_seconds must be positive and finite")
        if profile not in TASK_PROFILES:
            raise ValueError(f"Unknown UE Go1 task profile: {profile}")
        self.profile = profile
        if (isinstance(curriculum_step_scale, bool)
                or not isinstance(curriculum_step_scale, (int, np.integer))
                or curriculum_step_scale < 1):
            raise ValueError("curriculum_step_scale must be a positive integer")
        self.curriculum_step_scale = int(curriculum_step_scale)
        self.keyboard_v2 = profile == KEYBOARD_TASK_PROFILE
        if command_sampling not in ("source", "axis_balanced_v1", "axis_mixed_v2"):
            raise ValueError("Unknown command_sampling profile")
        if command_sampling != "source" and not self.keyboard_v2:
            raise ValueError("Experimental command sampling requires the keyboard v2 task")
        self.command_sampling = command_sampling
        if linear_tracking_reward not in ("source", "precision_v1"):
            raise ValueError("Unknown linear tracking reward")
        if linear_tracking_reward != "source" and not self.keyboard_v2:
            raise ValueError("Experimental tracking reward requires the keyboard v2 task")
        self.linear_tracking_reward = linear_tracking_reward
        self.actor_dim = 48
        self.critic_dim = 72 if self.keyboard_v2 else 48
        self.action_clip = KEYBOARD_ACTION_CLIP if self.keyboard_v2 else None
        self.reward_weights = dict(KEYBOARD_REWARD_WEIGHTS if self.keyboard_v2 else REWARD_WEIGHTS)
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.control_dt = float(control_dt)
        self.episode_seconds = float(episode_seconds)
        self.max_episode_length = math.ceil(episode_seconds / control_dt)
        self.observation_noise = bool(observation_noise)
        self.imu_offset = _finite_array(imu_offset, (3,), "imu_offset")
        self.rng = np.random.default_rng(self.seed)
        self.commands = np.zeros((num_envs, 3), dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int64)
        self.episode_returns = np.zeros(num_envs, dtype=np.float64)
        self.episode_term_sums = {
            name: np.zeros(num_envs) for name in self.reward_weights
        }
        self.common_step_counter = 0
        self.episode_history = deque(maxlen=1000)
        self._last_actions = np.zeros((num_envs, 12), dtype=np.float32)
        self._joint_reference = np.zeros((num_envs, 12), dtype=np.float64)
        self._command_remaining = np.zeros(num_envs)
        self._sim_times = np.full(num_envs, np.nan)
        self._initialized = np.zeros(num_envs, dtype=bool)
        self._terminal = np.zeros(num_envs, dtype=bool)
        self._states = [None] * num_envs
        self._soft_joint_limits = np.full((num_envs, 12, 2), np.nan)
        self._peak_heights = np.zeros((num_envs, 4), dtype=np.float64)
        self._root_heading = np.zeros(num_envs)
        self._heading_target = np.zeros(num_envs)
        self._is_heading_env = np.zeros(num_envs, dtype=bool)
        self._is_standing_env = np.zeros(num_envs, dtype=bool)
        self._initial_lengths_set = False

    @property
    def policy_metadata(self):
        values = {
            "joint_names": ",".join(JOINT_NAMES),
            "observation_names": ",".join(OBSERVATION_NAMES),
            "command_names": "twist",
            "ue_task_profile": self.profile,
            "ue_observation_reference": "reset_control_targets",
        }
        if self.keyboard_v2:
            values["ue_action_clip"] = str(KEYBOARD_ACTION_CLIP)
            values["action_clip"] = str(KEYBOARD_ACTION_CLIP)
        for name, vector in {
            "default_joint_pos": DEFAULT_JOINT_POS,
            "action_scale": ACTION_SCALE,
            "joint_stiffness": JOINT_STIFFNESS,
            "joint_damping": JOINT_DAMPING,
        }.items():
            values[name] = ",".join(f"{value:.10g}" for value in vector)
        if self._initialized.any():
            references = self._joint_reference[self._initialized]
            if not np.allclose(references, references[0], rtol=0.0, atol=1e-6):
                raise ValueError("Cannot export a single UE reference for differing assets")
            values["ue_bridge_default_joint_pos"] = ",".join(
                f"{value:.10g}" for value in references[0]
            )
        return values

    @property
    def task_manifest(self):
        values = {
            "profile": self.profile,
            "source_repository": "https://github.com/mujocolab/mjlab",
            "source_commit": SOURCE_COMMIT,
            "source_task": "Mjlab-Velocity-Flat-Unitree-Go1",
            "equivalent_to_source_task": False,
            "physics_source": "UE MuJoCo synchronous state RPC; no native MJLab sampler",
            "actor_dim": self.actor_dim,
            "critic_dim": self.critic_dim,
            "critic_difference": "No privileged foot inputs; clean actor state only",
            "actor_observation_noise": self.observation_noise,
            "noise_uniform_half_widths": NOISE_AMPLITUDE.tolist(),
            "control_dt": self.control_dt,
            "episode_seconds": self.episode_seconds,
            "max_episode_length": self.max_episode_length,
            "fall_angle_degrees": 70.0,
            "reward_weights": dict(self.reward_weights),
            "reward_scaled_by_dt": True,
            "disabled_source_rewards": {
                "dof_pos_limits": "Runtime soft joint limits are not reported",
                "foot_clearance": "UE single ray is not the source four-ray height sensor",
                "foot_swing_height": (
                    "Missing matching height sensor and substep contact history"
                ),
                "soft_landing": "Missing 3D net contact force and substep first-contact data",
            },
            "source_zero_weight_rewards": ["body_ang_vel", "angular_momentum", "air_time"],
            "disabled_source_events": {
                "foot_friction_slide_spin_roll": "No per-replica friction randomization RPC",
                "encoder_bias": "No bias injection into both sensing and actuation",
                "base_com": "No center-of-mass randomization RPC",
                "push_robot": "No root velocity perturbation RPC",
                "reset_base_random_pose": (
                    "Reset transforms are owned by UE pool, not randomized here"
                ),
            },
            "command_resampling_seconds": [3.0, 8.0],
            "standing_fraction": 0.1,
            "forward_only_fraction": 0.2,
            "heading_command": False,
            "heading_difference": (
                "Source 30% heading-controlled commands disabled; no root yaw field"
            ),
            "command_curriculum_vector_steps": [0, 120000 * self.curriculum_step_scale, 240000 * self.curriculum_step_scale],
            "curriculum_step_scale": self.curriculum_step_scale,
            "command_sampling": self.command_sampling,
            "command_sampling_experiment": (
                {"standing_fraction": 0.1,
                 "single_axis_fraction": 0.5 if self.command_sampling == "axis_balanced_v1" else 0.3,
                 "mixed_fraction": 0.4 if self.command_sampling == "axis_balanced_v1" else 0.6,
                 "single_axis_directions": ["forward", "backward", "left", "right", "yaw_left", "yaw_right"],
                 "linear_magnitude_range": [0.2, 0.7], "yaw_magnitude_range": [0.25, 0.5],
                 "heading_fraction_of_mixed": 0.3,
                 "note": "Balanced command practice experiment; reward and physics unchanged"}
                if self.command_sampling != "source" else None
            ),
            "random_episode_length_initialization": False,
            "joint_observation_conversion": (
                "raw + reset control_targets - policy default_joint_pos"
            ),
            "bridge_joint_references": self._joint_reference[self._initialized].tolist(),
            "imu_offset_in_trunk_m": self.imu_offset.tolist(),
            "linear_reward_frame": (
                "Root link, reconstructed from IMU velocity and angular velocity"
            ),
            "physics_limitations": [
                "v3.1.0 does not expose full physics_config for runtime verification",
                "UE collision snapshot and local support corrections differ from a source infinite plane",
                "Caller must verify IMU offset against runtime MJCF",
            ],
            "policy_metadata": self.policy_metadata,
        }
        if self.keyboard_v2:
            values.update(
                critic_difference="Source flat critic terms restored; telemetry supplied by UE MuJoCo",
                disabled_source_rewards={},
                heading_command=True,
                heading_fraction=0.3,
                heading_control_stiffness=0.5,
                heading_difference="Same source heading/forward/standing masks; NumPy RNG rather than source Torch RNG",
                random_episode_length_initialization=True,
                action_clip=KEYBOARD_ACTION_CLIP,
                action_contract="PPO raw sample clipped to [-5,5] before UE; observations/rewards use the executed clipped action",
                critic_observation_order=["actor_clean_48", "foot_heights_4", "foot_current_air_time_4",
                                          "foot_contacts_4", "signed_log1p_world_contact_forces_12"],
                critic_extra_clip_scale="No additional clip/scale; force transform is sign(F)*log1p(abs(F))",
                contact_force_convention="Source MuJoCo contact sensor world netforce: primary foot onto secondary terrain",
                required_training_telemetry_version=2,
                foot_height_sensor="Minimum of 5 yaw-aligned rays: center plus four radius-0.04m ring rays; max distance 1m",
                first_contact="0 < substep-updated current_contact_time < control_dt + 1e-6",
                swing_peak_reset_difference="Clear selected peak heights on episode reset; fixed source term has no reset hook",
                soft_joint_pos_limit_factor=0.9,
                physics_limitations=[
                    "Version-2 runtime physics configuration must be verified from UE metadata by the caller",
                    "UE collision snapshot and local support corrections differ from a source infinite plane",
                    "Caller must verify IMU offset against runtime MJCF",
                ],
                restored_physics_contract={
                    "requested_profile": "keyboard_v2",
                    "actuator_stiffness": JOINT_STIFFNESS,
                    "actuator_velocity_feedback_damping": JOINT_DAMPING,
                    "passive_dof_damping": 0.0,
                    "joint_frictionloss": 0.0,
                    "target_clipping": "No extra actuator ctrlrange target clamp in opt-in keyboard profile",
                    "verification": "Requested contract; actual runtime confirmation is recorded by the caller",
                },
                requested_reset_pose_distribution={
                    "x_offset_m": [-0.5, 0.5], "y_offset_m": [-0.5, 0.5],
                    "z_offset_m": [0.01, 0.05], "yaw_rad": [-math.pi, math.pi],
                    "verification": "Caller/UE-owned; confirm seed and sampled offsets in environment metadata",
                },
                remaining_source_differences=[
                    "Additional executed-action clipping at 5.0; source clip_actions defaults to None",
                    "Per-episode swing-peak reset prevents old-episode peak leakage",
                    "Friction, encoder-bias, base-COM and push randomization are not implemented by this task",
                    "Reset-pose randomization is caller/UE-owned and must be recorded by the environment",
                    "UE collision snapshots/support corrections differ from the native MJLab flat plane",
                    "CPU MuJoCo in UE replaces the source MJLab simulator; runtime parity requires verification",
                ],
            )
            values["disabled_source_events"]["reset_base_random_pose"] = (
                "Caller/UE-owned; task does not assert a reset distribution without environment evidence"
            )
        if self.command_sampling != "source":
            mix = values["command_sampling_experiment"]
            values.update(forward_only_fraction=mix["single_axis_fraction"] / 6,
                          heading_fraction=mix["mixed_fraction"] * 0.3,
                          heading_difference="Heading applied to 30% of mixed commands only; six pure-axis directions retain explicit targets")
        if self.linear_tracking_reward != "source":
            values["linear_tracking_reward"] = self.linear_tracking_reward
            values["linear_tracking_reward_experiment"] = {
                "squared_error_denominators": [0.25, 0.0625],
                "mixture_weights": [0.5, 0.5],
                "note": "Half original broad kernel plus half precision kernel; original reward weight, peak, other terms and physics unchanged",
            }
            if values.get("command_sampling_experiment"):
                values["command_sampling_experiment"]["note"] = "Command sampler unchanged; see separately declared tracking reward experiment"
        return values

    def _indices(self, indices):
        values = np.asarray(indices)
        if values.size == 0:
            return np.empty(0, dtype=np.int64)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("indices must be a one-dimensional integer sequence")
        if (values < 0).any() or (values >= self.num_envs).any():
            raise IndexError("environment index is out of range")
        if len(np.unique(values)) != len(values):
            raise ValueError("duplicate environment indices")
        return values.astype(np.int64)

    def _validate_state(self, state):
        if state.get("policy_profile") != "velocity" or state.get("synchronous") is not True:
            raise ValueError("Expected a synchronous UE velocity-profile state")
        observation = _finite_array(state["obs"], (48,), "obs")
        sim_time = float(state["sim_time"])
        if not math.isfinite(sim_time) or sim_time < 0:
            raise ValueError("sim_time must be finite and nonnegative")
        _finite_array(state["foot_velocities"], (12,), "foot_velocities")
        contacts = _finite_array(state["foot_contacts"], (4,), "foot_contacts")
        if not np.isin(contacts, (0, 1)).all():
            raise ValueError("foot_contacts must contain booleans or 0/1")
        if not math.isclose(np.linalg.norm(observation[6:9]), 1.0, abs_tol=1e-3):
            raise ValueError("projected gravity must be a unit vector")
        if self.keyboard_v2:
            return self._keyboard_telemetry(state)
        return None

    def _keyboard_telemetry(self, state):
        if type(state.get("training_telemetry_version")) is not int or state["training_telemetry_version"] != 2:
            raise ValueError("ue_keyboard_flat_v2 requires real training_telemetry_version=2")
        telemetry = {
            name: _finite_array(state[name], (size,), name)
            for name, size in (
                ("foot_heights", 4), ("foot_current_air_time", 4),
                ("foot_current_contact_time", 4), ("foot_contact_forces_world", 12),
                ("root_quat_wxyz", 4), ("soft_joint_pos_limits", 24),
            )
        }
        for name in ("foot_current_air_time", "foot_current_contact_time"):
            if (telemetry[name] < 0).any():
                raise ValueError(f"{name} must be nonnegative seconds")
        contacts = _finite_array(state["foot_contacts"], (4,), "foot_contacts").astype(bool)
        if ((telemetry["foot_current_air_time"] > 0) & contacts).any():
            raise ValueError("A contacting foot cannot have positive current_air_time")
        if ((telemetry["foot_current_contact_time"] > 0) & ~contacts).any():
            raise ValueError("An airborne foot cannot have positive current_contact_time")
        limits = telemetry["soft_joint_pos_limits"].reshape(12, 2)
        if (limits[:, 0] >= limits[:, 1]).any():
            raise ValueError("soft_joint_pos_limits must contain ordered lower/upper radians")
        quaternion = telemetry["root_quat_wxyz"]
        if not math.isclose(np.linalg.norm(quaternion), 1.0, abs_tol=1e-3):
            raise ValueError("root_quat_wxyz must be a unit MuJoCo-world quaternion")
        return telemetry

    def _snapshot_state(self, state):
        # Retain independently owned task inputs for partial resets/resume.
        # RPC capabilities, physics asset exports and reset diagnostics belong
        # to the pool and are not consumed by this task.
        fields = (
            "obs", "sim_time", "synchronous", "policy_profile", "control_targets",
            "foot_velocities", "foot_contacts",
        )
        if self.keyboard_v2:
            fields += (
                "training_telemetry_version", "foot_heights", "foot_current_air_time",
                "foot_current_contact_time", "foot_contact_forces_world", "root_quat_wxyz",
                "soft_joint_pos_limits",
            )
        return {key: deepcopy(state[key]) for key in fields if key in state}

    @staticmethod
    def _heading_from_quaternion(quaternion):
        w, x, y, z = quaternion
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _update_heading_commands(self, indices):
        if not self.keyboard_v2 or len(indices) == 0:
            return
        indices = np.asarray(indices, dtype=np.int64)
        heading_ids = indices[self._is_heading_env[indices]]
        error = (self._heading_target[heading_ids] - self._root_heading[heading_ids] + np.pi) % (2 * np.pi) - np.pi
        yaw_limit = 0.7 if self.common_step_counter >= 120000 * self.curriculum_step_scale else 0.5
        self.commands[heading_ids, 2] = np.clip(0.5 * error, -yaw_limit, yaw_limit)
        self.commands[indices[self._is_standing_env[indices]]] = 0.0

    def _resample_commands(self, indices):
        count = len(indices)
        if count == 0:
            return
        vx_range = (-1.0, 1.0)
        yaw_range = (-0.5, 0.5)
        if self.common_step_counter >= 120000 * self.curriculum_step_scale:
            vx_range, yaw_range = (-1.5, 2.0), (-0.7, 0.7)
        if self.common_step_counter >= 240000 * self.curriculum_step_scale:
            vx_range = (-2.0, 3.0)
        if self.command_sampling != "source":
            self._resample_axis_commands(indices, vx_range, yaw_range)
            return
        sampled = self.rng.uniform(
            [vx_range[0], -1.0, yaw_range[0]],
            [vx_range[1], 1.0, yaw_range[1]],
            (count, 3),
        )
        if self.keyboard_v2:
            self._heading_target[indices] = self.rng.uniform(-np.pi, np.pi, count)
            self._is_heading_env[indices] = self.rng.random(count) <= 0.3
        standing = self.rng.random(count) <= 0.1
        forward = self.rng.random(count) <= 0.2
        sampled[forward, 0] = np.maximum(np.abs(sampled[forward, 0]), 0.3)
        sampled[forward, 1:] = 0.0
        sampled[standing] = 0.0
        self.commands[indices] = sampled
        self._command_remaining[indices] = self.rng.uniform(3.0, 8.0, count)
        if self.keyboard_v2:
            self._is_standing_env[indices] = standing
            # Match source ordering: heading can override forward-only yaw;
            # standing is applied last and overrides both.
            self._update_heading_commands(indices)

    def _resample_axis_commands(self, indices, vx_range, yaw_range):
        """Exercise zero cross-axis targets while retaining broad mixed commands."""
        indices = np.asarray(indices, dtype=np.int64)
        count = len(indices)
        sampled = self.rng.uniform([vx_range[0], -1.0, yaw_range[0]],
                                   [vx_range[1], 1.0, yaw_range[1]], (count, 3))
        category = self.rng.random(count)
        standing = category < 0.1
        mixed_start = 0.6 if self.command_sampling == "axis_balanced_v1" else 0.4
        single = (category >= 0.1) & (category < mixed_start)
        direction = self.rng.integers(0, 6, count)
        axis = direction // 2
        magnitude = self.rng.uniform(0.2, 0.7, count)
        angular_magnitude = self.rng.uniform(0.25, 0.5, count)
        magnitude[axis == 2] = angular_magnitude[axis == 2]
        sampled[single] = 0.0
        rows = np.flatnonzero(single)
        sampled[rows, axis[rows]] = magnitude[rows] * np.where(direction[rows] % 2 == 0, 1., -1.)
        sampled[standing] = 0.0
        self.commands[indices] = sampled
        self._heading_target[indices] = self.rng.uniform(-np.pi, np.pi, count)
        # Heading control must not override the explicit zero-yaw target for
        # pure translations, or turn a standing / yaw-only case into another task.
        self._is_heading_env[indices] = (category >= mixed_start) & (self.rng.random(count) <= 0.3)
        self._is_standing_env[indices] = standing
        self._command_remaining[indices] = self.rng.uniform(3.0, 8.0, count)
        self._update_heading_commands(indices)

    def reset(self, indices, states):
        """Accept fresh physics resets; preserve all unselected episode histories."""
        indices = self._indices(indices)
        if len(states) == self.num_envs:
            selected = [states[index] for index in indices]
        elif len(states) == len(indices):
            selected = list(states)
        else:
            raise ValueError("states must contain N states or one per selected index")
        references, selected_telemetry = [], []
        for state in selected:
            selected_telemetry.append(self._validate_state(state))
            if abs(float(state["sim_time"])) > 1e-8:
                raise ValueError("reset calibration requires sim_time=0")
            if not np.allclose(state["obs"][33:45], 0.0, atol=1e-7):
                raise ValueError("reset requires untouched zero-action UE reset states")
            references.append(
                _finite_array(state["control_targets"], (12,), "control_targets")
            )
        # Validate the whole batch before changing any episode bookkeeping.
        for index, state, reference, telemetry in zip(indices, selected, references, selected_telemetry):
            self._joint_reference[index] = reference
            self._sim_times[index] = float(state["sim_time"])
            self._states[index] = self._snapshot_state(state)
            if self.keyboard_v2:
                self._soft_joint_limits[index] = telemetry["soft_joint_pos_limits"].reshape(12, 2)
                self._root_heading[index] = self._heading_from_quaternion(telemetry["root_quat_wxyz"])
                self._peak_heights[index] = 0.0
        self.episode_lengths[indices] = 0
        self.episode_returns[indices] = 0.0
        self._last_actions[indices] = 0.0
        self._terminal[indices] = False
        self._initialized[indices] = True
        for values in self.episode_term_sums.values():
            values[indices] = 0.0
        self._resample_commands(indices)
        return self.observe(self._states) if self._initialized.all() else None

    def initialize_episode_lengths(self, lengths=None):
        """Set source-style random initial timeout phases before any physics step.

        The caller must synchronize its runner's episode_length_buf with the
        returned array. Rewards, actions, commands and simulator state stay fresh.
        """
        if (not self._initialized.all() or self.episode_lengths.any()
                or np.any(np.abs(self._sim_times) > 1e-8) or self._initial_lengths_set):
            raise RuntimeError("Initial episode lengths may be set once, immediately after all fresh resets")
        values = self.rng.integers(0, self.max_episode_length, self.num_envs) if lengths is None else np.asarray(lengths)
        if (values.shape != (self.num_envs,) or not np.issubdtype(values.dtype, np.integer)
                or (values < 0).any() or (values >= self.max_episode_length).any()):
            raise ValueError("Initial episode lengths must be integer N-vector in [0, max_episode_length)")
        self.episode_lengths[:] = values
        self._initial_lengths_set = True
        return self.episode_lengths.copy()

    def restore_common_step_counter(self, value):
        """Restore command curriculum after fresh resets when loading PPO weights.

        This resumes the curriculum clock, not a mid-episode physics/RNG snapshot.
        Commands are resampled immediately at the restored stage so the first
        rollout does not keep the stage-zero commands sampled during construction.
        """
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("common_step_counter must be a nonnegative integer")
        if value < 0:
            raise ValueError("common_step_counter must be a nonnegative integer")
        # Source-style initial timeout phases are bookkeeping, not elapsed
        # physics. Preserve them when restoring a freshly constructed runner.
        if not self._initialized.all() or np.any(np.abs(self._sim_times) > 1e-8):
            raise RuntimeError("Restore curriculum only immediately after all UE resets")
        self.common_step_counter = int(value)
        self._resample_commands(np.arange(self.num_envs))
        return self.observe(self._states)

    def set_commands(self, commands):
        """Hold explicit body-frame commands for deterministic evaluation."""
        self.commands[:] = _finite_array(commands, (self.num_envs, 3), "commands")
        self._command_remaining[:] = np.inf
        self._is_heading_env[:] = False
        self._is_standing_env[:] = False

    def _validated_observation_inputs(self, states):
        if not self._initialized.all() or len(states) != self.num_envs:
            raise ValueError("All environments must be initialized and supplied")
        telemetry = [self._validate_state(state) for state in states]
        clean = np.asarray([state["obs"] for state in states], dtype=np.float64)
        clean[:, 9:21] += self._joint_reference - DEFAULT_JOINT_POS
        clean[:, 33:45] = self._last_actions
        clean[:, 45:48] = self.commands
        return clean, telemetry

    def observe(self, states):
        # Public callers still receive the complete state/telemetry validation.
        clean, telemetry = self._validated_observation_inputs(states)
        contacts = np.asarray([state["foot_contacts"] for state in states], dtype=np.float64)
        return self._observe_validated(clean, telemetry, contacts)

    def _observe_validated(self, clean, telemetry, contacts):
        clean = _finite_float32(clean, "clean observations")
        actor = clean.copy()
        if self.observation_noise:
            actor += self.rng.uniform(-NOISE_AMPLITUDE, NOISE_AMPLITUDE, actor.shape)
        actor = _finite_float32(actor, "actor observations")
        critic = clean
        if self.keyboard_v2:
            forces = np.asarray([item["foot_contact_forces_world"] for item in telemetry])
            critic = _finite_float32(np.concatenate((
                clean,
                np.asarray([item["foot_heights"] for item in telemetry]),
                np.asarray([item["foot_current_air_time"] for item in telemetry]),
                contacts,
                np.sign(forces) * np.log1p(np.abs(forces)),
            ), axis=1), "critic observations")
        return {"actor": actor, "critic": critic}

    def step(self, states, actions):
        """Process exactly one completed 20 ms physics step, before any reset."""
        if self._terminal.any():
            raise RuntimeError("Completed environments must be reset before another step")
        actions = _finite_array(actions, (self.num_envs, 12), "actions")
        if self.keyboard_v2 and (np.abs(actions) > KEYBOARD_ACTION_CLIP + 1e-6).any():
            raise ValueError("ue_keyboard_flat_v2 requires executed actions clipped to [-5,5] by its caller")
        clean, telemetry = self._validated_observation_inputs(states)
        times = np.asarray([state["sim_time"] for state in states], dtype=np.float64)
        if not np.allclose(times - self._sim_times, self.control_dt, rtol=0, atol=1e-6):
            raise RuntimeError("Each UE state must advance exactly one 20 ms control step")
        reported_actions = np.asarray([state["obs"][33:45] for state in states])
        if not np.allclose(reported_actions, actions, rtol=1e-5, atol=1e-6):
            raise RuntimeError("UE returned a different last action from the submitted action")

        root_velocity = clean[:, :3] - np.cross(clean[:, 3:6], self.imu_offset)
        linear_error = np.sum((root_velocity[:, :2] - self.commands[:, :2]) ** 2, axis=1)
        linear_error += root_velocity[:, 2] ** 2
        angular_error = np.sum(clean[:, 3:5] ** 2, axis=1)
        angular_error += (clean[:, 5] - self.commands[:, 2]) ** 2
        command_speed = np.linalg.norm(self.commands[:, :2], axis=1) + np.abs(self.commands[:, 2])
        posture_std = np.where(
            (command_speed < 0.05)[:, None],
            np.array([0.05, 0.05, 0.1] * 4),
            np.array([0.3, 0.3, 0.6] * 4),
        )
        foot_velocity = np.asarray(
            [state["foot_velocities"] for state in states]
        ).reshape(-1, 4, 3)
        foot_contacts = np.asarray([state["foot_contacts"] for state in states])
        raw_terms = {
            "track_linear_velocity": np.exp(-linear_error / 0.25),
            "track_angular_velocity": np.exp(-angular_error / 0.5),
            "upright": np.exp(-np.sum(clean[:, 6:8] ** 2, axis=1) / 0.2),
            "pose": np.exp(-np.mean((clean[:, 9:21] / posture_std) ** 2, axis=1)),
            "action_rate_l2": np.sum((actions - self._last_actions) ** 2, axis=1),
            "foot_slip": np.sum(
                np.sum(foot_velocity[:, :, :2] ** 2, axis=2) * foot_contacts, axis=1,
            ) * (command_speed > 0.05),
        }
        if self.linear_tracking_reward == "precision_v1":
            raw_terms["track_linear_velocity"] = (
                0.5 * raw_terms["track_linear_velocity"] + 0.5 * np.exp(-linear_error / 0.0625)
            )
        next_peak_heights = None
        keyboard_log = {}
        if self.keyboard_v2:
            limits = np.asarray([item["soft_joint_pos_limits"].reshape(12, 2) for item in telemetry])
            if not np.allclose(limits, self._soft_joint_limits, rtol=0, atol=1e-10):
                raise RuntimeError("Static soft_joint_pos_limits changed without an explicit reset")
            joint_positions = clean[:, 9:21] + DEFAULT_JOINT_POS
            foot_heights = np.asarray([item["foot_heights"] for item in telemetry])
            contact_time = np.asarray([item["foot_current_contact_time"] for item in telemetry])
            forces = np.asarray([item["foot_contact_forces_world"] for item in telemetry]).reshape(-1, 4, 3)
            first_contact = (contact_time > 0.0) & (contact_time < self.control_dt + 1e-6)
            active = command_speed > 0.05
            peaks = np.where(foot_contacts == 0, np.maximum(self._peak_heights, foot_heights), self._peak_heights)
            force_magnitude = np.linalg.norm(forces, axis=2)
            raw_terms.update({
                "dof_pos_limits": np.sum(
                    np.maximum(limits[:, :, 0] - joint_positions, 0)
                    + np.maximum(joint_positions - limits[:, :, 1], 0), axis=1,
                ),
                "foot_clearance": np.sum(
                    np.abs(foot_heights - 0.1) * np.linalg.norm(foot_velocity[:, :, :2], axis=2), axis=1,
                ) * active,
                "foot_swing_height": np.sum((peaks / 0.1 - 1.0) ** 2 * first_contact, axis=1) * active,
                "soft_landing": np.sum(force_magnitude * first_contact, axis=1) * active,
            })
            next_peak_heights = np.where(first_contact, 0.0, peaks)
            landing_count = max(int(first_contact.sum()), 1)
            keyboard_log = {
                "Metrics/peak_height_mean": float(np.sum(peaks * first_contact) / landing_count),
                "Metrics/landing_force_mean": float(np.sum(force_magnitude * first_contact) / landing_count),
            }
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terms = {}
        for name, raw in raw_terms.items():
            terms[name] = raw * self.reward_weights[name] * self.control_dt
            rewards += terms[name]
        rewards_float32 = _finite_float32(rewards, "rewards")

        self._sim_times = times.copy()
        self._states = [self._snapshot_state(state) for state in states]
        self._last_actions[:] = actions
        if self.keyboard_v2:
            self._peak_heights[:] = next_peak_heights
            self._root_heading[:] = [self._heading_from_quaternion(item["root_quat_wxyz"]) for item in telemetry]
        self.episode_lengths += 1
        self.episode_returns += rewards
        self.common_step_counter += 1
        for name, values in terms.items():
            self.episode_term_sums[name] += values
        terminated = -clean[:, 8] < math.cos(math.radians(70.0))
        truncated = (self.episode_lengths >= self.max_episode_length) & ~terminated
        self._terminal = terminated | truncated
        completed = []
        for index in np.flatnonzero(self._terminal):
            episode = {
                "env_index": int(index),
                "return": float(self.episode_returns[index]),
                "length": int(self.episode_lengths[index]),
                "seconds": float(self.episode_lengths[index] * self.control_dt),
                "terminated": bool(terminated[index]),
                "truncated": bool(truncated[index]),
                "reward_terms": {
                    name: float(values[index]) for name, values in self.episode_term_sums.items()
                },
            }
            self.episode_history.append(episode)
            completed.append(episode)
        metrics = {
            "reward_terms": terms,
            "completed_episodes": completed,
            "tracking_mae": np.mean(
                np.abs(
                    np.column_stack((root_velocity[:, :2], clean[:, 5])) - self.commands
                ),
                axis=0,
            ),
            "log": {f"Reward/{name}": float(values.mean()) for name, values in terms.items()},
        }
        metrics["log"].update(keyboard_log)
        self._command_remaining -= self.control_dt
        # Keep the command associated with terminal states for bootstrap/evaluation.
        self._resample_commands(np.flatnonzero((self._command_remaining <= 0) & ~self._terminal))
        self._update_heading_commands(np.flatnonzero(~self._terminal))
        # Reward used the incoming command and previous action. Observations
        # retain the original ordering: publish the newly executed action and
        # resampled/heading-updated command, then draw actor noise exactly once.
        clean[:, 33:45] = self._last_actions
        clean[:, 45:48] = self.commands
        return self._observe_validated(clean, telemetry, foot_contacts), rewards_float32, terminated, truncated, metrics
