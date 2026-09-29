"""OpenAI Gym interface for UnrealCV MuJoCo robots."""
import json
import math

import numpy as np
from gym import spaces
from gym_unrealcv.envs.base_env_mujoco import UnrealCvMujocoBase


ROBOT_SPECS = {
    "go1": {
        "blueprint": "/Game/robot-dog-unitree-go1/BP_UnitreeGo1.BP_UnitreeGo1",
        "observation_dim": 48,
        "action_dim": 12,
        "settle_mode": "simple",
        "settle_offset_cm": 35.0,
        "camera_forward_cm": 300.0,
        "camera_up_cm": 100.0,
    },
    "g1": {
        "blueprint": "/Game/robot-humanoid-unitree-g1/BP_UnitreeG1.BP_UnitreeG1",
        "observation_dim": 480,
        "action_dim": 29,
        "settle_mode": "simple",
        "settle_offset_cm": 0.0,
        "camera_forward_cm": 300.0,
        "camera_up_cm": 100.0,
    },
    "microduck": {
        "blueprint": "/Game/robot-biped-microduck/BP_MicroDuck.BP_MicroDuck",
        "observation_dim": 61,
        "action_dim": 14,
        "settle_mode": "bounds",
        "settle_offset_cm": 0.0,
        "camera_forward_cm": 250.0,
        "camera_up_cm": 100.0,
    },
}

MICRODUCK_DEFAULT_POSE = np.asarray(
    [
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,
        0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530,
    ],
    dtype=np.float32,
)
MICRODUCK_LEG_INDICES = np.asarray((0, 1, 2, 3, 4, 9, 10, 11, 12, 13))


class UnrealCvMujocoEnv(UnrealCvMujocoBase):
    """Single-robot MuJoCo environment for attached PIE or packaged UnrealZoo."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        robot,
        host="127.0.0.1",
        port=9000,
        actor_name="",
        spawn_location=None,
        spawn_camera_id="0",
        spawn_camera_forward_cm=None,
        spawn_camera_right_cm=0.0,
        spawn_camera_up_cm=None,
        spawn_yaw_offset=0.0,
        settle_trace_start_cm=100.0,
        settle_trace_length_cm=5000.0,
        keep_actor=False,
        setting_file=None,
        physics_config=None,
        launch=None,
        resolution=(160, 160),
        sleep_time=None,
        nullrhi=None,
        request_timeout=120,
    ):
        super().__init__(
            robot=robot,
            setting_file=setting_file,
            physics_config=physics_config,
            host=host,
            port=port,
            launch=launch,
            resolution=resolution,
            sleep_time=sleep_time,
            nullrhi=nullrhi,
            request_timeout=request_timeout,
        )
        self.specification = dict(ROBOT_SPECS[robot])
        for key in self.specification:
            if key in self.robot_setting:
                self.specification[key] = self.robot_setting[key]
        self.actor_name = actor_name
        self.owns_actor = not bool(actor_name)
        self.spawn_location = (
            self.robot_setting.get(
                "spawn_location", self.mujoco_setting.get("spawn_location")
            )
            if spawn_location is None
            else spawn_location
        )
        self.spawn_camera_id = spawn_camera_id
        self.spawn_camera_forward_cm = (
            self.specification["camera_forward_cm"]
            if spawn_camera_forward_cm is None
            else spawn_camera_forward_cm
        )
        self.spawn_camera_right_cm = spawn_camera_right_cm
        self.spawn_camera_up_cm = (
            self.specification["camera_up_cm"]
            if spawn_camera_up_cm is None
            else spawn_camera_up_cm
        )
        self.spawn_yaw_offset = spawn_yaw_offset
        self.settle_trace_start_cm = settle_trace_start_cm
        self.settle_trace_length_cm = settle_trace_length_cm
        self.keep_actor = keep_actor
        self.command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.specification["action_dim"], dtype=np.float32)
        self.previous_targets = None
        self.started = False
        self.physics_configured = False
        self.steps = 0
        self._reset_location = None
        self._reset_rotation = None

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.specification["action_dim"],),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.specification["observation_dim"],),
            dtype=np.float32,
        )

    def _camera_spawn_pose(self):
        location = np.asarray(
            self.request(
                "vget /camera/{}/location".format(self.spawn_camera_id)
            ).replace(",", " ").split(),
            dtype=np.float32,
        )
        rotation = np.asarray(
            self.request(
                "vget /camera/{}/rotation".format(self.spawn_camera_id)
            ).replace(",", " ").split(),
            dtype=np.float32,
        )
        yaw = math.radians(float(rotation[1]))
        spawn = np.asarray(
            [
                location[0]
                + math.cos(yaw) * self.spawn_camera_forward_cm
                - math.sin(yaw) * self.spawn_camera_right_cm,
                location[1]
                + math.sin(yaw) * self.spawn_camera_forward_cm
                + math.cos(yaw) * self.spawn_camera_right_cm,
                location[2] + self.spawn_camera_up_cm,
            ],
            dtype=np.float32,
        )
        return spawn, float(rotation[1] + self.spawn_yaw_offset)

    def _spawn_actor(self):
        if self.spawn_location is None:
            location, yaw = self._camera_spawn_pose()
        else:
            location = np.asarray(self.spawn_location, dtype=np.float32)
            yaw = self.spawn_yaw_offset

        self.actor_name = self.request(
            "vset /objects/spawn_from_path {} {:.6f} {:.6f} {:.6f}".format(
                self.specification["blueprint"],
                location[0],
                location[1],
                location[2],
            )
        )
        self.request(
            "vset /object/{}/rotation 0 {:.6f} 0".format(self.actor_name, yaw)
        )
        self.request(
            "vset /object/{}/settle_to_ground {} {:.6f} {:.6f} {:.6f}".format(
                self.actor_name,
                self.specification["settle_mode"],
                self.settle_trace_start_cm,
                self.settle_trace_length_cm,
                self.specification["settle_offset_cm"],
            )
        )
        self.configure_mujoco_physics(self.actor_name)
        self.physics_configured = True

    @staticmethod
    def _parse_actor_vector(response, field_name):
        values = np.asarray(
            str(response).replace(",", " ").split(), dtype=np.float64
        )
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise RuntimeError(
                "Invalid {} returned by UnrealCV: {}".format(
                    field_name, response
                )
            )
        return values

    def _capture_reset_transform(self):
        """Capture the authored/settled pose before MuJoCo can move the Actor."""
        self._reset_location = self._parse_actor_vector(
            self.request("vget /object/{}/location".format(self.actor_name)),
            "actor location",
        )
        self._reset_rotation = self._parse_actor_vector(
            self.request("vget /object/{}/rotation".format(self.actor_name)),
            "actor rotation",
        )

    def _restore_reset_transform(self):
        """Restore the immutable spawn pose before rebuilding MuJoCo state."""
        if self._reset_location is None or self._reset_rotation is None:
            raise RuntimeError("MuJoCo reset transform has not been captured")
        location = " ".join(
            "{:.9f}".format(float(value)) for value in self._reset_location
        )
        rotation = " ".join(
            "{:.9f}".format(float(value)) for value in self._reset_rotation
        )
        self.request(
            "vset /object/{}/location {}".format(self.actor_name, location)
        )
        self.request(
            "vset /object/{}/rotation {}".format(self.actor_name, rotation)
        )

    def _send_command(self):
        values = " ".join("{:.6f}".format(float(value)) for value in self.command)
        if self.robot == "go1":
            endpoint = "mujoco_go1_policy_command"
        elif self.robot == "g1":
            endpoint = "mujoco_g1_policy_command"
        else:
            return
        self.request(
            "vset /object/{}/{} {}".format(self.actor_name, endpoint, values)
        )

    def set_command(self, command):
        self.command = np.asarray(command, dtype=np.float32)
        if self.actor_name:
            self._send_command()

    def _microduck_observation(self, state):
        command = np.concatenate(
            (self.command, np.zeros(10, dtype=np.float32))
        )
        return np.concatenate(
            (
                np.asarray(state["base_angular_velocity"], dtype=np.float32),
                np.asarray(state["projected_gravity"], dtype=np.float32),
                np.asarray(state["joint_positions"], dtype=np.float32)
                - MICRODUCK_DEFAULT_POSE,
                np.asarray(state["joint_velocities"], dtype=np.float32),
                self.last_action,
                command,
            )
        ).astype(np.float32)

    def _start_robot(self):
        if self.robot == "go1":
            self.request(
                "vset /object/{}/mujoco_quadruped_pose_preview/start go1".format(
                    self.actor_name
                )
            )
            self._send_command()
            payload = json.loads(
                self.request(
                    "vset /object/{}/mujoco_go1_policy_sync/start".format(
                        self.actor_name
                    )
                )
            )
            return np.asarray(payload["obs"], dtype=np.float32), payload

        if self.robot == "g1":
            self._send_command()
            payload = json.loads(
                self.request(
                    "vset /object/{}/mujoco_g1_policy_sync/start".format(
                        self.actor_name
                    )
                )
            )
            return np.asarray(payload["obs"], dtype=np.float32), payload

        self.request(
            "vset /object/{}/mujoco_microduck/start".format(self.actor_name)
        )
        state = json.loads(
            self.request(
                "vset /object/{}/mujoco_microduck_control_sync/start".format(
                    self.actor_name
                )
            )
        )
        return self._microduck_observation(state), state

    def reset(self):
        self._ensure_session()
        if not self.actor_name:
            self._spawn_actor()
        elif not self.physics_configured:
            self.configure_mujoco_physics(self.actor_name)
            self.physics_configured = True

        # MuJoCo writes its floating-base pose back to the owning UE Actor. A
        # stop/start cycle therefore cannot be a reset by itself: without an
        # explicit restore, the next model captures the previous episode's
        # fallen (and potentially underground) Actor transform as its new
        # initial qpos. Capture once, before the first simulation starts, and
        # always restore that immutable transform before rebuilding the model.
        if self._reset_location is None:
            self._capture_reset_transform()
        if self.started:
            self._stop_robot()
        self._restore_reset_transform()

        self.steps = 0
        self.last_action.fill(0.0)
        self.previous_targets = None
        observation, self.state = self._start_robot()
        self.started = True
        return observation

    def _microduck_targets(self, action):
        moving = np.linalg.norm(self.command) >= 0.05
        scale = 0.9 if moving else 1.0
        targets = MICRODUCK_DEFAULT_POSE + action * scale
        if self.previous_targets is not None:
            targets[5:9] = (
                0.5 * targets[5:9] + 0.5 * self.previous_targets[5:9]
            )
            targets[MICRODUCK_LEG_INDICES] = (
                0.7 * targets[MICRODUCK_LEG_INDICES]
                + 0.3 * self.previous_targets[MICRODUCK_LEG_INDICES]
            )
        self.previous_targets = targets.copy()
        return targets

    def _build_step_command(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self.robot == "go1":
            values = " ".join("{:.9f}".format(float(value)) for value in action)
            return "vset /object/{}/mujoco_go1_policy_step {}".format(
                self.actor_name, values
            )
        if self.robot == "g1":
            values = ",".join("{:.9f}".format(float(value)) for value in action)
            return "vset /object/{}/mujoco_g1_policy_step {}".format(
                self.actor_name, values
            )
        targets = self._microduck_targets(action)
        values = ",".join("{:.9g}".format(float(value)) for value in targets)
        return "vset /object/{}/mujoco_microduck_control_step {}".format(
            self.actor_name, values
        )

    def _consume_step_response(self, action, response):
        action = np.asarray(action, dtype=np.float32)
        self.state = json.loads(self._response_text(response))
        if self.robot in ("go1", "g1"):
            observation = np.asarray(self.state["obs"], dtype=np.float32)
        else:
            self.last_action = action.copy()
            observation = self._microduck_observation(self.state)

        self.steps += 1
        info = dict(self.state)
        info["actor"] = self.actor_name
        info["command"] = self.command.copy()
        info["steps"] = self.steps
        return observation, 0.0, False, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        response = self.request(self._build_step_command(action))
        return self._consume_step_response(action, response)

    def _stop_robot(self):
        if self.robot == "go1":
            self.request(
                "vset /object/{}/mujoco_quadruped_pose_preview/stop".format(
                    self.actor_name
                )
            )
        elif self.robot == "g1":
            self.request(
                "vset /object/{}/mujoco_g1_policy_command 0.000000 0.000000 0.000000".format(
                    self.actor_name
                )
            )
        else:
            self.request(
                "vset /object/{}/mujoco_microduck/stop".format(self.actor_name)
            )
        self.started = False

    def _close_actor(self):
        if self.actor_name:
            if self.started:
                self._stop_robot()
            if self.owns_actor and not self.keep_actor:
                self.request(
                    "vset /object/{}/destroy".format(self.actor_name)
                )
            self.actor_name = ""

    def close(self):
        self._close_actor()
        super().close()
