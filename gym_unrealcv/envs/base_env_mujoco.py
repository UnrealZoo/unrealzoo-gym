"""Low-latency UnrealCV lifecycle shared by MuJoCo robot environments."""
import copy
import json
import math
import os
import sys

import gym
import unrealcv
from unrealcv.launcher import RunUnreal

from gym_unrealcv.envs.utils import misc


MUJOCO_PHYSICS_DEFAULTS = {
    "go1": {
        "timestep": 0.005,
        "control_decimation": 4,
    },
    "g1": {
        "timestep": 0.002,
        "control_decimation": 10,
    },
    "microduck": {
        "timestep": 0.005,
        "control_decimation": 4,
    },
}

_COMMON_PHYSICS_DEFAULTS = {
    "gravity": [0.0, 0.0, -9.81],
    "solver_iterations": 0,
    "joint_damping_scale": 1.0,
    "joint_armature_scale": 1.0,
    "joint_frictionloss_scale": 1.0,
    "geom_friction_scale": [1.0, 1.0, 1.0],
    "actuator_force_limit_scale": 1.0,
}


def _merged_physics_config(robot, setting, override):
    config = copy.deepcopy(_COMMON_PHYSICS_DEFAULTS)
    config.update(copy.deepcopy(MUJOCO_PHYSICS_DEFAULTS[robot]))

    mujoco_setting = setting.get("mujoco", {})
    config.update(copy.deepcopy(mujoco_setting.get("physics", {})))
    robot_setting = mujoco_setting.get("robots", {}).get(robot, {})
    config.update(copy.deepcopy(robot_setting.get("physics", {})))
    if override is not None:
        config.update(copy.deepcopy(override))
    return _validate_physics_config(config)


def _validate_physics_config(config):
    allowed = set(_COMMON_PHYSICS_DEFAULTS)
    allowed.update(("timestep", "control_decimation"))
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            "Unknown MuJoCo physics parameter(s): {}".format(", ".join(unknown))
        )

    result = copy.deepcopy(config)
    scalar_keys = (
        "timestep",
        "joint_damping_scale",
        "joint_armature_scale",
        "joint_frictionloss_scale",
        "actuator_force_limit_scale",
    )
    for key in scalar_keys:
        result[key] = float(result[key])
        if not math.isfinite(result[key]):
            raise ValueError("{} must be finite".format(key))

    solver_iterations = float(result["solver_iterations"])
    control_decimation = float(result["control_decimation"])
    if not solver_iterations.is_integer():
        raise ValueError("solver_iterations must be an integer")
    if not control_decimation.is_integer():
        raise ValueError("control_decimation must be an integer")
    result["solver_iterations"] = int(solver_iterations)
    result["control_decimation"] = int(control_decimation)
    if not 0.0 < result["timestep"] <= 0.1:
        raise ValueError("timestep must be in (0, 0.1]")
    if not 1 <= result["control_decimation"] <= 1000:
        raise ValueError("control_decimation must be in [1, 1000]")
    if not 0 <= result["solver_iterations"] <= 10000:
        raise ValueError("solver_iterations must be in [0, 10000]")
    for key in (
        "joint_damping_scale",
        "joint_armature_scale",
        "joint_frictionloss_scale",
    ):
        if result[key] < 0.0:
            raise ValueError("{} must be non-negative".format(key))
    if result["actuator_force_limit_scale"] <= 0.0:
        raise ValueError("actuator_force_limit_scale must be positive")

    for key in ("gravity", "geom_friction_scale"):
        values = [float(value) for value in result[key]]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("{} must contain three finite values".format(key))
        result[key] = values
    if any(value < 0.0 for value in result["geom_friction_scale"]):
        raise ValueError("geom_friction_scale values must be non-negative")
    return result


class UnrealCvMujocoBase(gym.Env):
    """Unreal process, connection, and MuJoCo initialization without camera setup."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        robot,
        setting_file=None,
        physics_config=None,
        host="127.0.0.1",
        port=9000,
        launch=None,
        resolution=(160, 160),
        sleep_time=None,
        nullrhi=None,
        request_timeout=120,
    ):
        if robot not in MUJOCO_PHYSICS_DEFAULTS:
            raise ValueError("Unsupported MuJoCo robot: {}".format(robot))

        self.robot = robot
        self.setting_file = setting_file
        self.setting = (
            misc.load_env_setting(setting_file) if setting_file else {}
        )
        self.env_name = self.setting.get("env_name", "")
        self.mujoco_setting = self.setting.get("mujoco", {})
        self.robot_setting = self.mujoco_setting.get("robots", {}).get(robot, {})
        self.physics_config = _merged_physics_config(
            robot, self.setting, physics_config
        )
        self.control_period = (
            self.physics_config["timestep"]
            * self.physics_config["control_decimation"]
        )

        runtime = self.mujoco_setting.get("runtime", {})
        self.host = host
        self.port = int(port)
        self.resolution = tuple(runtime.get("resolution", resolution))
        self.sleep_time = (
            float(runtime.get("sleep_time", 8.0))
            if sleep_time is None
            else float(sleep_time)
        )
        self.nullrhi = (
            bool(runtime.get("nullrhi", False))
            if nullrhi is None
            else bool(nullrhi)
        )
        self.offscreen_rendering = bool(runtime.get("offscreen_rendering", False))
        self.use_opengl = bool(runtime.get("use_opengl", False))
        self.gpu_id = runtime.get("gpu_id")
        self.request_timeout = request_timeout
        self.launch = bool(setting_file) if launch is None else bool(launch)

        self.client = None
        self.ue_binary = None
        self.effective_physics_config = None
        self._owns_unreal_process = False
        if self.launch:
            if not self.setting:
                raise ValueError("setting_file is required when launch=True")
            if not os.environ.get("UnrealEnv"):
                raise RuntimeError(
                    "Set UnrealEnv to the directory containing the packaged "
                    "environment before using launch=True"
                )
            self.ue_binary = RunUnreal(
                ENV_BIN=self._resolve_environment_binary(),
                ENV_MAP=self.setting.get("env_map"),
            )
            self._ensure_unrealcv_ini()

    def _resolve_environment_binary(self):
        if "linux" in sys.platform:
            key = "env_bin"
        elif "darwin" in sys.platform:
            key = "env_bin_mac"
        else:
            key = "env_bin_win"
        return self.setting[key]

    def _ensure_unrealcv_ini(self):
        ini_path = self.ue_binary.path2unrealcv
        if os.path.isfile(ini_path):
            return
        with open(ini_path, "w", encoding="utf-8") as ini_file:
            ini_file.write(
                "[UnrealCV.Core]\n"
                "Port={}\n"
                "Width={}\n"
                "Height={}\n".format(
                    self.port, self.resolution[0], self.resolution[1]
                )
            )

    def _ensure_session(self):
        if self.client is not None and self.client.isconnected():
            return

        host = self.host
        port = self.port
        if self.launch:
            host, port = self.ue_binary.start(
                resolution=self.resolution,
                opengl=self.use_opengl,
                offscreen=self.offscreen_rendering,
                nullrhi=self.nullrhi,
                gpu_id=self.gpu_id,
                sleep_time=self.sleep_time,
            )
            self._owns_unreal_process = True

        self.client = unrealcv.Client((host, port))
        self.client.connect()
        if not self.client.isconnected():
            raise RuntimeError(
                "Unable to connect to UnrealCV at {}:{}".format(host, port)
            )
        self.host = host
        self.port = int(port)

    def request(self, command):
        self._ensure_session()
        return str(
            self.client.request(command, timeout=self.request_timeout)
        ).strip()

    def configure_mujoco_physics(self, actor_name):
        """Apply validated physical parameters before the actor starts MuJoCo."""
        payload = json.dumps(
            self.physics_config, separators=(",", ":"), sort_keys=True
        )
        response = self.request(
            "vset /object/{}/mujoco_physics_config {} {}".format(
                actor_name, self.robot, payload
            )
        )
        self.effective_physics_config = json.loads(response)
        return copy.deepcopy(self.effective_physics_config)

    def get_physics_config(self):
        return copy.deepcopy(self.physics_config)

    def close(self):
        if self.client is not None and self.client.isconnected():
            self.client.disconnect()
        self.client = None
        if self._owns_unreal_process and self.ue_binary is not None:
            self.ue_binary.close()
        self._owns_unreal_process = False
