import os
import unittest
from unittest import mock

from gym_unrealcv.envs.base_env_mujoco import (
    MUJOCO_PHYSICS_DEFAULTS,
    _merged_physics_config,
)
from gym_unrealcv.envs.mujoco import UnrealCvMujocoEnv


class MujocoPhysicsConfigTest(unittest.TestCase):

    def test_robot_defaults_keep_twenty_millisecond_control_period(self):
        for robot, defaults in MUJOCO_PHYSICS_DEFAULTS.items():
            period = defaults["timestep"] * defaults["control_decimation"]
            self.assertAlmostEqual(period, 0.02, msg=robot)

    def test_runtime_override_has_highest_precedence(self):
        setting = {
            "mujoco": {
                "physics": {"joint_damping_scale": 1.1},
                "robots": {
                    "g1": {
                        "physics": {
                            "joint_damping_scale": 1.2,
                            "solver_iterations": 15,
                        }
                    }
                },
            }
        }
        config = _merged_physics_config(
            "g1", setting, {"joint_damping_scale": 1.3}
        )
        self.assertEqual(config["joint_damping_scale"], 1.3)
        self.assertEqual(config["solver_iterations"], 15)

    def test_unknown_parameter_is_rejected(self):
        with self.assertRaises(ValueError):
            _merged_physics_config("go1", {}, {"dmaping": 2.0})

    def test_fractional_decimation_is_rejected(self):
        with self.assertRaises(ValueError):
            _merged_physics_config(
                "microduck", {}, {"control_decimation": 4.5}
            )

    def test_environment_construction_is_lazy(self):
        env = UnrealCvMujocoEnv("go1", launch=False)
        self.assertIsNone(env.client)
        self.assertAlmostEqual(env.control_period, 0.02)
        env.close()

    def test_launch_requires_unreal_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Set UnrealEnv"):
                UnrealCvMujocoEnv(
                    "go1",
                    setting_file="Mujoco/SuburbNeighborhood_Day.json",
                    launch=True,
                )


if __name__ == "__main__":
    unittest.main()
