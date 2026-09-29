import json
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

    def test_reset_restores_original_actor_transform_before_restart(self):
        env = UnrealCvMujocoEnv(
            "go1", actor_name="Go1_Test", launch=False
        )
        commands = []

        def fake_request(command):
            commands.append(command)
            if command == "vget /object/Go1_Test/location":
                return "100.0, 200.0, 44.5"
            if command == "vget /object/Go1_Test/rotation":
                return "0.0 90.0 0.0"
            if "mujoco_physics_config" in command:
                return json.dumps(env.physics_config)
            if command.endswith("mujoco_go1_policy_sync/start"):
                return json.dumps({"obs": [0.0] * 48})
            return "ok"

        env._ensure_session = lambda: None
        env.request = fake_request

        env.reset()
        first_reset_command_count = len(commands)
        env.reset()
        second_reset_commands = commands[first_reset_command_count:]

        self.assertEqual(
            commands.count("vget /object/Go1_Test/location"), 1
        )
        self.assertEqual(
            commands.count("vget /object/Go1_Test/rotation"), 1
        )
        self.assertEqual(
            second_reset_commands[0],
            "vset /object/Go1_Test/mujoco_quadruped_pose_preview/stop",
        )
        self.assertEqual(
            second_reset_commands[1],
            "vset /object/Go1_Test/location 100.000000000 200.000000000 44.500000000",
        )
        self.assertEqual(
            second_reset_commands[2],
            "vset /object/Go1_Test/rotation 0.000000000 90.000000000 0.000000000",
        )
        start_index = second_reset_commands.index(
            "vset /object/Go1_Test/mujoco_quadruped_pose_preview/start go1"
        )
        self.assertGreater(start_index, 2)


if __name__ == "__main__":
    unittest.main()
