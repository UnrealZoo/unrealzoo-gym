import unittest

import numpy as np
from gym import spaces

from gym_unrealcv.envs.mujoco_vector import UnrealCvMujocoVectorEnv


class FakeMujocoEnv:
    next_actor = 0

    def __init__(
        self,
        robot,
        host="127.0.0.1",
        port=9000,
        actor_name="",
        launch=None,
        spawn_location=None,
        spawn_camera_right_cm=0.0,
        **kwargs
    ):
        self.robot = robot
        self.host = host
        self.port = int(port)
        self.actor_name = actor_name
        self.launch = launch
        self.spawn_location = spawn_location
        self.spawn_camera_right_cm = spawn_camera_right_cm
        self.client = object()
        self.control_period = 0.02
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(4,), dtype=np.float32
        )
        self.reset_count = 0
        self.batch_sizes = []
        self.command = np.zeros(3, dtype=np.float32)
        self.closed = False

    def set_command(self, command):
        self.command = np.asarray(command, dtype=np.float32)

    def reset(self):
        if not self.actor_name:
            self.actor_name = "FakeAgent{}".format(FakeMujocoEnv.next_actor)
            FakeMujocoEnv.next_actor += 1
        self.reset_count += 1
        return np.full(4, self.reset_count, dtype=np.float32)

    def _build_step_command(self, action):
        return np.asarray(action, dtype=np.float32).copy()

    def request_batch(self, commands):
        self.batch_sizes.append(len(commands))
        return commands

    def _consume_step_response(self, action, response):
        observation = np.asarray(
            [response[0], response[1], self.port, self.reset_count],
            dtype=np.float32,
        )
        return observation, 0.0, False, {"actor": self.actor_name}

    def get_physics_config(self):
        return {"timestep": 0.005, "control_decimation": 4}

    def _close_actor(self):
        self.actor_name = ""

    def close(self):
        self._close_actor()
        self.closed = True


class MujocoVectorEnvTest(unittest.TestCase):

    def setUp(self):
        FakeMujocoEnv.next_actor = 0

    def test_two_environments_with_three_agents_each(self):
        env = UnrealCvMujocoVectorEnv(
            "go1",
            num_env=2,
            num_agent=3,
            agent_spacing_cm=250.0,
            env_factory=FakeMujocoEnv,
            host="127.0.0.1",
            port=9100,
            launch=True,
        )
        observations = env.reset()
        self.assertEqual(observations.shape, (6, 4))
        self.assertEqual(env.ports, [9100, 9101])
        self.assertEqual(len(env.actor_names), 6)
        for group in env.groups:
            self.assertEqual(len(group.agents), 3)
            self.assertTrue(
                all(agent.client is group.owner.client for agent in group.agents)
            )
            self.assertEqual(
                [agent.spawn_camera_right_cm for agent in group.agents],
                [-250.0, 0.0, 250.0],
            )

        actions = np.arange(12, dtype=np.float32).reshape(6, 2) / 12.0
        next_observations, rewards, dones, infos = env.step(actions)
        self.assertEqual(next_observations.shape, (6, 4))
        self.assertEqual(rewards.shape, (6,))
        self.assertEqual(dones.shape, (6,))
        self.assertEqual(len(infos), 6)
        self.assertEqual(env.groups[0].owner.batch_sizes, [3])
        self.assertEqual(env.groups[1].owner.batch_sizes, [3])

        env.reset([1, 4])
        self.assertEqual(env.groups[0].agents[1].reset_count, 2)
        self.assertEqual(env.groups[1].agents[1].reset_count, 2)
        env.close()
        self.assertTrue(all(group.owner.closed for group in env.groups))

    def test_named_actor_rejects_multi_instance_topology(self):
        with self.assertRaisesRegex(ValueError, "actor_name"):
            UnrealCvMujocoVectorEnv(
                "go1",
                num_env=1,
                num_agent=2,
                actor_name="PlacedGo1",
                env_factory=FakeMujocoEnv,
            )


if __name__ == "__main__":
    unittest.main()
