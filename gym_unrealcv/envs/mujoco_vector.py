"""Vectorized MuJoCo robots across Unreal processes and in-process agents."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from gym import spaces

from gym_unrealcv.envs.mujoco import ROBOT_SPECS, UnrealCvMujocoEnv


class _MujocoAgentGroup:
    """Agents that share one Unreal process, UnrealCV client, and batch request."""

    def __init__(
        self,
        robot,
        env_index,
        num_agent,
        agent_spacing_cm,
        env_factory,
        command,
        **env_options
    ):
        self.robot = robot
        self.env_index = int(env_index)
        self.num_agent = int(num_agent)
        self.agent_spacing_cm = float(agent_spacing_cm)
        self.env_factory = env_factory
        self.command = np.asarray(command, dtype=np.float32)
        self.env_options = dict(env_options)
        self.agents = []
        self.initialized = False

        owner_options = self._agent_options(
            0, launch=env_options.get("launch")
        )
        self.owner = self.env_factory(robot, **owner_options)
        self.agents.append(self.owner)

    def _agent_offset_cm(self, agent_index):
        center = 0.5 * (self.num_agent - 1)
        return (float(agent_index) - center) * self.agent_spacing_cm

    def _agent_options(self, agent_index, launch):
        options = dict(self.env_options)
        options["launch"] = launch if launch is None else bool(launch)
        options["actor_name"] = (
            options.get("actor_name", "") if agent_index == 0 else ""
        )
        offset_cm = self._agent_offset_cm(agent_index)
        spawn_location = options.get("spawn_location")
        if spawn_location is None:
            options["spawn_camera_right_cm"] = (
                float(options.get("spawn_camera_right_cm", 0.0)) + offset_cm
            )
        else:
            location = np.asarray(spawn_location, dtype=np.float32).copy()
            location[1] += offset_cm
            options["spawn_location"] = tuple(float(value) for value in location)
        return options

    @property
    def actor_names(self):
        return [agent.actor_name for agent in self.agents]

    @property
    def port(self):
        return self.owner.port

    def set_command(self, command):
        self.command = np.asarray(command, dtype=np.float32)
        for agent in self.agents:
            agent.set_command(self.command)

    def initialize(self):
        if self.initialized:
            return self.reset()

        self.owner.set_command(self.command)
        observations = [self.owner.reset()]
        for agent_index in range(1, self.num_agent):
            options = self._agent_options(agent_index, launch=False)
            options["host"] = self.owner.host
            options["port"] = self.owner.port
            agent = self.env_factory(self.robot, **options)
            # A group deliberately uses one client connection. UnrealCV handles
            # the group's control commands as one ordered batch per vector step.
            agent.client = self.owner.client
            agent.set_command(self.command)
            observations.append(agent.reset())
            self.agents.append(agent)
        self.initialized = True
        return np.stack(observations)

    def reset(self, agent_indices=None):
        if not self.initialized:
            if agent_indices is not None:
                raise RuntimeError("Cannot partially reset an uninitialized group")
            return self.initialize()
        indices = (
            range(self.num_agent) if agent_indices is None else agent_indices
        )
        return np.stack([self.agents[index].reset() for index in indices])

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        commands = [
            agent._build_step_command(action)
            for agent, action in zip(self.agents, actions)
        ]
        responses = self.owner.request_batch(commands)
        transitions = [
            agent._consume_step_response(action, response)
            for agent, action, response in zip(self.agents, actions, responses)
        ]
        observations, rewards, dones, infos = zip(*transitions)
        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            list(infos),
        )

    def close(self):
        # Child agents share the owner's client and must release only their
        # actors. The owner disconnects and terminates the Unreal process last.
        for agent in reversed(self.agents[1:]):
            agent._close_actor()
            agent.client = None
        self.owner.close()
        self.initialized = False


class UnrealCvMujocoVectorEnv:
    """``num_env`` Unreal processes with ``num_agent`` robots in each process."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        robot,
        num_env=1,
        num_agent=1,
        agent_spacing_cm=400.0,
        env_factory=UnrealCvMujocoEnv,
        **env_options
    ):
        if robot not in ROBOT_SPECS:
            raise ValueError("Unsupported MuJoCo robot: {}".format(robot))
        if int(num_env) != num_env or int(num_env) <= 0:
            raise ValueError("num_env must be a positive integer")
        if int(num_agent) != num_agent or int(num_agent) <= 0:
            raise ValueError("num_agent must be a positive integer")
        if float(agent_spacing_cm) <= 0.0:
            raise ValueError("agent_spacing_cm must be positive")
        if env_options.get("actor_name") and (
            int(num_env) != 1 or int(num_agent) != 1
        ):
            raise ValueError("actor_name is only valid for num_env=1, num_agent=1")

        self.robot = robot
        self.num_env = int(num_env)
        self.num_agent = int(num_agent)
        self.num_instances = self.num_env * self.num_agent
        self.agent_spacing_cm = float(agent_spacing_cm)
        self.command = np.zeros(3, dtype=np.float32)
        self.closed = False
        self.observations = None

        base_port = int(env_options.get("port", 9000))
        self.groups = []
        for env_index in range(self.num_env):
            group_options = dict(env_options)
            group_options["port"] = base_port + env_index
            self.groups.append(
                _MujocoAgentGroup(
                    robot=robot,
                    env_index=env_index,
                    num_agent=self.num_agent,
                    agent_spacing_cm=self.agent_spacing_cm,
                    env_factory=env_factory,
                    command=self.command,
                    **group_options
                )
            )

        first = self.groups[0].owner
        self.control_period = first.control_period
        self.single_action_space = first.action_space
        self.single_observation_space = first.observation_space
        action_dim = self.single_action_space.shape[0]
        observation_dim = self.single_observation_space.shape[0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_instances, action_dim),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_instances, observation_dim),
            dtype=np.float32,
        )
        self._executor = (
            ThreadPoolExecutor(max_workers=self.num_env)
            if self.num_env > 1
            else None
        )

    @property
    def actor_names(self):
        return [name for group in self.groups for name in group.actor_names]

    @property
    def ports(self):
        return [group.port for group in self.groups]

    @property
    def unwrapped(self):
        return self

    def set_command(self, command):
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,):
            raise ValueError("command must have shape (3,)")
        self.command = command.copy()
        for group in self.groups:
            group.set_command(self.command)

    def _run_groups(self, method_name, arguments):
        if self._executor is None:
            return [getattr(self.groups[0], method_name)(*arguments[0])]
        futures = [
            self._executor.submit(getattr(group, method_name), *args)
            for group, args in zip(self.groups, arguments)
        ]
        return [future.result() for future in futures]

    def reset(self, indices=None):
        if indices is None:
            # Initial launches are intentionally serial because RunUnreal uses a
            # shared UnrealCV.ini while assigning each process its port.
            if not all(group.initialized for group in self.groups):
                chunks = [group.initialize() for group in self.groups]
            else:
                chunks = self._run_groups(
                    "reset", [(None,) for _ in self.groups]
                )
            self.observations = np.concatenate(chunks, axis=0)
            return self.observations.copy()

        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if self.observations is None:
            raise RuntimeError("Vector environment must be fully reset first")
        grouped = [[] for _ in self.groups]
        for global_index in indices:
            if global_index < 0 or global_index >= self.num_instances:
                raise IndexError("agent index out of range: {}".format(global_index))
            env_index, agent_index = divmod(int(global_index), self.num_agent)
            grouped[env_index].append(agent_index)
        for env_index, agent_indices in enumerate(grouped):
            if not agent_indices:
                continue
            reset_observations = self.groups[env_index].reset(agent_indices)
            for agent_index, observation in zip(agent_indices, reset_observations):
                global_index = env_index * self.num_agent + agent_index
                self.observations[global_index] = observation
        return self.observations[indices].copy()

    def step(self, actions):
        if self.observations is None:
            raise RuntimeError("Vector environment must be reset before step")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != self.action_space.shape:
            raise ValueError(
                "actions must have shape {}, got {}".format(
                    self.action_space.shape, actions.shape
                )
            )
        chunks = [
            (actions[index * self.num_agent:(index + 1) * self.num_agent],)
            for index in range(self.num_env)
        ]
        results = self._run_groups("step", chunks)
        observations = np.concatenate([result[0] for result in results], axis=0)
        rewards = np.concatenate([result[1] for result in results], axis=0)
        dones = np.concatenate([result[2] for result in results], axis=0)
        infos = [info for result in results for info in result[3]]
        self.observations = observations
        return observations.copy(), rewards, dones, infos

    def get_physics_config(self):
        return self.groups[0].owner.get_physics_config()

    def close(self):
        if self.closed:
            return
        for group in self.groups:
            group.close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self.closed = True
