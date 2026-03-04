from types import SimpleNamespace

import numpy as np

from gym_unrealcv.envs.wrappers.agents import NavAgents
from gym_unrealcv.envs.wrappers.augmentation import _extract_reset_type


class _DummyWrappedEnv:
    def __init__(self) -> None:
        self.unwrapped = SimpleNamespace(obj_poses=[])
        self._last_action = None

    def step(self, action):
        self._last_action = action
        obs = np.array([10, 20])
        reward = np.array([1.0, 2.0])
        done = False
        info = {}
        return obs, reward, done, info


def test_navagents_step_does_not_mutate_input_actions() -> None:
    wrapper = NavAgents.__new__(NavAgents)
    wrapper.env = _DummyWrappedEnv()
    wrapper.mask_agent = True
    wrapper.nav_list = [-1, -1]
    wrapper.agents = [None, None]

    original_actions = [[0, 1], [2, 3]]
    actions_snapshot = [list(item) for item in original_actions]

    obs, reward, done, info = wrapper.step(original_actions)

    assert original_actions == actions_snapshot
    assert wrapper.env._last_action == actions_snapshot
    assert obs.tolist() == [10, 20]
    assert reward.tolist() == [1.0, 2.0]
    assert done is False
    assert info == {}


class _DummyEnvForResetType:
    def __init__(self, spec_id, fallback_reset_type: int) -> None:
        self.spec = SimpleNamespace(id=spec_id)
        self.unwrapped = SimpleNamespace(reset_type=fallback_reset_type)


def test_extract_reset_type_from_spec_id() -> None:
    env = _DummyEnvForResetType('UnrealTrack-track_train-ContinuousColor-v5', 0)
    assert _extract_reset_type(env) == 5


def test_extract_reset_type_falls_back_when_spec_missing_version() -> None:
    env = _DummyEnvForResetType('UnrealTrack-track_train-ContinuousColor', 3)
    assert _extract_reset_type(env) == 3
