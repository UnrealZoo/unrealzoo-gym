"""Tests for lazy environment registration in gym_unrealcv.__init__."""
import importlib

import pytest

from gym_unrealcv._gym_compat import gym


def _entry_point(spec: object) -> object:
    return getattr(spec, 'entry_point', getattr(spec, '_entry_point', None))


def _kwargs(spec: object) -> dict:
    return getattr(spec, 'kwargs', getattr(spec, '_kwargs', {}))


def _all_registered_ids(gym_module: object) -> list:
    registry = gym_module.envs.registry
    if hasattr(registry, 'keys'):
        return list(registry.keys())
    if hasattr(registry, 'env_specs'):
        env_specs = registry.env_specs
        if hasattr(env_specs, 'keys'):
            return list(env_specs.keys())
        return list(env_specs)
    return []


def test_import_registers_zero_envs():
    """Importing gym_unrealcv should NOT eagerly register Unreal env specs."""
    before = {s for s in _all_registered_ids(gym) if 'Unreal' in s}
    importlib.import_module('gym_unrealcv')
    after = {s for s in _all_registered_ids(gym) if 'Unreal' in s}
    assert after == before, (
        f"Import should not eagerly register envs. Before={len(before)}, after={len(after)}"
    )


def test_lazy_make_registers_on_demand(monkeypatch: pytest.MonkeyPatch):
    """gym.make() should trigger lazy registration when an ID is missing."""
    module = importlib.import_module('gym_unrealcv')
    env_id = 'UnrealUnitTest-CartPole-v0'
    if gym.__name__ == 'gym':
        from gym.envs.classic_control import CartPoleEnv as entry_point
    else:
        from gymnasium.envs.classic_control.cartpole import CartPoleEnv as entry_point

    def _fake_parse_and_register(candidate_id: str) -> bool:
        if candidate_id != env_id:
            return False
        module.register(id=env_id, entry_point=entry_point, max_episode_steps=5)
        return True

    monkeypatch.setattr(module, '_parse_and_register', _fake_parse_and_register)

    gym.spec(env_id)
    env = gym.make(env_id)
    try:
        env.reset()
        assert env.spec is not None
        assert env.spec.id == env_id
    finally:
        env.close()


def test_lazy_lookup_agent_env():
    """Lazily looking up an UnrealAgent env should succeed and register it."""
    importlib.import_module('gym_unrealcv')

    env_id = 'UnrealAgent-Greek_Island-DiscreteColor-v0'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert _entry_point(spec) == 'gym_unrealcv.envs:UnrealCv_base'
    assert _kwargs(spec)['action_type'] == 'Discrete'
    assert _kwargs(spec)['observation_type'] == 'Color'
    assert _kwargs(spec)['reset_type'] == 0


def test_lazy_lookup_task_env():
    """Task-oriented env (Navigation) should register lazily."""
    importlib.import_module('gym_unrealcv')

    env_id = 'UnrealNavigation-Demo_Roof-DiscreteColor-v0'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert _entry_point(spec) == 'gym_unrealcv.envs:Navigation'
    assert spec.max_episode_steps == 1000


def test_lazy_lookup_task_track():
    """Track task should lazily register with max_episode_steps=500."""
    importlib.import_module('gym_unrealcv')

    env_id = 'UnrealTrack-FlexibleRoom-ContinuousDepth-v3'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert _entry_point(spec) == 'gym_unrealcv.envs:Track'
    assert spec.max_episode_steps == 500


def test_lazy_lookup_arm():
    """Legacy robot arm env should still work."""
    importlib.import_module('gym_unrealcv')

    env_id = 'UnrealArm-ContinuousRgbd-v2'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert _kwargs(spec)['version'] == 2


def test_lazy_lookup_spline_tracking():
    """Legacy spline tracking env should register."""
    importlib.import_module('gym_unrealcv')

    env_id = 'UnrealTrack-City2StefaniPath1-DiscreteColor-v0'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert _kwargs(spec)['reset_type'] == 'Static'


def test_lazy_lookup_mc_tracking():
    """Multi-camera tracking env should register."""
    importlib.import_module('gym_unrealcv')

    env_id = 'UnrealMCRoom-DiscreteColorRandom-v3'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert 'UnrealCvMC' in _entry_point(spec)


def test_lazy_lookup_mcmt():
    """MCMT tracking env should register."""
    importlib.import_module('gym_unrealcv')

    env_id = 'UnrealMCFlexibleRoom-ContinuousDepthGoal-v2'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert 'UnrealCvMultiCam' in _entry_point(spec)


def test_invalid_env_raises():
    """An unrecognised env ID should still raise an error."""
    importlib.import_module('gym_unrealcv')

    with pytest.raises(Exception):
        gym.spec('UnrealAgent-NonExistentMap-DiscreteColor-v0')


def test_invalid_obs_raises():
    """An invalid observation type should not match the pattern."""
    importlib.import_module('gym_unrealcv')

    with pytest.raises(Exception):
        gym.spec('UnrealAgent-Greek_Island-DiscreteInvalid-v0')
