"""Tests for lazy environment registration in gym_unrealcv.__init__."""
import pytest


def test_import_registers_zero_envs():
    """Importing gym_unrealcv should NOT eagerly register any Unreal envs."""
    import gym
    import gym_unrealcv  # noqa: F401

    unreal_specs = [s for s in gym.envs.registry.env_specs if 'Unreal' in s]
    assert len(unreal_specs) == 0, (
        f"Expected 0 eager registrations, got {len(unreal_specs)}"
    )


def test_lazy_lookup_agent_env():
    """Lazily looking up an UnrealAgent env should succeed and register it."""
    import gym
    import gym_unrealcv  # noqa: F401

    env_id = 'UnrealAgent-Greek_Island-DiscreteColor-v0'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert spec._entry_point == 'gym_unrealcv.envs:UnrealCv_base'
    assert spec._kwargs['action_type'] == 'Discrete'
    assert spec._kwargs['observation_type'] == 'Color'
    assert spec._kwargs['reset_type'] == 0


def test_lazy_lookup_task_env():
    """Task-oriented env (Navigation) should register lazily."""
    import gym
    import gym_unrealcv  # noqa: F401

    env_id = 'UnrealNavigation-Demo_Roof-DiscreteColor-v0'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert spec._entry_point == 'gym_unrealcv.envs:Navigation'
    assert spec.max_episode_steps == 1000


def test_lazy_lookup_task_track():
    """Track task should lazily register with max_episode_steps=500."""
    import gym
    import gym_unrealcv  # noqa: F401

    env_id = 'UnrealTrack-FlexibleRoom-ContinuousDepth-v3'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert spec._entry_point == 'gym_unrealcv.envs:Track'
    assert spec.max_episode_steps == 500


def test_lazy_lookup_arm():
    """Legacy robot arm env should still work."""
    import gym
    import gym_unrealcv  # noqa: F401

    env_id = 'UnrealArm-ContinuousRgbd-v2'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert spec._kwargs['version'] == 2


def test_lazy_lookup_spline_tracking():
    """Legacy spline tracking env should register."""
    import gym
    import gym_unrealcv  # noqa: F401

    env_id = 'UnrealTrack-City2StefaniPath1-DiscreteColor-v0'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert spec._kwargs['reset_type'] == 'Static'


def test_lazy_lookup_mc_tracking():
    """Multi-camera tracking env should register."""
    import gym
    import gym_unrealcv  # noqa: F401

    env_id = 'UnrealMCRoom-DiscreteColorRandom-v3'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert 'UnrealCvMC' in spec._entry_point


def test_lazy_lookup_mcmt():
    """MCMT tracking env should register."""
    import gym
    import gym_unrealcv  # noqa: F401

    env_id = 'UnrealMCFlexibleRoom-ContinuousDepthGoal-v2'
    spec = gym.spec(env_id)
    assert spec.id == env_id
    assert 'UnrealCvMultiCam' in spec._entry_point


def test_invalid_env_raises():
    """An unrecognised env ID should still raise an error."""
    import gym
    import gym_unrealcv  # noqa: F401

    with pytest.raises(Exception):
        gym.spec('UnrealAgent-NonExistentMap-DiscreteColor-v0')


def test_invalid_obs_raises():
    """An invalid observation type should not match the pattern."""
    import gym
    import gym_unrealcv  # noqa: F401

    with pytest.raises(Exception):
        gym.spec('UnrealAgent-Greek_Island-DiscreteInvalid-v0')
