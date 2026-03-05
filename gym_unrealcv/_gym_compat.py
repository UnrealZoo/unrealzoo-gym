"""Compatibility shim for gym/gymnasium imports.

Prefer legacy gym for API compatibility with existing env implementations;
fall back to gymnasium when gym is unavailable.
"""

from __future__ import annotations

try:
    import gym as legacy_gym
    from gym import spaces as legacy_spaces
    from gym.envs.registration import register as legacy_register

    gym = legacy_gym
    spaces = legacy_spaces
    register = legacy_register
except ImportError:  # pragma: no cover - fallback path
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.envs.registration import register

Wrapper = gym.Wrapper
Env = gym.Env

__all__ = ["gym", "spaces", "register", "Wrapper", "Env", "spec"]


def spec(env_id: str):
    return gym.spec(env_id)
