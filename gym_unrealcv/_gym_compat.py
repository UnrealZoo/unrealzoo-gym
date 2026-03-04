"""Compatibility shim for gym/gymnasium imports.

Prefer gymnasium when available; fall back to legacy gym.
"""

from __future__ import annotations

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.envs.registration import register
except ImportError:  # pragma: no cover - fallback path
    import gym as legacy_gym
    from gym import spaces as legacy_spaces
    from gym.envs.registration import register as legacy_register

    gym = legacy_gym
    spaces = legacy_spaces
    register = legacy_register

Wrapper = gym.Wrapper
Env = gym.Env

__all__ = ["gym", "spaces", "register", "Wrapper", "Env", "spec"]


def spec(env_id: str):
    return gym.spec(env_id)
