"""Typed data structures for environment and agent configuration.

Replaces raw ``dict`` agent configs with proper dataclasses, giving IDE
auto-complete, early validation, and one clear definition of the expected
config shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------
Pose6D = Tuple[float, float, float, float, float, float]
"""A 6-DOF pose: ``(x, y, z, roll, yaw, pitch)``."""

Location3D = List[float]
"""A 3-element ``[x, y, z]`` coordinate."""


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    """Typed mirror of the per-agent dict that ``misc.convert_dict`` produces.

    Construct from a raw dict via :meth:`from_dict` (tolerant of extra keys)
    or use directly.
    """

    agent_type: str
    """Category: ``'player'``, ``'animal'``, ``'drone'``, ``'car'``, …"""

    name: str = ""
    """Agent instance name in the UE scene (e.g. ``'player_0'``)."""

    cam_id: int = -1
    """Attached camera index (``-1`` if no camera attached)."""

    class_name: str = "bp_character_C"
    """UE blueprint class name for spawning."""

    scale: Union[float, List[float]] = 1.0
    """Uniform or per-axis scale factor."""

    internal_nav: bool = False
    """Whether the agent uses UE NavMesh internally."""

    relative_location: Location3D = field(default_factory=lambda: [0, 30, 70])
    """Camera location relative to the agent."""

    relative_rotation: Location3D = field(default_factory=lambda: [0, 0, 0])
    """Camera rotation relative to the agent."""

    move_action: Optional[List[List[float]]] = None
    """Discrete move-action table (``None`` for continuous-only agents)."""

    move_action_continuous: Optional[Dict[str, List[float]]] = None
    """Continuous move-action bounds (``{'low': [...], 'high': [...]}``).  ``None`` for discrete-only agents."""

    head_action: Optional[List[List[float]]] = None
    """Discrete head/turn-action table (used in Mixed action space)."""

    animation_action: Optional[List[str]] = None
    """Animation action table (used in Mixed action space)."""

    # Catch-all for forwards-compatibility with new config keys.
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Build an ``AgentConfig`` from a plain dict (as produced by ``convert_dict``).

        Unknown keys are stored in :attr:`extra` rather than discarded, so
        existing configs with custom fields keep working.
        """
        known_keys = {f.name for f in cls.__dataclass_fields__.values() if f.name != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known_keys}
        extras = {k: v for k, v in data.items() if k not in known_keys}
        return cls(**kwargs, extra=extras)

    def to_dict(self) -> Dict[str, Any]:
        """Round-trip back to a plain dict (including extras)."""
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# Environment settings (readonly after load)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvSettings:
    """Top-level settings loaded from a JSON config file.

    Read-only (``frozen=True``) after construction to prevent accidental
    mutation during a run.
    """

    env_name: str
    height: float

    # camera
    third_cam_id: int
    height_top_view: float

    # agents & environment
    agents: Dict[str, Any]
    env: Dict[str, Any]
    reset_area: List[float]
    safe_start: List[Location3D]
    interval: float
    random_init: bool

    # binary paths (platform-dependent)
    env_bin: str = ""
    env_bin_mac: str = ""
    env_bin_win: str = ""
    env_map: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvSettings":
        """Construct from a raw JSON-loaded dict."""
        third_cam = data.get("third_cam", {})
        return cls(
            env_name=data["env_name"],
            height=data["height"],
            third_cam_id=third_cam.get("cam_id", 0),
            height_top_view=third_cam.get("height_top_view", 1000),
            agents=data["agents"],
            env=data["env"],
            reset_area=data["reset_area"],
            safe_start=data["safe_start"],
            interval=data["interval"],
            random_init=data["random_init"],
            env_bin=data.get("env_bin", ""),
            env_bin_mac=data.get("env_bin_mac", ""),
            env_bin_win=data.get("env_bin_win", ""),
            env_map=data.get("env_map"),
        )
