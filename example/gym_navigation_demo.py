#!/usr/bin/env python3
"""Shared Navigation-environment setup used by the v4 feature demos."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
from typing import Callable


DEFAULT_ENV_ID = "UnrealNavigation-SuburbNeighborhood_Day-MixedColor-v0"


def add_navigation_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_env_id: str = DEFAULT_ENV_ID,
    default_width: int = 640,
    default_height: int = 360,
) -> None:
    """Add the launch arguments shared by the Navigation-style demos."""
    group = parser.add_argument_group("Gym UnrealZoo environment")
    group.add_argument("-e", "--env-id", default=default_env_id, help="Registered gym_unrealcv environment ID")
    group.add_argument(
        "--binary",
        type=Path,
        help="Optional packaged executable override; by default the registered environment resolves env_bin under UnrealEnv",
    )
    group.add_argument("--map", help="Map argument passed to RunUnreal; defaults to the environment map")
    group.add_argument("--port", type=int, default=9000, help="Initial UnrealCV port")
    group.add_argument("--width", type=int, default=default_width)
    group.add_argument("--height", type=int, default=default_height)
    group.add_argument("--offscreen", action="store_true")
    group.add_argument("--render-quality", type=int, default=5)
    group.add_argument("--sleep-time", type=float, default=20.0, help="Wait before the first UnrealCV request")
    group.add_argument("--seed", type=int, default=10)
    group.add_argument("--comm-mode", choices=("tcp", "socket"), default="tcp")


def validate_navigation_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    # unrealcv.launcher currently parses four decimal digits from unrealcv.ini.
    if not 1024 <= args.port <= 9999:
        parser.error("--port must be in the range 1024..9999")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.sleep_time < 0:
        parser.error("--sleep-time must be non-negative")


def _resolve_binary(binary: Path | None) -> Path | None:
    if binary is None:
        return None

    value = binary.expanduser()
    if value.is_file():
        return value.resolve()
    if not value.is_dir():
        raise FileNotFoundError(f"UnrealZoo package does not exist: {value}")
    candidates = (
        value / "Windows" / "UnrealZoo_UE5_7" / "Binaries" / "Win64" / "UnrealZoo_UE5_7.exe",
        value / "UnrealZoo_UE5_7" / "Binaries" / "Win64" / "UnrealZoo_UE5_7.exe",
        value / "Binaries" / "Win64" / "UnrealZoo_UE5_7.exe",
        value / "UnrealZoo_UE5_7.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = [item for item in value.rglob("UnrealZoo_UE5_7.exe") if "Engine" not in item.parts]
    if not matches:
        raise FileNotFoundError(f"No UnrealZoo_UE5_7.exe found below {value}")
    return max(matches, key=lambda item: item.stat().st_size).resolve()


def _configure_unrealcv_ini(executable: Path, port: int, resolution: tuple[int, int]) -> None:
    """Create the four-line configuration expected by unrealcv.launcher.RunUnreal."""
    ini_path = executable.parent / "unrealcv.ini"
    extras: list[str] = []
    if ini_path.exists():
        lines = ini_path.read_text(encoding="utf-8", errors="replace").splitlines()
        extras = [
            line
            for line in lines
            if line
            and not line.startswith("[")
            and not line.casefold().startswith(("port=", "width=", "height="))
        ]
    contents = [
        "[UnrealCV.Core]",
        f"Port={port}",
        f"Width={resolution[0]}",
        f"Height={resolution[1]}",
        *extras,
    ]
    ini_path.write_text("\n".join(contents) + "\n", encoding="utf-8")


def make_navigation_env(
    args: argparse.Namespace,
    *,
    agent_category: str = "player",
    population: int = 1,
    configure_task: Callable[[object], None] | None = None,
):
    """Register, construct and wrap a Navigation environment without resetting it."""
    import gym
    import gym_unrealcv  # noqa: F401 - importing registers UnrealZoo environments
    from gym_unrealcv.envs.wrappers import augmentation, configUE
    from unrealcv.launcher import RunUnreal

    class NavigationRunUnreal(RunUnreal):
        """RunUnreal with a bind-only Windows port availability check."""

        def isPortFree(self, ip, port):  # noqa: N802 - keep upstream API name
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind((ip, port))
                except OSError:
                    return False
            return True

    executable = _resolve_binary(args.binary)
    override_name = "GYM_UNREALCV_BINARY_OVERRIDE"
    previous_override = os.environ.get(override_name)
    if executable is not None:
        os.environ[override_name] = str(executable)
    try:
        env = gym.make(args.env_id)
    finally:
        if previous_override is None:
            os.environ.pop(override_name, None)
        else:
            os.environ[override_name] = previous_override

    if executable is not None:
        _configure_unrealcv_ini(executable, args.port, (args.width, args.height))
        env.unwrapped.ue_binary = NavigationRunUnreal(
            ENV_BIN=str(executable),
            ENV_MAP=args.map or env.unwrapped.env_name,
        )
        print(f"Gym binary: {executable}")
    elif args.map:
        env.unwrapped.ue_binary.env_map = args.map

    env = configUE.ConfigUEWrapper(
        env,
        offscreen=args.offscreen,
        resolution=(args.width, args.height),
        render_quality=args.render_quality,
        sleep_time=args.sleep_time,
        comm_mode=args.comm_mode,
    )
    env.unwrapped.agents_category = [agent_category]
    if hasattr(env.unwrapped, "set_navigation_targets"):
        env.unwrapped.set_navigation_targets([])
    if configure_task is not None:
        configure_task(env.unwrapped)
    env = augmentation.RandomPopulationWrapper(
        env,
        population,
        population,
        random_target=False,
    )
    env.seed(args.seed)
    return env


def navigation_client(env):
    """Return the UnrealCV client owned by a reset Navigation environment."""
    return env.unwrapped.unrealcv.client


def navigation_camera_id(env, index: int = 0) -> int:
    camera_id = int(env.unwrapped.cam_list[index])
    if camera_id < 0:
        raise RuntimeError(f"Gym agent {env.unwrapped.player_list[index]!r} has no camera")
    return camera_id


def navigation_asset_path(env, role: str, default: str | None = None) -> str:
    """Resolve a spawn path generated into env.spawn_assets/agent templates."""
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    indexed = base_env.env_configs.get("spawn_assets", {}).get(role)
    if isinstance(indexed, str) and indexed.strip():
        return indexed.strip()
    if isinstance(indexed, list) and indexed and str(indexed[0]).strip():
        return str(indexed[0]).strip()
    template = base_env.agent_templates.get(role, {})
    paths = template.get("asset_path")
    if isinstance(paths, str) and paths.strip():
        return paths.strip()
    if isinstance(paths, list) and paths and str(paths[0]).strip():
        return str(paths[0]).strip()
    if default is not None:
        return default
    raise KeyError(f"No spawn asset path is configured for role {role!r}")
