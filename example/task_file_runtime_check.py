import os
import sys
import json
from typing import Any, Callable, Protocol, cast

import gym_unrealcv
from gym_unrealcv.envs.navigation import Navigation
from gym_unrealcv.envs.rendezvous import Rendezvous
from gym_unrealcv.envs.rescue import Rescue
from gym_unrealcv.envs.track import Track


class RuntimeEnv(Protocol):
    def reset(self) -> Any:
        ...

    def close(self) -> Any:
        ...


EnvFactory = Callable[..., RuntimeEnv]


def load_env_setting(filename: str) -> dict[str, Any]:
    if os.path.isabs(filename) and os.path.exists(filename):
        setting_path = filename
    elif os.path.exists(filename):
        setting_path = filename
    else:
        gympath = os.path.dirname(gym_unrealcv.__file__)
        setting_path = os.path.join(gympath, 'envs', 'setting', filename)
    with open(setting_path) as file:
        return cast(dict[str, Any], json.load(file))

SAMPLES: list[tuple[str, EnvFactory, dict[str, str]]] = [
    (
        'Track',
        Track,
        {'env_file': 'Track/SuburbNeighborhood_Day.json', 'task_file': 'Track/custom_track_task.json'},
    ),
    (
        'Navigation',
        Navigation,
        {'env_file': 'Navigation/SuburbNeighborhood_Day.json', 'task_file': 'Navigation/custom_navigation_task.json'},
    ),
    (
        'Rescue',
        Rescue,
        {'env_file': 'Track/SuburbNeighborhood_Day.json', 'task_file': 'Rescue/custom_rescue_task.json'},
    ),
    (
        'Rendezvous',
        Rendezvous,
        {'env_file': 'Track/SuburbNeighborhood_Day.json', 'task_file': 'Rendezvous/custom_rendezvous_task.json'},
    ),
]


def _resolve_binary_path(setting: dict[str, Any]) -> str:
    if sys.platform.startswith('linux'):
        env_bin = cast(str, setting['env_bin'])
    elif sys.platform == 'darwin':
        env_bin = cast(str, setting['env_bin_mac'])
    elif sys.platform.startswith('win'):
        env_bin = cast(str, setting['env_bin_win'])
    else:
        raise RuntimeError(f'Unsupported platform: {sys.platform}')

    unreal_env_root = os.environ.get('UnrealEnv')
    if unreal_env_root is None:
        unreal_env_root = os.path.join(os.path.expanduser('~'), '.unrealcv', 'UnrealEnv')
    return os.path.join(unreal_env_root, env_bin)


def main():
    print('Running optional runtime check for task_file samples...')
    print('This test needs Unreal binaries installed under $UnrealEnv (or ~/.unrealcv/UnrealEnv).')

    all_checked = True
    any_ran = False

    for env_name, env_class, kwargs in SAMPLES:
        setting = load_env_setting(kwargs['env_file'])
        binary_path = _resolve_binary_path(setting)

        if not os.path.exists(binary_path):
            all_checked = False
            print(f'SKIP: {env_name} (binary not found: {binary_path})')
            continue

        any_ran = True
        env = None
        try:
            env = env_class(**kwargs)
            env.reset()
            print(f'OK: {env_name} reset succeeded with task_file={kwargs["task_file"]}')
        finally:
            if env is not None:
                env.close()

    if any_ran:
        print('Runtime check finished.')
    else:
        print('No runtime checks executed because binaries are missing.')

    if not all_checked:
        print('Some checks were skipped due to missing binaries.')


if __name__ == '__main__':
    main()
