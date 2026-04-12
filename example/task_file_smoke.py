from gym_unrealcv.envs.base_env import UnrealCv_base
from gym_unrealcv.envs.utils import misc

SAMPLES = [
    (
        'Track',
        {'env_file': 'Track/SuburbNeighborhood_Day.json', 'task_file': 'Track/custom_track_task.json'},
    ),
    (
        'Navigation',
        {'env_file': 'Navigation/SuburbNeighborhood_Day.json', 'task_file': 'Navigation/custom_navigation_task.json'},
    ),
    (
        'Rescue',
        {'env_file': 'Track/SuburbNeighborhood_Day.json', 'task_file': 'Rescue/custom_rescue_task.json'},
    ),
    (
        'Rendezvous',
        {'env_file': 'Track/SuburbNeighborhood_Day.json', 'task_file': 'Rendezvous/custom_rendezvous_task.json'},
    ),
]


def main() -> None:
    print('Checking task_file samples in dry-run mode (JSON loading only)...')
    base_stub = object.__new__(UnrealCv_base)
    for env_name, kwargs in SAMPLES:
        env_setting = misc.load_env_setting(kwargs['env_file'])
        task_setting = UnrealCv_base.load_task_setting(base_stub, kwargs['task_file'])
        assert isinstance(env_setting, dict)
        assert isinstance(task_setting, dict)
        print(f"OK: {env_name} env_file={kwargs['env_file']} task_file={kwargs['task_file']}")
    print('All task_file samples were loaded successfully.')


if __name__ == '__main__':
    main()
