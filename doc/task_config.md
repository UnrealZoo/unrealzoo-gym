# Task config (`task_file`)

You can now pass a `task_file` JSON to task environments to override task-specific runtime parameters without editing the main environment config.

## Example usage

```python
import gym
from gym_unrealcv.envs.rescue import Rescue
from gym_unrealcv.envs.rendezvous import Rendezvous

env = gym.make(
    'UnrealTrack-SuburbNeighborhood_Day-DiscreteColor-v0',
    task_file='Track/custom_track_task.json',
)

env_nav = gym.make(
    'UnrealNavigation-SuburbNeighborhood_Day-DiscreteColor-v0',
    task_file='Navigation/custom_navigation_task.json',
)

env_rescue = Rescue(
    env_file='Track/SuburbNeighborhood_Day.json',
    task_file='Rescue/custom_rescue_task.json',
)

env_rendezvous = Rendezvous(
    env_file='Track/SuburbNeighborhood_Day.json',
    task_file='Rendezvous/custom_rendezvous_task.json',
)
```

`task_file` can be:
- a path under `gym_unrealcv/envs/setting` (for example `Track/custom_track_task.json`),
- a workspace-relative file path,
- or an absolute file path.

## Supported keys

### `Track`
- `max_lost_steps`
- `reward_type` (`dense` or `sparse`)
- `distance_threshold`
- `tracker_id`
- `target_id`
- `reward_params` (object with keys like `min_distance`, `max_direction`, `max_distance`, `exp_distance`, `exp_angle`)

### `Rescue`
- `max_reach_steps`
- `distance_threshold`
- `reward_type`
- `injured_agent_name`

### `Rendezvous`
- `max_meet_steps`
- `distance_threshold`

### `Navigation` / `NavigationMulti`
- `target_list`
- `reward_type`
- `max_collision_steps`
- `success_distance`
- `success_direction`

## Included sample files

- `gym_unrealcv/envs/setting/Track/custom_track_task.json`
- `gym_unrealcv/envs/setting/Navigation/custom_navigation_task.json`
- `gym_unrealcv/envs/setting/Rescue/custom_rescue_task.json`
- `gym_unrealcv/envs/setting/Rendezvous/custom_rendezvous_task.json`

## Smoke test script

Run the sample smoke test to validate all four task-file examples in dry-run mode (JSON loading only, no UE binary launch):

```bash
python example/task_file_smoke.py
```

## Optional runtime check

Run this only when Unreal binaries are installed locally. It attempts to create each sample env and call `reset()`:

```bash
python example/task_file_runtime_check.py
```

If binaries are missing, the script prints `SKIP` entries instead of failing immediately.
