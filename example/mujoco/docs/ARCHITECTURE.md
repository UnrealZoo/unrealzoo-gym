# MuJoCo example architecture

The source is organized by user intent rather than robot name:

```text
example/mujoco/
  mujoco_robot_demo.py       public single-robot Gym entry
  parkour/                   depth-conditioned Go1 example
  scenarios/                 multi-robot scene demonstrations
  tools/                     maintenance and diagnostics
  common/                    policy and input helpers
  runtime/                   legacy advanced-scenario protocol
  policies/                  models and provenance only
  third_party/               isolated upstream runtime
```

Reusable environment code belongs in the installed package:

```text
gym_unrealcv/envs/
  base_env_mujoco.py         process, connection, and physics configuration
  mujoco.py                  single-robot Gym lifecycle and protocol
```

`base_env.py` remains the image/navigation base and is intentionally separate
from `base_env_mujoco.py`.

The UE plugin separates three responsibilities: the runtime environment owns
terrain/collision queries, the shared coordinator owns synchronized batch
stepping and shared props, and each robot adapter owns its joints, actuator
mapping, observation layout, and visual-component mapping.

## Remaining migration

1. Add `UnrealCvMujocoBatchEnv` and move scenario protocol from `runtime/` into
   the installed package.
2. Convert scenario loops to `reset/step/close`.
3. Remove the standalone legacy protocol only after
   packaged tests cover the replacement paths.
