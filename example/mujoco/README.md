# MuJoCo robot examples

These examples connect UnrealZoo rendering and scene collision to MuJoCo-driven
Go1, G1, and MicroDuck robots. Start with the unified single-robot Gym example;
the other folders contain specialized demonstrations and maintenance tools.

| Keyboard control | Parkour third-person view | Parkour depth observation |
|:---:|:---:|:---:|
| ![MuJoCo keyboard control](../../doc/figs/new_features/mujoco_go1_keyboard.gif) | ![Parkour third-person view](../../doc/figs/new_features/mujoco_go1_parkour_third_person.gif) | ![Parkour depth observation](../../doc/figs/new_features/mujoco_go1_parkour_depth.gif) |

## Directory map

| Path | Purpose |
| --- | --- |
| `mujoco_robot_demo.py` | Primary Gym-style checkpoint and I/J/K/L locomotion entry point |
| `parkour/` | Go1 depth-conditioned recurrent Parkour demo |
| `scenarios/` | Multi-robot SchoolGym and DowntownWest showcases |
| `tools/` | Policy download and bridge diagnostics |
| `common/` | Shared command and pretrained-policy helpers |
| `runtime/` | Compatibility protocol used by advanced scenarios |
| `policies/` | Checkpoints, model configuration, provenance, and licenses |
| `third_party/` | Clearly isolated upstream inference runtime |
| `docs/` | Parkour, scenarios, physics, and maintainer architecture notes |

## Setup

From the repository root:

```cmd
python -m pip install -e .
```

Start the UnrealZoo v3.1.1 binary, wait for the map to load and UnrealCV to
listen on port 9000, then run one controller at a time. PIE remains supported
for development.

## Single-robot quick start

Keyboard mode uses I/K for forward/backward, J/L for turning, Space to stop,
and X/Esc to exit:

```cmd
python .\example\mujoco\mujoco_robot_demo.py go1 keyboard --host 127.0.0.1 --port 9000
python .\example\mujoco\mujoco_robot_demo.py g1 keyboard --host 127.0.0.1 --port 9000
python .\example\mujoco\mujoco_robot_demo.py microduck keyboard --host 127.0.0.1 --port 9000
```

Checkpoint mode continuously applies a fixed velocity command:

```cmd
python .\example\mujoco\mujoco_robot_demo.py go1 checkpoint --command-vx 0.5 --duration 20
python .\example\mujoco\mujoco_robot_demo.py g1 checkpoint --command-vx 0.5 --command-vy 0.0 --duration 20
python .\example\mujoco\mujoco_robot_demo.py microduck checkpoint --command-vx 0.4 --duration 20
```

The demo spawns the selected Blueprint in front of Camera 0, configures its
MuJoCo model before compilation, and removes only the actor it owns. Pass
`--actor NAME` to reuse an actor or `--keep-actor` to retain a spawned actor.

## Gym API

```python
import gym_unrealcv

env = gym_unrealcv.make_mujoco_env("go1", host="127.0.0.1", port=9000)
observation = env.reset()

for step in range(100):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

env.close()
```

The base environment is a low-latency control bridge. It deliberately returns
zero reward and no termination so that a task can add its own reward, reset,
and episode rules. See [physics configuration](docs/PHYSICS.md) for startup
parameters.

## Specialized examples

- [Parkour](docs/PARKOUR.md): Go1 proprioception plus delayed 48x64 depth and
  recurrent policy state.
- [Multi-robot scenarios](docs/SCENARIOS.md): MicroDuck skills, shared balls,
  mixed Go1/MicroDuck control, and follow-the-leader.
- [Architecture](docs/ARCHITECTURE.md): ownership boundaries and future batch
  environment migration.

Generated Parkour observation dumps are written under the repository-level
`artifacts/` directory, not beside the example source.
