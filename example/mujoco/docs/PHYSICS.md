# MuJoCo physics configuration

`gym_unrealcv.envs.base_env_mujoco.UnrealCvMujocoBase` owns the packaged
Unreal lifecycle, UnrealCV connection, and pre-start MuJoCo configuration. It
does not initialize image observations, annotations, or navigation targets.

Pass a setting file to launch the configured v3.1.1 binary automatically:

```python
import gym_unrealcv

env = gym_unrealcv.make_mujoco_env(
    "g1",
    setting_file="Mujoco/SuburbNeighborhood_Day.json",
    physics_config={
        "timestep": 0.002,
        "control_decimation": 10,
        "gravity": [0.0, 0.0, -9.81],
        "solver_iterations": 20,
        "joint_damping_scale": 1.1,
        "joint_armature_scale": 1.0,
        "joint_frictionloss_scale": 1.0,
        "geom_friction_scale": [1.2, 1.0, 1.0],
        "actuator_force_limit_scale": 0.9,
    },
)
```

The setting resolves its relative executable from the required `UnrealEnv`
directory. Set it to the directory containing the packaged environment folder
before constructing an environment with `launch=True`:

```cmd
set UnrealEnv=I:\UnrealProject\UnrealZoo_UE5_7\Binaries
```

Configuration is applied after actor creation and before its MuJoCo model is
compiled. Reconfiguration therefore requires stopping and recreating that
simulation. Runtime overrides take precedence over the common setting and the
robot-specific setting.

The control period is `timestep * control_decimation`. Defaults for Go1, G1,
and MicroDuck are all 20 ms. Changing only one value changes the policy control
frequency.
