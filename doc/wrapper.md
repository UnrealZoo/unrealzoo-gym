We also provide several wrappers that you can use to modify the environment in various ways.

Before using the wrappers, you need to create an environment by `gym.make`, as the following code shows:
```python
import gym
import gym_unrealcv
env = gym.make('UnrealTrack-track_train-ContinuousColor-v0')
```

### ConfigUE

The **ConfigUEWrapper** is used to configure the launching settings of Unreal Engine Binaries. It can be used to set various parameters of the Unreal Engine environment such as `offscreen rendering`, `resolution`, `communication protocol`, `gpu_id`, `docker usage`, etc.

```python
from gym_unrealcv.envs.wrappers import configUE
configUE.ConfigUEWrapper(env, docker=False, resolution=(160, 160), display=None,
                         offscreen=False, use_opengl=False, nullrhi=False, 
                         gpu_id=None, sleep_time=5, comm_mode='tcp')
```

### TimeDilation

The **TimeDilationWrapper** is used to control the speed of the simulation in the Unreal Engine environment. This can be useful for simulating various FPS situations for different tasks.

```python
from gym_unrealcv.envs.wrappers import time_dilation
env = time_dilation.TimeDilationWrapper(env, reference_fps=30, update_steps=60) 
# reference_fps: the target FPS to simulate, update_steps: the number of steps between each update
```

### EarlyDone

The **EarlyDoneWrapper** is used to end the episode early based on certain conditions. Users could modify the source code for different tasks. This can be useful for preventing episodes from running for too long.
    
```python
from gym_unrealcv.envs.wrappers import early_done
env = early_done.EarlyDoneWrapper(env, max_lost_steps=100)
```
    

### RandomPopulation

The **RandomPopulationWrapper** randomly populates the environment with a specified number of agents:
```python
from gym_unrealcv.envs.wrappers import augmentation
env = augmentation.RandomPopulationWrapper(env, num_min=5, num_max=10, random_target=False)   
```

### Navigation Agent
```python
from gym_unrealcv.envs.wrappers import agents
env = agents.NavAgents(env, mask_agent=True) 
# mask_agent=True: the observation and action will not be exposed to the user.
```