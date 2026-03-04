## Add a new unreal environment

In this section, we will show you how to add a new environment in gym-unrealzoo for your interactive tasks, step by step.
1. Download/Package a UE binary integrated with UnrealCV Server and move the binary to the `UnrealEnv` folder, which is our default location for binaries, the folder structures are as follows:
```
gym-unrealcv/  
|-- docs/                  
|-- example/                
|-- gym_unrealcv/              
|   |-- envs/    
|   |   |-- agent/     
|   |   |-- UnrealEnv/                    # Binary default location
|   |   |   |-- Collection_WinNoEditor/   # Binary folder
|   |-- setting/
|   |   |-- env_config/                   # environment config json file location  
...
generate_env_config.py                    # generate environment config json file
...
```
2. Run **generate_env_config.py** to automatically generate and store the config JSON file for the desired map
```
python generate_env_config.py --env-bin {binary relative path} --env-map {map name}  
# binary relative path : the executable file path relative to UnrealEnv folder
# map name: the user desired map for running.

#example:
python generate_env_config.py --env-bin Collection_WinNoEditor\\WindowsNoEditor\\Collection\\Binaries\\Win64\\Collection.exe --env-map track_train
```
3. Create a new python file in ```/gym-unrealcv/gym_unrealcv/envs```, and write your environment in this file. You can inheriting the base classes in [unrealcv_bas.py](./gym_unrealcv/envs/base_env.py), which provide the basic functions for the gym environment. You can also refer to the existing environments in the same folder for more details. Note that you need to write your own reward function in the new environment.

4. Import your environment into the ```__init__.py``` file of the collection. This file will be located at ```/gym-unrealcv/gym_unrealcv/envs/__init__.py.``` Add ```from gym_unrealcv.envs.{your_env_file} import {Your_Env_Class}``` to this file.
5. Register your env in ```gym-unrealcv/gym_unrealcv/_init_.py```
We have predefined a naming rule to define different environments and their corresponding task interfaces, we encourage you to follow this rule to name your environment. The naming rule is as follows:  
```Unreal{task}-{MapName}-{ActionSpace}{ObservationType}-v{version} ```
- ```{task}```: the name of the task, we currently support: ```Track```,```Navigation```,```Rendezvous```.
- ```{MapName}```: the name of the map you want to run, ```track_train```, ```Greek_Island```, etc.
- ```{ActionSpace}```: the action space of the agent, ```Discrete```, ```Continuous```, ```Mixed```. (Only Mixed type support interactive actions)
- ```{ObservationType}```: the observation type of the agent, ```Color```, ```Depth```, ```Rgbd```, ```Gray```, ```CG```, ```Mask```, ```Pose```,```MaskDepth```,```ColorMask```.
- ```{version}```: works on ```track_train``` map, ```0-5``` various the augmentation factor(light, obstacles, layout, textures).

6. Test your environment by running a random agent
```
python example/random_agent.py -e YOUR_ENV_NAME
```
If your environment is designed for multi-agent, you can run the following command:
```
python example/random_multi_agent.py -e YOUR_ENV_NAME
```

You will see your agent take some actions randomly and get reward as you defined in the new environment.
