UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI
===
### ICCV 2025 (Highlights🚀)

[📄 Paper](https://arxiv.org/abs/2412.20977) \| [🌐 Project Page](http://unrealzoo.site/) \| [📝 Document](https://unrealzoo.notion.site/) \| [🤖 ModelScope](https://www.modelscope.cn/organization/UnrealZoo) \| 
[🎬 YouTube](https://www.youtube.com/watch?v=Xe2VmsJYTAU)
![framework](doc/figs/overview.png)

# 🔥 News
- 2025/09 We update package to **v1.0.4**, which supports optical flow rendering and fixes some camera-related bugs.  
- 2025/07 We released the full version of UnrealZoo environment package, built on UE5.5 which contains **100+ scenes**.
  - **UnrealZoo_UE5_5_V1.0.1** is available for download, which contains 100+ scenes and playable entities, with a total size of **67GB**.
  - We integrate the latest chaos system for vehicles, enabling more realistic animation(enter/exit) and visual effect(collision, explode, fire, etc.).
  - The **object interaction(pick/drop)** now is available, you could spawn the object anywhere in the map and interact with it, just use one command to change its appearance.
- 2025/03 New Download Link in 🤖[ModelScope](https://www.modelscope.cn/organization/UnrealZoo) is released for fast downloading the UE4/UE5 binary.
- 2024/12 Paper Link: [UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI](https://arxiv.org/abs/2412.20977)
- 2024/12 Comprehensive documentation and Scene Gallery are available in [Notion Page](https://unrealzoo.notion.site/scene-gallery) 
- 2024/12 Project website is available at [UnrealZoo](http://unrealzoo.site/)


# Table of Contents

1. [What is UnrealZoo?](#what-is-unrealzoo)
   - [Key Features](#key-features)
   - [Framework](#framework)
2. [Quick Installation](#quick-installation)
   - [Dependencies](#dependencies)
   - [Install UnrealZoo-Gym](#install-gym-unrealzoo)
   - [Prepare UE Binary](#prepare-ue-binary)
3. [Run the Example Code](#run-the-example-code)
4. [Build your agent](#build-your-agent)
5. [Documentation](#documentation)
6. [TODO List](#todo-list)
7. [License and Acknowledgments](#license-and-acknowledgments)
8. [Citation](#citation)


## What is UnrealZoo?

UnrealZoo is a rich collection of photo-realistic 3D virtual worlds built on Unreal Engine, designed to reflect the complexity and variability of the open worlds. 
There are various playable entities for embodied AI, including human characters, robots, vehicles, and animals.
Integrated with [UnrealCV](https://unrealcv.org/), UnrealZoo provides a suite of easy-to-use Python APIs and tools for various potential applications, such as data annotation and collection, environment augmentation, distributed training, and benchmarking agents.

**💡This repository provides the gym interface based on UnrealCV APIs for UE-based environments, which is compatible with OpenAI Gym and supports the high-level agent-environment interactions in UnrealZoo.**

### Key Features
- **Photorealistic**: High-quality graphics rendering empowered by Unreal Engine (4.27/5.4).
- **Large-scale**: 100+ Scenes, the largest one covers 16km².
- **Diverse Scenes**: landscape🏜️🏞️🏝️, historic architecture⛩️🛕🏰, settlement🏘️🏙️, industrial zone🏭🏗️, facilities🤽🏻‍♂️🚉🏪...
- **Diverse Bodies**: human🚶🏻‍♂️️🚶🏻, robot🤖, vehicles🚗🏍️🛩️,animals🐘🐕🐎🐧🐢🐖🐏🐂...
- **Diverse Actions**: running🏃🏻‍♂️, climbing🧗🏻, sitting🧘🏻, jumping, squatting, pick-up...
- **Easy-to-use**: Pre-build UE binaries, integrated with optimized UnrealCV, are to be used without any prior knowledge of UE.
- **Cross-platform**: Runs on Linux, Windows, macOS
- **Flexible Python APIs**: Provide UnrealCV Python APIs and Gym Interfaces for various potential applications.
- **Multi-agent Interactions**: Real-time interaction of 10+ vision-based agents👩‍👩‍👧‍👧 in one scene.

### Framework
![framework](doc/figs/framework.png)

[//]: # (- ```UnrealCV``` is the basic bridge between ```Unreal Engine``` and ```OpenAI Gym```.)

[//]: # (- ```OpenAI Gym``` is a toolkit for developing an RL algorithm, compatible with most numerical computation libraries, such as TensorFlow or PyTorch. )
- The ```Unreal Engine Environments (Binary)``` contains the scenes and playable entities.
- The ```UnrealCV+ Server``` is built in the UE binary as a plugin, including modules for rendering , data capture, object/agent control, and command parsing. We have optimized the rendering pipeline and command system in the server.
- The ```UnrealCV+ Client``` provides Python-based utility functions for launching the binary, connecting with the server, and interacting with UE environments. It uses IPC sockets and batch commands for optimized performance.
- The ```OpenAI Gym Interface``` provides agent-level interface for agent-environment interactions, which has been widely used in the community. Our gym interface supports customizing the task in a configuration file and contains a toolkit with a set of gym wrappers for environment augmentation, population control, etc.

## Quick Installation
### Dependencies
- UnrealCV
- Gym
- CV2
- Matplotlib
- Numpy
- Docker(Optional)
- Nvidia-Docker(Optional)
 
We recommend you use [anaconda](https://www.continuum.io/downloads) to install and manage your Python environment.
```CV2``` is used for image processing, like extracting object masks and bounding boxes. ```Matplotlib``` is used for visualization.
### Install UnrealZoo-Gym

It is easy to install unrealzoo-gym, just run
```
git clone https://github.com/UnrealZoo/unrealzoo-gym.git
cd unrealzoo-gym
pip install -e . 
```
While installing gym-unrealcv, dependencies including OpenAI Gym, UnrealCV, numpy and matplotlib are installed.

### Prepare UE Binary
Before running the environments, you need to prepare unreal binaries. To accommodate users' local memory limitations, we provide two different lightweight environment options based on **UE4 and UE5**. A larger preview-version environment package, which currently contain 50 scene maps (>60GB), is also available for download **here**. The full version of the UnrealZoo environment package will be released soon.

**UE4 Example Scenes**
<table>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/track_train.png" width="480" height="200">  
        <figcaption style="text-align: center;">Track Train</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/Greek_island.png" width="480" height="200">  
        <figcaption style="text-align: center;">Greek Island</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/Containeryard_Day.png" width="480" height="200">  
        <figcaption style="text-align: center;">ContainerYard_Day</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/ContainerYard_Night.png" width="480" height="200">  
        <figcaption style="text-align: center;">ContainerYard_Night</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/SuburbNeighborhood_Day.png" width="480" height="200">  
        <figcaption style="text-align: center;">SuburbNeighborhood_Day</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/SuburbNeighborhood_Night.png" width="480" height="200">  
        <figcaption style="text-align: center;">SuburbNeighborhood_Night</figcaption>
      </figure>
    </td>

  </tr>
</table>

**UE5 Example Scenes**
<table>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/UE5_ExampleScene/ChemicalFactory.png" width="480" height="200">  
        <figcaption style="text-align: center;"> ChemicalFactory</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE5_ExampleScene/ModularOldTown.png" width="480" height="200">  
        <figcaption style="text-align: center;">Modular Old Town</figcaption>
      </figure>
    </td>

  </tr>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/UE5_ExampleScene/MiddleEast.png" width="480" height="200">  
        <figcaption style="text-align: center;">MiddleEast</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE5_ExampleScene/Roof-City.png" width="480" height="200">  
        <figcaption style="text-align: center;">Roof-City</figcaption>
      </figure>
    </td>
 
  </tr>
</table> 

**Preview-version Environment Package (UE4_Collection_Preview)**  
The preview-version environment package contains **60+ scene maps** in total, the detailed map name can be found here [EnvLists](doc/EnvLists.md).

**Full version Environment Package (UnrealZoo_UE5_5_V1.0.1)**
We released the full version of UnrealZoo environment package, built on UE5.5 which contains **100+ scenes**. All maps in the **[scene gallery](https://www.notion.so/Scene-Gallery-a801475ff98943159da66f641f4c38b2)** are included.


🗂️**You can load the above binaries using script `python load_env.py -e {Environment}` or from the following link:**

| Environment                       | Download Link                                                              | Size       |
|-----------------------------------|----------------------------------------------------------------------------|------------|
| UE4_ExampleScene                  | [🤖ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE4/files) | ~3GB       |
| UE5_ExampleScene                  | [🤖ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/files) | ~10GB      |
| UE4_Collection_Preview (60+ maps) | [🤖ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE4/files) | **\>48GB** |
|**UnrealZoo_UE5_5_Mac_V1.0.3** | [Cloud](https://pan.baidu.com/s/1lUqQYEkn7mfXd8G8F2tQGA?pwd=xvxh)|    ~67GB |
| **UnrealZoo_UE5_5_Win64_V1.0.6**  | [🤖ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/files)  |    ~72GB |
| **UnrealZoo_UE5_5_Linux_V1.0.5**  | [🤖ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/files) |    ~72GB |


Then unzip and move the downloaded binary to the `UnrealEnv` folder, which is our default location for binaries, the folder structures are as follows:
```
unrealzoo-gym/  
|-- docs/                  
|-- example/                
|-- gym_unrealcv/              
|   |-- envs/    
|   |   |-- agent/     
|   |-- setting/
|   |   |-- env_config/                   # environment config json file location  
...
|-- generate_env_config.py                    # generate environment config json file
...

UnrealEnv/                    # Binary default location
|-- Collection_WinNoEditor/   # Binary folder
|-- UE4_Example_Scene/        # Binary folder
...

```
If there is a permission issue, run the command ```chmod +x ./``` under the ```UnrealEnv``` folder path to give the necessary permissions.
#### Available Map Name in Exemplar Binary

| UE4                     | UE5                     |
|-------------------------|-------------------------|
| track\_train            | Map\_ChemicalPlant\_1   |
| Greek\_Island           | Old\_Town               |
| ContainerYard\_Day      | MiddleEast              |
| ContainerYard\_Night    | Demo\_Roof              |
| SuburbNeighborhood\_Day |                         |
| SuburbNeighborhood\_Night|                        |

#### Naming rule
We have predefined a naming rule to launch different environment maps and their corresponding task interfaces.  
```Unreal{task}-{MapName}-{ActionSpace}{ObservationType}-v{version} ```
- ```{task}```: the name of the task, we currently support: ```Track```,```Navigation```,```Rendezvous```.
- ```{MapName}```: the name of the map you want to run, ```track_train```, ```Greek_Island```, etc.
- ```{ActionSpace}```: the action space of the agent, ```Discrete```, ```Continuous```, ```Mixed```. (Only Mixed type support interactive actions)
- ```{ObservationType}```: the observation type of the agent, ```Color```, ```Depth```, ```Rgbd```, ```Gray```, ```CG```, ```Mask```, ```Pose```,```MaskDepth```,```ColorMask```.
- ```{version}```: works on ```track_train``` map, ```0-5``` various the augmentation factor(light, obstacles, layout, textures).

## Run the Example Code
#### Hint 💡 
- If your mouse cursor disappears after the game launches, press ``` ` ``` (the key above Tab) to release the mouse cursor from the game.

#### 1. Specify the environment location： 
 ##### Option 1: In the terminal. 
 - Default path to UnrealEnv is in user home directory under .unrealcv  
   - Windows: C:\\Users\\{username}\\.unrealcv\\UnrealEnv
   - Linux: /home/{username}/.unrealcv/UnrealEnv
   - Mac: /Users/{username}/.unrealcv/UnrealEnv
```
export UnrealEnv=/your/path/to/UnrealEnv
```
##### Option 2: In the example code:
Add environment variable in the example code
```
import os
os.environ['UnrealEnv']="/your/path/to/UnrealEnv"
```
#### 2. Run random agents
User could choose a map from the available map list, and run the random agent to interact with the environment.
```
python ./example/random_agent_multi.py -e UnrealTrack-track_train-ContinuousColor-v0
```
#### 3. Run a rule-based tracking agent 
```
python ./example/tracking_demo.py -e UnrealTrack-Greek_Island-ContinuousColor-v0
```

#### 4. Run a keyboard tracking agent 
Use the "I", "J", "K", and "L" keys to control the agent's movement.
```
python ./example/keyboard_agent.py -e UnrealTrack-Old_Town-MixedColor-v0
```
#### 5. Run a keyboard Navigation agent
Use the "I", "J", "K", and "L" keys to control the agent's movement.  
Space( ␣ ): Jump.  
Up Arrow(⬆️): Look up.  
Down Arrow(⬇️): Look down.  
(Double "Jump" will trigger the agent to climb)
```
python ./example/Keyboard_NavigationAgent.py -e UnrealNavigation-Demo_Roof-MixedColor-v0
```
Control the agent to navigate to the target location by using the keyboard. 

<table style="width: 100%; text-align: center;">
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/navigation/map.png" width="400" height="220">
           <figcaption> Map Reference(Demo_Roof)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/navigation/target.png" width="380" height="220">
           <figcaption>Target Reference(Demo_Roof)</figcaption>
      </figure>
    </td>
  </tr>

  <tr>
    <td>
      <figure>
        <img src="./doc/figs/navigation/map_2.png" width="400" height="220">
           <figcaption> Map Reference(SuburbNeighborhood_Day)</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/navigation/target_2.png" width="380" height="220">
           <figcaption>Target Reference(SuburbNeighborhood_Day)</figcaption>
      </figure>
    </td>
  </tr>
</table> 

## Build your agent
You can build your agent based on the latest method in reinforcement learning, evaluating it in UnrealZoo.
- Reinforcement Learning
  - Online RL
    - [Visual Navigation Agent]() comming soon..

  - Offline RL
    - [Embodied Visual Tracking Agent(ECCV24)](https://github.com/wukui-muc/Offline_RL_Active_Tracking)
- Large Vision-Language Model
  


## Documentation
- We provide a set of wrappers for FPS control, population randomization, etc.
Please refer to the [wrapper](doc/wrapper.md) for more details.
- We provide a document for adding a new environment in unrealzoo-gym. Please refer to the [addEnv](doc/addEnv.md) for more details.

# 🗓️ TODO List
- [x]  Release an all-in-one package of the collected environments
- [ ]  Add gym interface for heterogeneous mutli-agent co-operation.
- [x]  Expand the list of supported interactive actions.
- [x]  Add more detailed examples for reinforcement learning agents.
- [x]  Add more detailed examples for large vision-language models.

## License and Acknowledgments
The UnrealZoo project is licensed under the Apache 2.0. 
We acknowledge the following projects for their contributions to the UnrealZoo project:
- [UnrealCV](https://unrealcv.org/)
- [OpenAI Gym](https://gym.openai.com/)
- [Unreal Engine](https://www.unrealengine.com/)
- [The fantastic Plugins/Content from UE Marketplace](https://www.unrealengine.com/marketplace)
  -  [Smart Locomotion](https://www.fab.com/zh-cn/listings/7f881534-bf40-493b-97b4-a917daa87af0)
  -  [Animal Pack](https://www.fab.com/zh-cn/listings/856c42d7-58a3-4b95-8f70-1302e5bdafa0)
  -  [Drivable Car](https://www.fab.com/zh-cn/listings/65a0844c-6be4-4e38-9d7a-b9697681a274)
  - ...

## Citation

If you use UnrealZoo in your research, please consider citing:

```bibtex
@inproceedings{zhong2025unrealzoo,
  title={Unrealzoo: Enriching Photo-realistic Virtual Worlds for Embodied AI},
  author={Zhong, Fangwei and Wu, Kui and Wang, Churan and Chen, Hao and Ci, Hai and Li, Zhoujun and Wang, Yizhou},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5769--5779},
  year={2025}
}
