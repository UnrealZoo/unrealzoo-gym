<div align="center">

<!-- Hero Banner -->
<a href="https://youtu.be/Xe2VmsJYTAU">
  <img src="https://img.youtube.com/vi/Xe2VmsJYTAU/maxresdefault.jpg" width="100%" alt="UnrealZoo Demo Video">
</a>
<p align="center"><i>▶️ 点击观看完整演示视频</i></p>

<!-- Title -->
<h1>UnrealZoo</h1>
<h3>Large-scale Photo-realistic Virtual Worlds for Embodied AI</h3>
<p><b>面向具身智能体训练、评估与压力测试的照片级、可交互、异构世界研究平台。</b></p>
<p>100+ 场景 · 10+ 智能体 · 感知、导航、交互、协作与数据采集</p>

<!-- Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/UE-5.7-blue?style=flat-square&logo=unrealengine" />
  <img src="https://img.shields.io/badge/Python-3.9+-green?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Win%20%7C%20Mac-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Scenes-100+-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Agents-10+-ff69b4?style=flat-square" />
  <img src="https://img.shields.io/badge/ICCV_2025-Highlights-red?style=flat-square" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/✅-Production%20Ready-success" />
  <img src="https://img.shields.io/badge/🚀-Out%20of%20the%20Box-blueviolet" />
  <img src="https://img.shields.io/badge/👥-Multi--Agent-ff69b4" />
  <img src="https://img.shields.io/badge/🎮-Ready%20to%20Use-9cf" />
</p>

<!-- Language Switch -->
<p align="center">
  <a href="README.md">🇺🇸 English</a> | 🇨🇳 中文
</p>

<!-- Quick Links -->
<p align="center">
  <a href="#-quick-start"><b>🚀 Quick Start</b></a> •
  <a href="http://unrealzoo.site/"><b>🌐 Website</b></a> •
  <a href="https://arxiv.org/abs/2412.20977"><b>📄 Paper</b></a> •
  <a href="https://unrealzoo.notion.site/"><b>📚 Docs</b></a> •
</p>

</div>

---

## 📖 项目概览

<img src="doc/figs/overview.png" width="100%" alt="UnrealZoo Overview">

UnrealZoo is a rich collection of photo-realistic 3D virtual worlds built on Unreal Engine, designed to reflect the complexity and variability of the open worlds. There are various playable entities for embodied AI, including human characters, robots, vehicles, and animals.

Integrated with [UnrealCV](https://unrealcv.org/), UnrealZoo provides a suite of easy-to-use Python APIs and tools for various potential applications, such as data annotation and collection, environment augmentation, distributed training, and benchmarking agents.

**💡 This repository provides the gym interface based on UnrealCV APIs for UE-based environments, which is compatible with OpenAI Gym and supports the high-level agent-environment interactions in UnrealZoo.**

---

## 🔥 最新动态

> **UnrealZoo v3.1 在 v3.0 基础上继续扩展**，加入更快的视觉观测、三维感知、物理机器人、运行时智能体定制和外部打包环境支持。

### 🚀 v3.1 功能更新

| 特性 | 状态 | 说明 |
|------|------|----------------------------------------|
| **更快的 UnrealCV 采集** | ✅ 增强 | 标准相机的**采集吞吐率**达到 **2K 53.46 FPS**、**4K 29.59 FPS**（最高 6.31×）；**串行端到端采集**平均达到 **2K 22.40 FPS**、**4K 16.25 FPS** |
| **LiDAR 观测** | ✅ 新增 | 提供 XYZI 点云观测和玩家控制的街景建图示例 |
| **占用体素观测** | ✅ 新增 | 提供兼容 LINGO 的布尔占用网格，以及 bounds/mesh 两种模式 |
| **扩展全景观测** | ✅ 新增 | 扩展全景采集模态，用于具身感知与数据集采集 |
| **[共享内存观测传输](https://docs.unrealcv.org/en/latest/reference/unrealzoo_capture_transport.html)** | ✅ 新增 | 以更低延迟访问原始相机与占用观测 |
| **[运行时反射](https://docs.unrealcv.org/en/latest/unrealcv_plus/reference/runtime-reflection.html)** | ✅ 新增 | 通过 JSON 检查受支持的 Unreal 对象，并访问属性和调用函数 |
| **[电影相机控制](https://docs.unrealcv.org/en/latest/reference/cine_camera.html)** | ✅ 新增 | 提供物理相机设置、手动对焦控制与派生内参 |
| **[MQRC 采集](https://docs.unrealcv.org/en/latest/unrealcv_plus/reference/mqrc-rendering.html)** | ✅ 新增 | 提供可显式控制渲染与后处理的高质量光照图像采集 |
| **MuJoCo Unitree Go1** | ✅ 新增 | 提供键盘运动控制和 Robot Parkour 高级策略示例 |
| **无人机运行时视觉定制** | ✅ 新增 | 无需重新生成无人机即可切换五种带动态螺旋桨的正式模型、使用定制模板，或加载兼容的外部 Static Mesh |
| **社交动画** | ✅ 新增 | 运行时选择新增的聚会、日常和车内角色动画 |
| **外部 3DGS 环境** | ✅ 新增 | 动态加载用户打包的 3DGS 资产，并复用 UnrealZoo 智能体、相机和任务 API |
| **Runtime MCP** | ✅ 新增 | 将兼容 MCP 的智能体连接到运行中的 UnrealZoo 环境，用于场景检查和任务控制 |

📄 [v3.1 Changelog](doc/CHANGELOG_v3.1.md) · 📚 [v3.1 功能指南](example/new_features/README.md) · 📚 [UnrealCV+ 文档](https://docs.unrealcv.org/en/latest/unrealcv_plus/index.html)

> 💡 **命令参考：** 完整的 UnrealCV+ 命令列表参见[命令 Reference](https://docs.unrealcv.org/en/latest/unrealcv_plus/reference/commands.html)。

<details>
<summary><b>🚀 v3.0 核心更新</b></summary>

> **UnrealZoo v3.0 已发布！** 这是迄今为止最大的更新，带来完整的异构多智能体协同能力、开箱即用的交互系统，以及 UnrealCV+ Plugin 的全面升级。

| 特性 | 状态 | 说明                                     |
|------|------|----------------------------------------|
| **异构多智能体协同** | ✅ 已发布 | 地面 + 无人机编队跟随（UE内置导航/python外置导航 API示例）  |
| **模板化 Agent Spawn** | ✅ 已发布 | 运行时动态生成 agent，支持混合类别配置                 |
| **增强交互系统** | ✅ 已发布 | 开门/上下车/拾取/下蹲/跳跃/攀爬，提供 API-键盘映射便于理解交互逻辑 |
| **NavMesh 路径规划** | ✅ 已发布 | 通过 API 计算最短路径导航点，支持智能体自主导航控制，以及路径点导出   |
| **VLN Baseline 示例** | ✅ 已添加 | 将 Uni-NaVid、ViNT/NoMaD 和 StreamVLN 接入 UnrealZoo 导航任务，提供统一的 `env.step()` 示例。详见 [VLN Baseline README](example/VLN_Baseline/README.md) |

</details>

### 📦 版本与环境包兼容关系

| 代码分支 | Binary 环境包 | UE 版本 | 状态 | 推荐用途 | 下载 |
|---|---|---:|---|---|---|
| **v3.1** | **最新 UE5.7 完整版** | 5.7 | **当前推荐** | 最新感知、Go1、Runtime MCP、外观定制与外部 3DGS 功能 | [ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/files) · [Hugging Face](https://huggingface.co/datasets/UnrealZoo/UnrealZoo_UE5) |
| **v3.0** | UE5.6 完整版 | 5.6 | 上一稳定版本 | 多智能体任务与既有 v3.0 工作流 | [ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/tree/master/UnrealZoo_UE5_6_v3.0.0) |
| **v2.0** | UE5 / UE4 示例包 | UE5 / UE4 | 旧版兼容 | 旧教程与示例场景兼容 | [UE5 环境包](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/files) · [UE4 环境包](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE4/files) |

推荐的 v3.1 UE5.7 完整包大小约为 **70 GB**。

### 📜 发展历程

<details>
<summary><b>点击查看历史更新</b></summary>

#### 2024-12: 论文发布
- 📝 Paper: [UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI](https://arxiv.org/abs/2412.20977)
- 🌐 Website: [unrealzoo.github.io](https://unrealzoo.github.io/)
- 📚 Notion 文档: [Scene Gallery](https://www.notion.so/Scene-Gallery-a801475ff98943159da66f641f4c38b2)

#### 2025-01: UE5.6 完整环境包
- ✅ 100+ 场景，67GB 完整包
- ✅ Chaos 物理系统（载具、碰撞、爆炸、火焰）
- ✅ 物体交互系统（拾取/丢弃）
- ✅ 外观切换系统（玩家/动物类别合并）
- ✅ 跨平台二进制支持（Win/Mac/Linux 自动配置）
- ✅ ModelScope 国内镜像（高速下载通道）

#### 2026-04: v3.0 正式发布
- ✅ 异构多智能体协同
- ✅ 模板化 Agent Spawn
- ✅ 全套增强交互系统演示代码（API-键盘映射）
- ✅ NavMesh 路径规划以及任务应用演示代码
- ✅ UnrealCV+ Plugin 全面升级

#### 2026: v3.1 功能更新
- ✅ 更快的 UnrealCV 视觉观测与录制工作流
- ✅ LiDAR 和占用体素观测
- ✅ MuJoCo Go1 仿真示例
- ✅ 用户打包的 3DGS 环境支持
- ✅ 支持动态内置模型与外部 Static Mesh 的无人机运行时视觉定制
- ✅ 扩展角色社交动画

</details>

[查看 v3.1 更新日志](doc/CHANGELOG_v3.1.md)

---

## 🚀 快速开始

以下 smoke test 对应 **v3.1 分支 + UE5.7 完整包**，需要 Python 3.9+、
Git 和对应平台的 UnrealZoo binary。v3.1 新功能 Demo 当前主要在 Windows
环境中完成验证。

### 1. 创建 Conda 环境并安装（推荐）

```bash
git clone https://github.com/UnrealZoo/unrealzoo-gym.git
cd unrealzoo-gym
conda create -n unrealzoo python=3.9 -y
conda activate unrealzoo
python -m pip install --upgrade pip
python -m pip install -e .
```

Editable install 会从 PyPI 安装所需的 **UnrealCV 1.3.0** Python client。

```bash
python -c "import gym_unrealcv, unrealcv; print('gym_unrealcv OK; unrealcv', unrealcv.__version__)"
```

### 2. 下载并配置 UE5.7 环境包

从 [ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/files) 或
[Hugging Face](https://huggingface.co/datasets/UnrealZoo/UnrealZoo_UE5) 下载
**最新 UE5.7 完整版（推荐）**，解压后将 `UnrealEnv` 指向环境包所在目录。

**Windows CMD**

```cmd
set "UnrealEnv=D:\path\to\UnrealEnv"
```

**Linux / macOS**

```bash
export UnrealEnv=/path/to/UnrealEnv
```

```bash
python -c "import os; p=os.environ.get('UnrealEnv'); assert p and os.path.isdir(p), f'Invalid UnrealEnv: {p}'; print('UnrealEnv:', p)"
```

使用旧 binary 或分支前，请先查看[版本与环境包兼容关系](#-版本与环境包兼容关系)。

### 3. 运行 Smoke Test

Navigation 环境会根据 `UnrealEnv` 自动索引并启动 binary：

```bash
python example/navigation/keyboard/navigation_keyboard_human.py -e UnrealNavigation-SuburbNeighborhood_Day-MixedColor-v0
```

> 💡 **运行规则：** 常规 Gym Demo 会自动启动配置对应的 binary；Go1
> 以及部分录制/运行时工具需要连接到用户手动启动的 binary 或 Editor。
> 如果鼠标光标消失，按 `` ` ``（Tab 上方）释放鼠标。

### 4. 最小 Gym API 示例

```python
import gym
import numpy as np
import gym_unrealcv  # 注册 UnrealZoo environments。

env = gym.make("UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0")
obs = env.reset()

for _ in range(100):
    obs, reward, done, info = env.step(env.action_space.sample())
    if bool(np.asarray(done).all()):
        obs = env.reset()

env.close()
```

UnrealZoo v3.1 使用 OpenAI Gym API。多智能体环境返回按智能体组织的
observation、action、reward 和 done；具体视觉模态由注册任务配置决定。

---

## 🌟 功能与 Demo

### 为什么选择 UnrealZoo？

| 平台能力 | 研究价值 |
|---|---|
| **100+ 照片级世界与异构智能体** | 在城市、自然、建筑和工业场景中评估人类、机器人、车辆与动物智能体 |
| **具身感知** | 在同一可交互世界中组合 RGB-D、Mask、全景、LiDAR 与占用观测 |
| **交互式多智能体环境** | 研究导航、操作、载具交互、追踪与异构协作 |
| **运行时扩展能力** | 加载外部 3DGS 环境、定制资产与外观，并复用现有智能体和任务 API |
| **端到端智能体工作流** | 接入 Gym 策略、VLN/VLM 系统、Runtime MCP 智能体与数据采集管线 |

### 选择研究工作流

| 我想研究 | 建议入口 | 运行方式 |
|---|---|---|
| **多智能体协作 / 追踪** | [多智能体随机策略](example/multi_agent/baseline/multi_random_baseline.py) · [追踪示例](example/tracking/basic/tracking_auto_basic.py) | 自动启动 binary |
| **交互式导航** | [键盘导航](example/navigation/keyboard/navigation_keyboard_human.py) | 自动启动 binary |
| **RGB-D / LiDAR / 占用感知** | [v3.1 感知指南](example/new_features/README.md) · [LiDAR 建图](example/new_features/suburb_street_slam.py) · [占用可视化](example/new_features/realtime_scene_occupancy_gpu.py) | 自动启动 binary |
| **Unitree Go1 控制与跑酷** | [MuJoCo Go1 指南](example/mujoco/README.md) | 手动启动 binary 或 Editor |
| **VLN / VLM 智能体** | [VLN Baseline 指南](example/VLN_Baseline/README.md) | 按模型说明配置 |
| **Runtime MCP 智能体** | [Runtime MCP 示例](https://github.com/unrealcv/unrealcv-runtime-mcp) | 连接运行中的环境 |
| **数据采集 / 标注** | [视频录制管线](example/DataRecording/VideoRecordingPipeline.py) · [UnrealCV+ 文档](https://docs.unrealcv.org/en/latest/unrealcv_plus/index.html) | 手动启动 binary |

### 🎬 可视化展示

<div align="center">

**🐕 MuJoCo Go1 控制与跑酷**

| 键盘控制 | Parkour：第三人称视角 | Parkour：深度观测 |
|:---:|:---:|:---:|
| <img src="doc/figs/new_features/mujoco_go1_keyboard.gif" width="100%" alt="MuJoCo Go1 keyboard control"> | <img src="doc/figs/new_features/mujoco_go1_parkour_third_person.gif" width="100%" alt="MuJoCo Go1 parkour third-person view"> | <img src="doc/figs/new_features/mujoco_go1_parkour_depth.gif" width="100%" alt="MuJoCo Go1 parkour depth observation"> |
| 基础 `I/J/K/L` 运动控制 | 从机器人外部展示高级策略行为 | UnrealCV 原始深度与策略使用的深度输入 |

| 🚁 无人机运行时视觉定制 | 🎭 角色社交动画 | 🌐 外部 3DGS + UnrealZoo Actor |
|:---:|:---:|:---:|
| <img src="doc/figs/new_features/drone_mesh_switch.gif" width="100%" alt="UnrealZoo runtime drone mesh switching"> | <img src="doc/figs/new_features/social_animation.gif" width="100%" alt="UnrealZoo character social animations"> | <img src="doc/figs/new_features/3dgs_dynamic_load.gif" width="100%" alt="External 3DGS environment loaded with a supported UnrealZoo actor"> |
| [五种带动画的内置模型与外部 Static Mesh 支持](example/new_features/README.md#4-runtime-drone-visual-customization) | [`set_social_anim` 大小写敏感参数列表](example/new_features/README.md#5-character-social-animations) | 加载外部 3DGS 场景并复用 UnrealZoo 支持的 Actor |

**🌐 实时三维场景感知：占用体与全景深度**

| 实时占用体与全景深度观测 |
|:---:|
| <img src="doc/figs/new_features/perception/live_occupancy_panorama_depth.gif" width="100%" alt="UnrealZoo 中同步显示的实时占用体与全景深度观测"> |
| 相机相对坐标系下的占用更新与连续 360 度深度信息相互补充，为具身感知、导航和空间推理提供几何环境信息。参见[占用观测指南](https://docs.unrealcv.org/en/latest/unrealcv_plus/reference/scene-perception.html)和 [GPU 可视化示例](example/new_features/realtime_scene_occupancy_gpu.py)。 |

**🤖 Runtime MCP 智能体工作流**

| 复杂场景导航 | 场景描述 | 角色外观控制 |
|:---:|:---:|:---:|
| <img src="doc/figs/new_features/runtime_mcp/complex_scene_navigation.gif" width="100%" alt="Runtime MCP 复杂场景导航"> | <img src="doc/figs/new_features/runtime_mcp/scene_caption.gif" width="100%" alt="Runtime MCP 场景描述"> | <img src="doc/figs/new_features/runtime_mcp/change_character_appearance.gif" width="100%" alt="Runtime MCP 角色外观控制"> |
| 在场景地标之间导航，并从多个相机视角验证结果。 | 检查场景、采集六个方向的视图并合成完整描述。 | 发现并调用 Blueprint 外观接口，然后采集十种角色外观。 |

提示词、操作流程、截图和录制工作流参见 [Runtime MCP 示例](https://github.com/unrealcv/unrealcv-runtime-mcp)。

**🎥 电影摄影机手动对焦（Cine Camera）**

<p align="center">
  <img src="doc/figs/new_features/cine-focus-mqrc-demo-v4.gif" width="48%" alt="Cine Camera manual focus demo">
  <img src="doc/figs/new_features/cine-focus-mqrc-demo.gif" width="48%" alt="Cine Camera MQRC focus demo">
</p>

上面的演示展示了 **[Cine Camera](https://docs.unrealcv.org/en/latest/reference/cine_camera.html)** 的手动对焦距离动态调整效果。

**📡 LiDAR 街景建图**

<p align="center">
  <img src="doc/figs/new_features/lidar_street_slam.gif" width="70%" alt="UnrealZoo Suburb LiDAR voxel mapping">
</p>

键盘控制 LiDAR 观测，并结合实时位姿更新地图。

**🚗 载具交互**

<div align="center">

| | | |
|:---:|:---:|:---:|
| ![](doc/figs/interactiondemo_gif/vehicle/car.gif) | ![](doc/figs/interactiondemo_gif/vehicle/motorbike.gif) | ![](doc/figs/interactiondemo_gif/vehicle/enter_exit_car.gif) |

</div>

**🤸 动作与交互**

<table><tr>
<td><img src="doc/figs/interactiondemo_gif/action/squat.gif" height="180"></td>
<td><img src="doc/figs/interactiondemo_gif/action/kick.gif" height="180"></td>
<td><img src="doc/figs/interactiondemo_gif/action/climb.gif" height="180"></td>
<td><img src="doc/figs/interactiondemo_gif/action/climb_ladder.gif" height="180"></td>
<td><img src="doc/figs/interactiondemo_gif/action/climb_woman.gif" height="180"></td>
</tr></table>

**🤖 多样化可控智能体**

<div align="center">

| 无人机 (Drone) | 机器狗 (Robot Dog) | 多智能体协同 |
|:---:|:---:|:---:|
| ![](doc/figs/interactiondemo_gif/drone.gif) | ![](doc/figs/interactiondemo_gif/mobilerobot.gif) | ![](doc/figs/interactiondemo_gif/cooperation/cooperation.gif) |

</div>

</div>

### 🎮 运行示例

<details open>
<summary><b>✨ v3.1 功能示例</b></summary>

请在仓库根目录的 Windows CMD 中运行以下命令。Demo 使用已注册环境的
配置，并通过用户已有的 `UnrealEnv` 路径自动索引 binary：

```cmd
python example\new_features\suburb_street_slam.py
python example\new_features\realtime_scene_occupancy_gpu.py --method mesh
python example\new_features\drone_mesh_switch_demo.py --interval 2 --cycles 2 --render
```

两个 Go1 示例连接到用户手动启动的 binary 或 Editor PIE。请等待地图和
UnrealCV server 完成加载，再使用与服务一致的端口运行：

```cmd
python example\mujoco\mujoco_robot_demo.py go1 keyboard --host 127.0.0.1 --port 9000
python example\mujoco\parkour\demo.py --host 127.0.0.1 --port 9000 --command-mode keyboard
```

两个 Go1 示例均使用 `I/K` 前进后退、`J/L` 转向。高级示例适配自官方
[Robot Parkour Learning 仓库](https://github.com/ZiwenZhuang/parkour)，策略来源与引用信息见
[MuJoCo Parkour 文档](example/mujoco/docs/PARKOUR.md)。

**社交动画：** 参见 [`set_social_anim` API 与大小写敏感参数列表](example/new_features/README.md#5-character-social-animations)。

参数、控制方式、资产要求和观测格式见 [v3.1 功能指南](example/new_features/README.md)。

</details>

<details>
<summary><b>🌐 动态加载 3DGS 环境</b></summary>

启动 UnrealZoo v3.1 binary，打开其中的 UnrealCV 命令行，然后直接加载
外部打包的 3DGS 地图：

```text
vset /action/game/level /Game/3dgs/custom_3dgs
```

加载后仍可继续使用 UnrealZoo 的智能体、相机、观测和交互 API。资产及内容
包要求见[外部 3DGS 流程](example/new_features/README.md#6-dynamic-3dgs-environment-load)。

<img src="doc/figs/new_features/3dgs_dynamic_load.gif" width="100%" alt="External 3DGS environment loaded with a supported UnrealZoo actor">

</details>

<details>
<summary><b>📹 视频数据采集</b></summary>

C++ 视频录制管线，支持大规模数据集高效采集

```bash
python example/DataRecording/VideoRecordingPipeline.py
```

**说明**: 录制前打开二进制文件，输入 `vget /unrealcv/status` 检查端口号，确保与代码中的 `port` 参数一致

<img src="doc/figs/Datarecord_tutorial.png" width="100%">

</details>

<details>
<summary><b>🎯 多智能体追踪</b></summary>

```bash
python example/tracking/basic/tracking_auto_basic.py \
  -e UnrealTrack-Greek_Island-ContinuousColor-v0
```


</details>

<details>
<summary><b>🧭 键盘导航（含交互）</b></summary>

```bash
python example/navigation/keyboard/navigation_keyboard_human.py \
  -e UnrealNavigation-Demo_Roof-MixedColor-v0
```

**控制说明**:
- `I/J/K/L` - 移动
- `↑/↓` - 抬头/低头
- `F` - 开门 | `H` - 上下车 | `E` - 拾取 | `Ctrl` - 下蹲 |`Space` 跳跃 | `Space x 2` 攀爬

<img src="doc/figs/navigation/map.png" width="48%"> <img src="doc/figs/navigation/target.png" width="48%">

</details>

<details>
<summary><b>🚁 异构空地协同</b></summary>

```bash
python example/multi_agent/HeterogeneousCooperation/Aerial-Ground-Cooperative.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0
```

3 地面智能体 + 1 无人机协同追踪

</details>

<details>
<summary><b>🎮 无人机键盘控制</b></summary>

```bash
python example/navigation/keyboard/navigation_keyboard_drone.py \
  -e UnrealNavigation-Demo_Roof-ContinuousColor-v0
```

**控制说明**:
- `W/S` - 前后 | `A/D` - 左右 | `E/Q` - 升降 | `J/L` - 偏航

</details>

---

## 🏗️ 技术架构

<img src="doc/figs/framework.png" width="100%" alt="UnrealZoo Framework">

### 架构说明

- **Unreal Engine Environments (Binary)**: 包含场景和可玩实体的当前 UE5.7 运行时环境包
- **UnrealCV+ Server**: 内置于 UE 二进制中的插件，包含渲染、数据捕获、对象/智能体控制、命令解析模块。我们优化了渲染管线和命令系统
- **UnrealCV+ Client**: 提供基于 Python 的工具函数，用于启动二进制、连接服务器、与 UE 环境交互。使用 IPC sockets 和批量命令优化性能
- **OpenAI Gym Interface**: 提供智能体级别的环境交互接口，支持通过配置文件自定义任务，包含环境增强、人口控制等 Gym Wrappers 工具集

### 数据流

```
用户算法 (Python) ←→ Gym Interface ←→ UnrealCV Client ←→ UnrealCV Server ←→ UE5.7 Environment
                                              (Socket/WebSocket)
```

---

## 🌍 场景画廊

<div align="center">

**UE5 示例场景**

<img src="doc/figs/UE5_ExampleScene/ChemicalFactory.png" width="32%">
<img src="doc/figs/UE5_ExampleScene/ModularOldTown.png" width="32%">
<img src="doc/figs/UE5_ExampleScene/MiddleEast.png" width="32%">

<img src="doc/figs/UE5_ExampleScene/Roof-City.png" width="32%">
<img src="doc/figs/UE4_ExampleScene/SuburbNeighborhood_Day.png" width="32%">
<img src="doc/figs/UE4_ExampleScene/Greek_island.png" width="32%">

**更多场景**: [Scene Gallery](http://unrealzoo.site/)

---

**🎮 UnrealZoo 自定义任务示例**  

<img src="doc/figs/interactiondemo_gif/navigation.gif" width="80%">  

**立体空间导航任务**

</div>

---

## 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **场景规模** | 16 km² | 最大单场景面积 |
| **场景数量** | 100+ | 预置照片级场景 |
| **智能体数量** | 10+ | 同场景实时交互 |
| **标准相机采集吞吐率** | 2K 53.46 FPS；4K 29.59 FPS | 相机采集吞吐率测试，最高加速 6.31× |
| **串行端到端采集** | 2K 22.40 FPS；4K 16.25 FPS | 完整串行采集测试，不代表完整 Gym step 速率 |
| **物理引擎** | Chaos | 当前 UE5.7 环境包的原生物理系统 |
| **环境包大小** | ~70 GB | 推荐的 v3.1 UE5.7 完整包 |
| **下载渠道** | ModelScope + Hugging Face | 主要环境包镜像 |

采集吞吐率与串行端到端采集测量的是观测流程中的不同范围，不能作为同一
指标直接比较。实际 Gym step 性能还会受到场景、启用模态、智能体数量和
硬件的影响。测试范围见 [v3.1 Changelog](doc/CHANGELOG_v3.1.md)。

---

## 📦 应用案例

- 🏆 **Offline EVT（ECCV 2024）** — 在 UnrealZoo 中训练与评估的 Offline-RL 具身视觉追踪方法。[论文](https://arxiv.org/abs/2404.09857) · [代码](https://github.com/wukui-muc/Offline_RL_Active_Tracking)
- 🚁 **UAV-Flow: Flying-on-a-Word** — 面向语言条件无人机模仿学习与仿真评估。[主页](https://prince687028.github.io/UAV-Flow/) · [论文](https://arxiv.org/abs/2505.15725)
- 🧠 **EmbRACE-3K** — 面向复杂环境语言引导具身推理的 3,000+ 任务数据集。[主页](https://mxllc.github.io/EmbRACE-3K/) · [论文](https://arxiv.org/abs/2507.10548)
- 🎯 **ROCKET-2** — 面向跨视角目标对齐与视觉运动策略的仿真训练。[主页](https://craftjarvis.github.io/ROCKET-2/) · [论文](https://arxiv.org/abs/2503.02505)
- 🛟 **RescueBench: Can Embodied Agents Save Lives in the Wild?** — 基于 UnrealZoo 构建的照片级、多阶段搜救具身智能评测基准。[论文](https://arxiv.org/abs/2606.01848) · [代码](https://github.com/UnrealZoo/RescueBench)

> 💡 **您的项目也使用了 UnrealZoo？** 欢迎提交 PR 添加到本列表！

---

## 📖 文档

| 文档                                  | 说明 |
|-------------------------------------|------|
| [用户指南](doc/user_guide_latest_v3.md) | 完整使用指南（v3.0） |
| [Wrapper 说明](doc/wrapper.md)        | 环境包装器 API |
| [添加环境](doc/addEnv.md)               | 自定义环境教程 |
| [CHANGELOG](doc/CHANGELOG_v4.md)    | 版本更新记录 |
| [v3.1 Changelog](doc/CHANGELOG_v3.1.md) | v3.1 功能边界与更新重点 |
| [示例索引](example/README.md)           | 所有示例代码 |

---

## 🗓️ TODO List

- [x] 发布完整环境合集包
- [x] 提供异构多智能体协作的 Gym 接口
- [x] 扩展可用的交互动作
- [x] 提供强化学习智能体的详细示例
- [x] 提供大型视觉语言模型的详细示例
- [x] 提供 MuJoCo Unitree Go1 集成与示例
- [ ] 融合 MetaHuman
- [ ] 融合 Unitree G1 人形机器人

---

## 🤝 贡献与支持

- 🌐 **官方网站**: [http://unrealzoo.site/](http://unrealzoo.site/)
- 📧 **联系邮箱**: [zfw1226@gmail.com, wukui@buaa.edu.cn]
- 💬 **讨论区**: [GitHub Discussions](https://github.com/UnrealZoo/unrealzoo-gym/discussions)

如果对你有帮助，请给我们一个 ⭐ Star！

---

## 📄 引用

如果 UnrealZoo 对你的研究有帮助，请引用：

```bibtex
@inproceedings{zhong2025unrealzoo,
  title={UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI},
  author={Zhong, Fangwei and Wu, Kui and Wang, Churan and Chen, Hao and Ci, Hai and Li, Zhoujun and Wang, Yizhou},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

---

## 📜 许可与致谢

本项目基于 [Apache 2.0](LICENSE) 协议开源。

致谢:
- [UnrealCV](https://unrealcv.org/) — UE-Python 通信桥梁
- [OpenAI Gym](https://gym.openai.com/) — RL 环境接口标准
- [Unreal Engine](https://www.unrealengine.com/) — 渲染引擎
- [Smart Locomotion](https://www.fab.com/zh-cn/listings/7f881534-bf40-493b-97b4-a917daa87af0) — 角色动画
- [Animal Pack](https://www.fab.com/zh-cn/listings/856c42d7-58a3-4b95-8f70-1302e5bdafa0) — 动物模型
- [Drivable Car](https://www.fab.com/zh-cn/listings/65a0844c-6be4-4e38-9d7a-b9697681a274) — 载具系统

---

<div align="center">

**[⬆ 回到顶部](#)**

Made with ❤️ by UnrealZoo Team

</div>
