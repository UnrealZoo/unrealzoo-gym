<div align="center">

<!-- Hero Banner -->
<a href="https://youtu.be/Xe2VmsJYTAU">
  <img src="https://img.youtube.com/vi/Xe2VmsJYTAU/maxresdefault.jpg" width="100%" alt="UnrealZoo Demo Video">
</a>
<p align="center"><i>▶️ Click to watch full demo video</i></p>

<!-- Title -->
<h1>UnrealZoo</h1>
<h3>Large-scale Photo-realistic Virtual Worlds for Embodied AI</h3>
<p>Photo-realistic virtual environments with 100+ scenes, 10+ agents real-time collaboration</p>

<!-- Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/UE-5.7-blue?style=flat-square&logo=unrealengine" />
  <img src="https://img.shields.io/badge/Python-3.8+-green?style=flat-square&logo=python" />
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
  🇺🇸 English | <a href="README_ch.md">🇨🇳 中文</a>
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

## 📖 Overview

<img src="doc/figs/overview.png" width="100%" alt="UnrealZoo Overview">

UnrealZoo is a rich collection of photo-realistic 3D virtual worlds built on Unreal Engine, designed to reflect the complexity and variability of open worlds. There are various playable entities for embodied AI, including human characters, robots, vehicles, and animals.

Integrated with [UnrealCV](https://unrealcv.org/), UnrealZoo provides a suite of easy-to-use Python APIs and tools for various potential applications, such as data annotation and collection, environment augmentation, distributed training, and benchmarking agents.

**💡 This repository provides the gym interface based on UnrealCV APIs for UE-based environments, which is compatible with OpenAI Gym and supports the high-level agent-environment interactions in UnrealZoo.**

---

## 🔥 What's New

> **UnrealZoo v4.0 expands the v3.0 foundation** with faster visual observation, 3-D perception, physics-driven robots, runtime agent customization, and externally packaged environments.

### 🚀 v4.0 Core Updates

| Feature | Status | Description |
|------|------|----------------------------------------|
| **Faster UnrealCV Capture** | ✅ Enhanced | Standard-camera capture reaches 96.20 FPS at 2K and 65.21 FPS at 4K; current benchmarks show up to 14.66× standard-camera and 4.96× panorama speedup |
| **LiDAR Observation** | ✅ Added | XYZI point-cloud observations with a player-controlled street-mapping example |
| **Occupancy Voxel Observation** | ✅ Added | LINGO-compatible boolean occupancy grids with bounds- and mesh-based modes |
| **MuJoCo Unitree Go1** | ✅ Added | Keyboard locomotion and an advanced Robot Parkour policy example |
| **Runtime Drone Visual Customization** | ✅ Added | Switch among five production-ready models with animated propellers, use a template appearance, or load a compatible external Static Mesh without respawning the drone |
| **Social Animation** | ✅ Added | Select newly packaged party, everyday, and in-car character animations at runtime |
| **External 3DGS Environments** | ✅ Added | Load user-packaged 3DGS assets and reuse UnrealZoo agents, cameras, and task APIs |

📄 [v4.0 Changelog](doc/CHANGELOG_v4.0.md) · 📚 [v4.0 Feature Guide](example/new_features/README.md)

<details>
<summary><b>🚀 v3.0 Core Updates</b></summary>

> **UnrealZoo v3.0 is released!** This is our biggest update yet, bringing complete heterogeneous multi-agent collaboration capabilities, out-of-the-box interaction systems, and a comprehensive upgrade to the UnrealCV+ Plugin.

| Feature | Status | Description |
|------|------|----------------------------------------|
| **Heterogeneous Multi-Agent Collaboration** | ✅ Released | Ground + UAV formation following (UE built-in navigation / Python external navigation API examples) |
| **Template-based Agent Spawn** | ✅ Released | Runtime dynamic agent generation with mixed category support |
| **Enhanced Interaction System** | ✅ Released | Door open / vehicle enter-exit / pickup / crouch / jump / climb, with API-keyboard mapping for easy understanding |
| **NavMesh Path Planning** | ✅ Released | Calculate shortest path waypoints via API, support autonomous agent navigation control and waypoint export |
| **VLN Baseline Examples** | ✅ Added | Integrates Uni-NaVid, ViNT/NoMaD, and StreamVLN with UnrealZoo navigation via a shared `env.step()` example. See [VLN Baseline README](example/VLN_Baseline/README.md) |

</details>

### 🔌 UnrealCV+ Plugin Upgrade

| Feature | Description |
|------|------------------------|
| **Faster Camera Capture** | Standard-camera capture reaches **96.20 FPS at 2K** and **65.21 FPS at 4K**; measured speedup reaches 14.66× for standard cameras and 4.96× for panoramas |
| **LiDAR Observation** | Provides XYZI point clouds for synchronized observation and mapping workflows |
| **Occupancy Voxel Observation** | Exposes LINGO-compatible boolean scene-occupancy grids with configurable profiles and voxelization modes |
| **Expanded Panorama Observation** | Extends panoramic capture modalities for embodied perception and dataset collection |
| **Rendering Performance Boost** | Image rendering speed improved by 120%, multi-agent scene FPS significantly enhanced |
| **PAK Runtime Mounting** | Dynamically extend content resources without rebuilding the project |
| **Panoramic Camera Support** | 360° equirectangular image/video export, supports VR preview |
| **C++ Video Recording Pipeline** | More efficient large-scale collection workflow |
| **Object Spawning from Path** | Spawn objects directly using full asset paths |
| **Scene Annotation System** | Supports semantic segmentation and object detection training workflows |
| **Stable CID Camera Identifier** | Long-term script configuration compatibility |

> 💡 **Solving User Pain Points**: Panoramic export, NavMesh path planning, UAV simulation, and the complete interaction system are supported out-of-the-box.

📄 [View Full Changelog](doc/CHANGELOG_v4.0.md) | 📚 [UnrealCV+ Documentation](https://docs.unrealcv.org/en/latest/unrealcv_plus/index.html) | 📚 [View Notion Technical Docs](https://unrealzoo.notion.site/)

### 📦 Environment Package Download

| Version                                                  | Content | Size  | Download |
|----------------------------------------------------------|------|-------|------|
| **Latest UE5.7 Full Package (Recommended)**              | UnrealZoo v4.0 full environment package | ~70GB | [ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/files), [Hugging Face](https://huggingface.co/datasets/UnrealZoo/UnrealZoo_UE5) |
| **UE5.6 Full Version**                                   | 100+ scenes, Chaos physics | ~70GB | [ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/tree/master/UnrealZoo_UE5_6_v3.0.0) , [HuggingFace](https://huggingface.co/datasets/UnrealZoo/UnrealZoo_UE5) |
| UE5 Demo Version (1.0 version, require v2.0 branch code) | 4 example scenes | ~10GB | [ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/files), [HuggingFace](https://huggingface.co/datasets/UnrealZoo/UnrealZoo_UE5) |
| UE4 Demo Version (1.0 version, require v2.0 branch code) | 6 example scenes | ~3GB  | [ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE4/files), [HuggingFace](https://huggingface.co/datasets/UnrealZoo/UnrealZoo_UE4) |

### 📜 Development History

<details>
<summary><b>Click to view historical updates</b></summary>

#### 2024-12: Paper Release
- 📝 Paper: [UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI](https://arxiv.org/abs/2412.20977)
- 🌐 Website: [unrealzoo.github.io](https://unrealzoo.github.io/)
- 📚 Notion Docs: [Scene Gallery](https://www.notion.so/Scene-Gallery-a801475ff98943159da66f641f4c38b2)

#### 2025-01: UE5.6 Full Environment Package
- ✅ 100+ scenes, 67GB full package
- ✅ Chaos physics system (vehicles, collisions, explosions, fire)
- ✅ Object interaction system (pickup/drop)
- ✅ Appearance switching system (player/animal categories merged)
- ✅ Cross-platform binary support (Win/Mac/Linux auto-configuration)
- ✅ ModelScope China mirror (high-speed download channel)

#### 2025-04: v3.0 Official Release
- ✅ Heterogeneous multi-agent collaboration
- ✅ Template-based Agent Spawn
- ✅ Full enhanced interaction system demo code (API-keyboard mapping)
- ✅ NavMesh path planning and task application demo code
- ✅ UnrealCV+ Plugin comprehensive upgrade

#### 2026: v4.0 Development Snapshot
- ✅ Faster UnrealCV visual observation and recording workflows
- ✅ LiDAR and occupancy voxel observations
- ✅ MuJoCo Go1 simulation examples
- ✅ User-packaged 3DGS environment support
- ✅ Runtime drone visual customization with animated built-in models and external Static Mesh support
- ✅ Expanded character social animations

</details>

[View Full Changelog](CHANGELOG.md)

[View v4.0 Changelog](doc/CHANGELOG_v4.0.md)

---

## ⚡ 30-Second Quick Start

```bash
# 1. Install
pip install -e .

# 2. Set environment path
export UnrealEnv=/path/to/UnrealEnv

# 3. Run multi-agent tracking demo
python example/multi_agent/baseline/multi_random_baseline.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0
```

> 💡 **Tip**: First run requires downloading UE5 environment package (67GB), we recommend using [ModelScope](https://modelscope.cn/datasets/UnrealZoo) China mirror for faster download

---

## 🌟 Core Features

<div align="center">

| 📡 **LiDAR Observation** | 🧊 **Occupancy Voxels** | 🐕 **MuJoCo Go1** | 🌐 **External 3DGS** |
|:---:|:---:|:---:|:---:|
| XYZI point-cloud observation | LINGO-compatible 3-D grids | Keyboard and policy control | Load user-packaged scenes |
| Pose-conditioned street mapping | Bounds and mesh modes | Unreal-rendered simulation | Reuse agents and task APIs |

| 🚁 **Drone Visual Customization** | 🎭 **Social Animation** | ⚡ **Faster UnrealCV Capture** |
|:---:|:---:|:---:|
| Five animated production-ready models plus a customization template | Select character social actions at runtime | Standard camera: 96.20 FPS at 2K, 65.21 FPS at 4K |
| External Static Mesh support preserves control, physics, camera, and task state | Party, everyday, and in-car animation groups | Up to 14.66× standard and 4.96× panorama speedup |

| 🏙️ **100+ Scenes** | 👥 **10+ Agents** | 🚗 **Vehicle Interaction** | 📦 **Object Manipulation** |
|:---:|:---:|:---:|:---:|
| Urban/Natural/Architectural/Industrial | Real-time collaboration in same scene | Enter/Drive/Exit | Pickup/Carry/Place |
| 16km² max scene size | Humanoid/Vehicle/Animal | Realistic vehicle animations | Spawn at arbitrary locations |

| ⚡ **UE5.6 Chaos** | 🎮 **Out of the Box** | 🐕 **Diverse Entities** | 🌐 **Cross-Platform** |
|:---:|:---:|:---:|:---:|
| Collision/Explosion/Fire | pip install and run | Humanoid/Vehicle/Animal | Linux/Win/Mac |
| Physics-level realism | No UE knowledge required | Real-time appearance switching | Pre-compiled binaries |

</div>

---

## 🎬 System Interaction Demo

<div align="center">

**🐕 MuJoCo Go1 Control & Parkour**

| Keyboard Control | Parkour: Third-Person View | Parkour: Depth Observation |
|:---:|:---:|:---:|
| <img src="doc/figs/new_features/mujoco_go1_keyboard.gif" width="100%" alt="MuJoCo Go1 keyboard control"> | <img src="doc/figs/new_features/mujoco_go1_parkour_third_person.gif" width="100%" alt="MuJoCo Go1 parkour third-person view"> | <img src="doc/figs/new_features/mujoco_go1_parkour_depth.gif" width="100%" alt="MuJoCo Go1 parkour depth observation"> |
| Basic `I/J/K/L` locomotion demo | Advanced policy behavior rendered from outside the robot | UnrealCV raw depth and the policy depth input |

| 🚁 Runtime Drone Visual Customization | 🎭 Character Social Animation | 🌐 External 3DGS + UnrealZoo Actor |
|:---:|:---:|:---:|
| <img src="doc/figs/new_features/drone_mesh_switch.gif" width="100%" alt="UnrealZoo runtime drone mesh switching"> | <img src="doc/figs/new_features/social_animation.gif" width="100%" alt="UnrealZoo character social animations"> | <img src="doc/figs/new_features/3dgs_dynamic_load.gif" width="100%" alt="External 3DGS environment loaded with a supported UnrealZoo actor"> |
| [Five animated built-in models plus external Static Mesh support](example/new_features/README.md#4-runtime-drone-visual-customization) | [`set_social_anim` case-sensitive argument list](example/new_features/README.md#5-character-social-animations) | Load an external 3DGS scene and reuse an UnrealZoo-supported actor |

**📡 LiDAR Street Mapping**

<img src="doc/figs/new_features/lidar_street_slam.gif" width="70%" alt="UnrealZoo Suburb LiDAR voxel mapping">

Player-controlled LiDAR observation with pose-conditioned map updates.

**🚗 Vehicle Interaction**

<div align="center">

| | | |
|:---:|:---:|:---:|
| ![](doc/figs/interactiondemo_gif/vehicle/car.gif) | ![](doc/figs/interactiondemo_gif/vehicle/motorbike.gif) | ![](doc/figs/interactiondemo_gif/vehicle/enter_exit_car.gif) |

</div>

**🤸 Actions & Interactions**

<table><tr>
<td><img src="doc/figs/interactiondemo_gif/action/squat.gif" height="180"></td>
<td><img src="doc/figs/interactiondemo_gif/action/kick.gif" height="180"></td>
<td><img src="doc/figs/interactiondemo_gif/action/climb.gif" height="180"></td>
<td><img src="doc/figs/interactiondemo_gif/action/climb_ladder.gif" height="180"></td>
<td><img src="doc/figs/interactiondemo_gif/action/climb_woman.gif" height="180"></td>
</tr></table>

**🤖 Diverse Controllable Agents**

<div align="center">

| Drone | Robot Dog | Multi-Agent Collaboration |
|:---:|:---:|:---:|
| ![](doc/figs/interactiondemo_gif/drone.gif) | ![](doc/figs/interactiondemo_gif/mobilerobot.gif) | ![](doc/figs/interactiondemo_gif/cooperation/cooperation.gif) |

</div>

</div>

---

## 🎮 Interaction Demo (Example Code)

<details open>
<summary><b>✨ v4.0 Feature Demos</b></summary>

Run these commands from the repository root in Windows CMD. They use the
registered environment configuration and resolve its binary from the existing
`UnrealEnv` path:

```cmd
python example\new_features\suburb_street_slam.py
python example\new_features\occupancy_voxel_demo.py --profile lingo_vis --method mesh
python example\new_features\drone_mesh_switch_demo.py --interval 2 --cycles 2 --render
```

The two Go1 examples connect to a binary that you start
manually. Wait for the map and UnrealCV server to finish loading, then use the
matching port:

```cmd
python example\mujoco\go1_keyboard_control.py --host 127.0.0.1 --port 9000
python example\mujoco\go1_parkour.py --host 127.0.0.1 --port 9000 --command-mode keyboard
```

Both Go1 examples use `I/K` for forward/backward and `J/L` for turning. The
advanced example adapts the official [Robot Parkour Learning repository](https://github.com/ZiwenZhuang/parkour);
source and citation details are recorded in the [MuJoCo example guide](example/mujoco/README.md#advanced-policy-source-and-citation).

**Social animation:** See the [`set_social_anim` API and case-sensitive
argument list](example/new_features/README.md#5-character-social-animations).

See the [v4.0 feature guide](example/new_features/README.md) for arguments,
controls, asset requirements, and observation formats.

</details>

<details>
<summary><b>🌐 Dynamic 3DGS Environment Loading</b></summary>

Start the UnrealZoo v4.0 binary, open its UnrealCV command console, and load
the externally packaged 3DGS level directly:

```text
vset /action/game/level /Game/3dgs/custom_3dgs
```

The loaded level continues to use UnrealZoo agents, cameras, observations, and
interaction APIs. See the [external 3DGS package workflow](example/new_features/README.md#6-dynamic-3dgs-environment-load)
for the asset and package requirements.

<img src="doc/figs/new_features/3dgs_dynamic_load.gif" width="100%" alt="External 3DGS environment loaded with a supported UnrealZoo actor">

</details>

<details>
<summary><b>📹 Video Data Recording</b></summary>

C++ video recording pipeline for efficient large-scale dataset collection

```bash
python example/DataRecording/VideoRecordingPipeline.py
```

**Note**: Before recording, open the binary and type `vget /unrealcv/status` to check the port number, ensure it matches the `port` parameter in the code

<img src="doc/figs/Datarecord_tutorial.png" width="100%">

</details>

<details>
<summary><b>🎯 Multi-Agent Tracking</b></summary>

```bash
python example/tracking/basic/tracking_auto_basic.py \
  -e UnrealTrack-Greek_Island-ContinuousColor-v0
```


</details>

<details>
<summary><b>🧭 Keyboard Navigation (with Interactions)</b></summary>

```bash
python example/navigation/keyboard/navigation_keyboard_human.py \
  -e UnrealNavigation-Demo_Roof-MixedColor-v0
```

**Controls**:
- `I/J/K/L` - Move
- `↑/↓` - Look up/down
- `F` - Open door | `H` - Enter/Exit vehicle | `E` - Pickup | `Ctrl` - Crouch | `Space` - Jump | `Space x 2` - Climb

<img src="doc/figs/navigation/map.png" width="48%"> <img src="doc/figs/navigation/target.png" width="48%">

</details>

<details>
<summary><b>🚁 Heterogeneous Air-Ground Collaboration</b></summary>

```bash
python example/multi_agent/HeterogeneousCooperation/Aerial-Ground-Cooperative.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0
```

3 ground agents + 1 UAV collaborative tracking

</details>

<details>
<summary><b>🎮 Drone Keyboard Control</b></summary>

```bash
python example/navigation/keyboard/navigation_keyboard_drone.py \
  -e UnrealNavigation-Demo_Roof-ContinuousColor-v0
```

**Controls**:
- `W/S` - Forward/Back | `A/D` - Left/Right | `E/Q` - Ascend/Descend | `J/L` - Yaw

</details>

---

## 🏗️ Technical Architecture

<img src="doc/figs/framework.png" width="100%" alt="UnrealZoo Framework">

### Architecture Overview

- **Unreal Engine Environments (Binary)**: UE5.6 runtime environment containing scenes and playable entities
- **UnrealCV+ Server**: Plugin built into UE binary, including rendering, data capture, object/agent control, command parsing modules. We optimized the rendering pipeline and command system
- **UnrealCV+ Client**: Python-based utility functions for launching binaries, connecting to servers, and interacting with UE environments. Uses IPC sockets and batch commands for performance optimization
- **OpenAI Gym Interface**: Agent-level environment interaction interface, supports task customization via configuration files, includes Gym Wrappers toolkit for environment augmentation and population control

### Data Flow

```
User Algorithm (Python) ←→ Gym Interface ←→ UnrealCV Client ←→ UnrealCV Server ←→ UE5.6 Environment
                                              (Socket/WebSocket)
```

---

## 🌍 Scene Gallery

<div align="center">

**UE5 Example Scenes**

<img src="doc/figs/UE5_ExampleScene/ChemicalFactory.png" width="32%">
<img src="doc/figs/UE5_ExampleScene/ModularOldTown.png" width="32%">
<img src="doc/figs/UE5_ExampleScene/MiddleEast.png" width="32%">

<img src="doc/figs/UE5_ExampleScene/Roof-City.png" width="32%">
<img src="doc/figs/UE4_ExampleScene/SuburbNeighborhood_Day.png" width="32%">
<img src="doc/figs/UE4_ExampleScene/Greek_island.png" width="32%">

**More Scenes**: [Scene Gallery](http://unrealzoo.site/)

---

**🎮 UnrealZoo Custom Task Example**  

<img src="doc/figs/interactiondemo_gif/navigation.gif" width="80%">  

**3D Spatial Navigation Task**

</div>

---

## 📊 Performance Metrics

| Metric | Value | Description |
|------|------|------|
| **Scene Scale** | 16 km² | Maximum single scene area |
| **Scene Count** | 100+ | Pre-built photo-realistic scenes |
| **Agent Count** | 10+ | Real-time interaction in same scene |
| **Rendering Performance** | 60+ FPS | Real-time multi-modal rendering |
| **Physics Engine** | Chaos | UE5.6 native physics |
| **Package Size** | 67 GB | UE5 full version |
| **Download Channels** | GitHub + ModelScope | China acceleration mirror |

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
git clone https://github.com/UnrealZoo/unrealzoo-gym.git
cd unrealzoo-gym
pip install -e .
```

### Step 2: Download Environment Package

| Package                                                 | Download Link | Size  |
|---------------------------------------------------------|----------|-------|
| **Latest UE5.7 Full Package (Recommended)**             | [🤖 ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/files), [🤗 Hugging Face](https://huggingface.co/datasets/UnrealZoo/UnrealZoo_UE5) | ~70GB |
| **UE5.6 Full Version**                                  | [🤖 ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/tree/master/UnrealZoo_UE5_6_v3.0.0) | ~70GB |
| UE5 Demo Scenes (1.0 version, require v2.0 branch code) | [🤖 ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE5/files) | ~10GB |
| UE4 Demo Scenes (1.0 version, require v2.0 branch code) | [🤖 ModelScope](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE4/files) | ~3GB  |

Extract to `UnrealEnv` directory:

```bash
export UnrealEnv=/path/to/UnrealEnv
```

### Step 3: Run Demos

```bash
# Multi-agent random policy
python example/multi_agent/baseline/multi_random_baseline.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0

# Keyboard control navigation
python example/navigation/keyboard/navigation_keyboard_human.py \
  -e UnrealNavigation-Demo_Roof-MixedColor-v0
```

> 💡 **Tip**: If mouse disappears, press `` ` `` (above Tab) to release mouse

---

## 📦 Applications

<div align="center">

### 🏆 Offline EVT (ECCV 2024)

**Embodied Visual Tracking Agent**

Active tracking algorithm based on Offline RL, trained and validated in UnrealZoo environments

[📄 Paper](https://arxiv.org/abs/2404.09857) • [💻 Code](https://github.com/wukui-muc/Offline_RL_Active_Tracking)


---

### 🚁 UAV-Flow: Flying-on-a-Word

**Language-Guided UAV Control**

Beihang University team uses UnrealZoo for simulation evaluation, supporting language-conditioned UAV imitation learning

[🌐 Homepage](https://prince687028.github.io/UAV-Flow/) • [📄 Paper](https://arxiv.org/abs/2505.15725)

---

### 🧠 EmbRACE-3K: Embodied Reasoning

**Embodied Reasoning in Complex Environments**

HKU, Tsinghua, and BNU jointly built 3,000+ language-guided task dataset based on UnrealCV-Zoo framework

[🌐 Homepage](https://mxllc.github.io/EmbRACE-3K/) • [📄 Paper](https://arxiv.org/abs/2507.10548)

---

### 🎯 ROCKET-2: Cross-View Goal Alignment

**Cross-View Goal Alignment**

Peking University and UCLA team uses UnrealZoo for cross-view visuomotor policy simulation training

[🌐 Homepage](https://craftjarvis.github.io/ROCKET-2/) • [📄 Paper](https://arxiv.org/abs/2503.02505)

> 💡 **Your project also uses UnrealZoo?** Welcome to submit a PR to add to this list!

</div>

---

## 📖 Documentation

| Document                                  | Description |
|-------------------------------------------|------|
| [User Guide](doc/user_guide_latest_v3.md) | Complete usage guide (v3.0) |
| [Wrapper Guide](doc/wrapper.md)           | Environment wrapper APIs |
| [Add Environment](doc/addEnv.md)          | Custom environment tutorial |
| [CHANGELOG](doc/CHANGELOG_v4.md)          | Version update history |
| [v4.0 Changelog](doc/CHANGELOG_v4.0.md)   | v4.0 feature boundary and release highlights |
| [Example Index](example/README.md)        | All example code |

---

## 🗓️ TODO List

- ✅ Release an all-in-one package of the collected environments
- ✅ Add gym interface for heterogeneous multi-agent co-operation
- ✅ Expand the list of supported interactive actions
- [ ] Add more detailed examples for reinforcement learning agents
- [ ] Add more detailed examples for large vision-language models
- [ ] Add MuJoCo Unitree G1 integration and examples.

---

## 🤝 Contributing & Support

- 🌐 **Official Website**: [http://unrealzoo.site/](http://unrealzoo.site/)
- 📧 **Contact Email**: [[zfw1226@gmail.com, wukui@buaa.edu.cn]]
- 💬 **Discussions**: [GitHub Discussions](https://github.com/UnrealZoo/unrealzoo-gym/discussions)

If you find this helpful, please give us a ⭐ Star!

---

## 📄 Citation

If UnrealZoo helps your research, please cite our ICCV 2025 paper:

```bibtex
@inproceedings{zhong2025unrealzoo,
  title={UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI},
  author={Zhong, Fangwei and Wu, Kui and Wang, Churan and Chen, Hao and Ci, Hai and Li, Zhoujun and Wang, Yizhou},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

---

## 📜 License & Acknowledgments

This project is open-sourced under [Apache 2.0](LICENSE) license.

Acknowledgments:
- [UnrealCV](https://unrealcv.org/) — UE-Python communication bridge
- [OpenAI Gym](https://gym.openai.com/) — RL environment interface standard
- [Unreal Engine](https://www.unrealengine.com/) — Rendering engine
- [Smart Locomotion](https://www.fab.com/zh-cn/listings/7f881534-bf40-493b-97b4-a917daa87af0) — Character animation
- [Animal Pack](https://www.fab.com/zh-cn/listings/856c42d7-58a3-4b95-8f70-1302e5bdafa0) — Animal models
- [Drivable Car](https://www.fab.com/zh-cn/listings/65a0844c-6be4-4e38-9d7a-b9697681a274) — Vehicle system

---

<div align="center">

**[⬆ Back to Top](#)**

Made with ❤️ by UnrealZoo Team

</div>
