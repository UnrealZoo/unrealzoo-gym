# UnrealZoo 用户指南（v3.0）

**版本**: v3.0-stage  
**最后更新**: 2026-04-13  
**对应代码**: [CHANGELOG.md](CHANGELOG.md)

---

## 1. 什么是 UnrealZoo？

UnrealZoo 是一个基于 **Unreal Engine 5** 的强化学习仿真环境，兼容 **OpenAI Gym** 接口。

### 核心特性

| 特性 | 说明 |
|------|------|
| **高保真渲染** | UE5 的 Nanite + Lumen 提供照片级视觉质量 |
| **多智能体支持** | 同构/异构 agent 混合，支持空地协同 |
| **丰富任务类型** | Tracking、Navigation、Rescue、Rendezvous 等 |
| **键盘交互** | 人类玩家可直接介入，支持人机协作 |
| **跨平台** | Linux / Windows / macOS 三平台支持 |

### 架构概览

```
你的算法 (Python)  ←→  Gym API  ←→  UnrealCV  ←→  UE5 仿真环境
                                          (TCP/UDS)
```

---

## 2. 5分钟跑通（最短路径）

> **本节目标**: 验证安装成功，看到第一个仿真画面

### 2.1 设置二进制根目录

```bash
export UnrealEnv=/your/path/to/UnrealEnv
```

### 2.2 安装项目

```bash
git clone https://github.com/UnrealZoo/unrealzoo-gym.git
cd unrealzoo-gym
pip install -e .
```

### 2.3 运行 baseline

```bash
python example/multi_agent/baseline/multi_random_baseline.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0
```

看到 UE5 窗口弹出，agent 开始随机移动 → **安装成功** ✓

### 2.4 试试键盘控制

```bash
python example/navigation/keyboard/navigation_keyboard_human.py \
  -e UnrealNavigation-SuburbNeighborhood_Day-MixedColor-v0
```

按 `I/J/K/L` 移动，`F` 开门，`E` 拾取 → **交互验证成功** ✓

> 若启动失败，检查 `UnrealEnv` 路径和二进制执行权限（Linux: `chmod +x`）

---

## 3. 环境准备（详细）

> **本节目标**: 完整理解安装流程和配置选项

### 3.1 获取 Unreal 二进制

地图二进制文件可从以下渠道下载：

| 平台 | 下载链接 |
|------|----------|
| ModelScope | [待补充] |
| 百度网盘 | [待补充] |

下载后解压到本机路径，例如 `/media/wuk/T9/UnrealEnv/`。

### 3.2 设置环境变量

```bash
export UnrealEnv=/your/path/to/UnrealEnv
```

建议在 shell 配置文件（如 `.bashrc`）中长期保存：

```bash
echo 'export UnrealEnv=/your/path/to/UnrealEnv' >> ~/.bashrc
source ~/.bashrc
```

> **备选方式**: 某些 demo 脚本开头可手动指定路径：
> ```python
> import os
> os.environ['UnrealEnv'] = '/your/path/to/UnrealEnv'
> ```

### 3.3 环境 ID 命名规则

统一格式：

```
Unreal{Task}-{MapName}-{ActionType}{ObservationType}-v{reset}
```

**示例**:
- `UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0`
- `UnrealNavigation-Demo_Roof-MixedColor-v0`

**字段说明**:

| 字段 | 可选值 | 说明 |
|------|--------|------|
| `Task` | Track / Navigation / NavigationMulti / Rendezvous / Rescue | 任务类型 |
| `ActionType` | Discrete / Continuous / Mixed | 动作空间类型 |
| `ObservationType` | Color / Depth / Rgbd / Gray / CG / Mask / Pose / ... | 观测模态 |
| `reset` | v0-v6 | 重置变体，**默认 v0** |

> ⚠️ **reset 字段注意**:
> - 仅 `FlexibleRoom` 支持 `v1-v6`（随机化背景和障碍物）
> - 其他地图统一使用 `v0`

---

## 4. 运行示例（按任务分类）

> **本节目标**: 了解不同任务的运行方式，找到适合你研究的入口

### 4.1 Tracking（目标追踪）

Agent 追踪移动目标（如动物、其他 agent）。

```bash
# 自动追踪（基础）
python example/tracking/basic/tracking_auto_basic.py \
  -e UnrealTrack-Greek_Island-ContinuousColor-v0

# NavMesh 自动目标 + PID 跟随
python example/tracking/navmesh/tracking_auto_navmesh.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0

# 键盘控制追踪动物
python example/tracking/basic/tracking_keyboard_animal.py \
  -e UnrealTrack-Old_Town-MixedColor-v0
```

### 4.2 Navigation（导航）

Agent 在环境中导航到目标点，支持交互（开门、拾取、上下车）。

```bash
# 人类键盘导航（含交互动作）
python example/navigation/keyboard/navigation_keyboard_human.py \
  -e UnrealNavigation-SuburbNeighborhood_Day-MixedColor-v0

# 无人机键盘导航（4D 连续控制）
python example/navigation/keyboard/navigation_keyboard_drone.py \
  -e UnrealNavigation-Demo_Roof-ContinuousColor-v0
```

### 4.3 Multi-Agent（多智能体）

多 agent 协作或独立任务。

```bash
# 同构多智能体随机策略 baseline
python example/multi_agent/baseline/multi_random_baseline.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0

# 键盘 + 随机策略混合
python example/multi_agent/baseline/multi_keyboard_plus_random.py \
  -e UnrealTrack-Map_ChemicalPlant_1-MixedColor-v0

# 异构空地协同（3地面 + 1无人机）
python example/multi_agent/HeterogeneousCooperation/Aerial-Ground-Cooperative.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0
```

---

## 5. 交互控制详解

> **本节目标**: 掌握人类玩家的控制方式

### 5.1 人类导航（Mixed Action）

适用于 `player` 类别，支持移动 + 交互。

#### 基础移动

| 按键 | 功能 | 说明 |
|------|------|------|
| `I` | 前进 | 向面朝方向移动 |
| `K` | 后退 | 向背对方向移动 |
| `J` | 左移 | 向左侧平移 |
| `L` | 右移 | 向右侧平移 |
| `↑` (上箭头) | 抬头 | 增加俯仰角 |
| `↓` (下箭头) | 低头 | 减小俯仰角 |

#### 交互动作

| 按键 | 功能 | 触发条件 | 底层 API |
|------|------|----------|----------|
| `F` | 开门 | 门 **1.5m** 范围内 | `vbp {player} set_open_door 1` |
| `H` | 上下车 | 靠近载具 | `vbp {obj} enter_exit_car {player_index}` |
| `Ctrl` | 下蹲 | - | `vbp {player} set_crouch` |
| `E` | 拾取 | 物品 **1.5m** 范围内 | `vbp {player} set_pickup` |
| `Space` | 跳跃 | - | `vbp {player} set_jump` |

> ⚠️ **交互要点**: 靠近交互物（1.5m 内）后**按对应键触发**，不是自动触发。注意调整头部视角确保物体在视野内。

#### 自定义键盘映射

交互动作通过 `animation_dict` 映射到 API，你可以在代码中自定义按键：

```python
# 示例：修改 navigation_keyboard_human.py 中的按键映射
if event.key == pygame.K_f:  # 原 F 键开门
    action['animation'] = 'open_door'

# 改为 X 键开门
if event.key == pygame.K_x:
    action['animation'] = 'open_door'
```

可用的 `animation` 动作名称：
- `'open_door'` → `vbp {player} set_open_door {state}`
- `'enter_vehicle'` → `vbp {obj} enter_exit_car {player_index}`
- `'pickup'` → `vbp {player} set_pickup`
- `'crouch'` → `vbp {player} set_crouch`
- `'jump'` → `vbp {player} set_jump`
- `'stand'` → 站立状态
- `'liedown'` → 躺下
- `'drop'` → 丢弃

完整映射定义在: `gym_unrealcv/envs/agent/character.py` 的 `animation_dict`

### 5.2 无人机导航（Continuous Action）

适用于 `drone` 类别，4 自由度连续控制。

| 按键 | 功能 | 控制维度 |
|------|------|----------|
| `W` | 前进 | `vx` (前后速度) |
| `S` | 后退 | `vx` (前后速度) |
| `A` | 左移 | `vy` (左右速度) |
| `D` | 右移 | `vy` (左右速度) |
| `E` | 上升 | `vz` (垂直速度) |
| `Q` | 下降 | `vz` (垂直速度) |
| `J` | 左转 | `vyaw` (偏航角速度) |
| `L` | 右转 | `vyaw` (偏航角速度) |

---

## 6. Wrappers 速查

> **本节目标**: 理解示例代码中的 wrapper 作用，学会自定义环境行为

Wrappers 用于**在不修改环境源码的情况下**，动态调整环境行为。

### 6.1 ConfigUEWrapper — 启动参数配置

控制 Unreal 二进制启动方式。

```python
from gym_unrealcv.envs.wrappers import configUE

env = configUE.ConfigUEWrapper(
    env,
    resolution=(160, 160),    # 渲染分辨率 (宽, 高)
    gpu_id=0,                 # 指定 GPU ID
    offscreen=False,          # 是否离屏渲染（无窗口）
    display=None,             # 指定显示器（Linux, 如 ':0.0'）
    use_opengl=False,         # 是否使用 OpenGL（默认 Vulkan）
    nullrhi=False,            # 是否禁用渲染（纯逻辑仿真）
    render_quality=7,         # 渲染质量 0-7（7 为最高）
    use_lumen=False,          # 是否启用 Lumen 全局光照
    sleep_time=5,             # 启动后等待时间（秒）
    comm_mode='tcp'           # 通信模式: 'tcp' / 'websocket'
)
```

**典型场景**:
- 多 GPU 训练: `gpu_id=0,1,2...`
- 服务器部署: `offscreen=True`
- 快速验证逻辑: `nullrhi=True`（不渲染，速度最快）
- 视觉质量优先: `render_quality=7, use_lumen=True`
- 性能优先: `render_quality=3, use_lumen=False`

### 6.2 RandomPopulationWrapper — 动态人口

随机设置每回合的 agent 数量。

```python
from gym_unrealcv.envs.wrappers import augmentation

# 固定 4 个 agent
env = augmentation.RandomPopulationWrapper(env, 4, 4, random_target=False)

# 每回合随机 5-10 个 agent
env = augmentation.RandomPopulationWrapper(env, 5, 10, random_target=False)
```

**典型场景**: 训练策略的泛化能力，模拟不同密度的人群。

### 6.3 TimeDilationWrapper — 仿真速度

控制 UE 仿真速度。

```python
from gym_unrealcv.envs.wrappers import time_dilation

# 目标 30 FPS，每 60 步更新一次
env = time_dilation.TimeDilationWrapper(env, reference_fps=30, update_steps=60)
```

**典型场景**: 与外部系统同步，或加速/减速仿真。

> **完整 Wrapper API**: 见 `doc/wrapper.md`

---

## 7. Agent Spawn 机制（v3.0 核心变化）

> **本节目标**: 理解 v3.0 的模板化 spawn 机制，能够自定义 agent 配置

### 7.1 新方案 vs 旧方案

| 维度 | 旧方案 | 新方案（v3.0） |
|------|--------|----------------|
| 配置方式 | 固定 actor 名称 + 固定数量 | 类别模板 + 运行时动态生成 |
| 扩展性 | 扩展混合类别成本高 | 数量、类别组合、实验配置解耦 |
| 多任务复用 | 每任务需单独 JSON | 同一地图 JSON 可复用 |

### 7.2 模板化配置示例

**旧方案**（固定实例名）：
```json
"agents": {
  "player_EP0_0": { "class_name": "bp_character_C" },
  "player_EP0_1": { "class_name": "bp_character_C" }
}
```

**新方案**（按类别模板）：
```json
"agents": {
  "player": {
    "class_name": ["bp_character_C"],
    "asset_path": ["/Game/.../BP_Character.BP_Character_C"],
    "move_action": [[0, 100], [0, -100], [15, 50], [-15, 50], [30, 0], [-30, 0], [0, 0]]
  },
  "drone": {
    "class_name": ["BP_drone01_C"],
    "asset_path": ["/Game/.../BP_drone01.BP_drone01_C"],
    "move_action": [[0.5, 0, 0, 0], [-0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, -0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, -0.5, 0], [0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]]
  }
}
```

### 7.3 脚本侧控制

```python
# 关键：在 reset() 之前设置 agents_category
env.unwrapped.agents_category = ["player", "player", "player", "drone"]
env = augmentation.RandomPopulationWrapper(env, 4, 4, random_target=False)
obs = env.reset()
```

### 7.4 混合类别初始化规则

| `agents_category` 长度 | 行为 |
|------------------------|------|
| 1 | 全体使用同一类别（如 `["player"]` 表示全是 player） |
| 等于 population | 按 slot 逐个初始化 |
| 其他（如 2 ≤ len < population） | **回退到第一个类别并告警** |

> ⚠️ **关键时序**: `agents_category` 必须在 `reset()` 之前设置，否则不生效！

### 7.5 迁移收益

- 解耦地图配置与实验参数
- 支持动态 population 变化
- 支持异构 agent 混合

---

## 8. 配置与地图 JSON

> **本节目标**: 理解 JSON 配置结构，能够自定义地图和任务

### 8.1 平台自动切换

生成 JSON 时写入三平台字段，**运行时自动选择**：

```json
{
  "env_bin": "Linux 路径",
  "env_bin_win": "Windows 路径",
  "env_bin_mac": "macOS 路径"
}
```

### 8.2 完整 JSON 示例

```json
{
  "env_bin": "UnrealZoo_UE5_6_Linux_v2.0.2/Linux/UnrealZoo_UE5_6/Binaries/Linux/UnrealZoo_UE5_6",
  "env_bin_win": "UnrealZoo_UE5_6_Win64_v2.0.2/UnrealZoo_UE5_6/Binaries/Win64/UnrealZoo_UE5_6.exe",
  "env_bin_mac": "UnrealZoo_UE5_6_Mac/UnrealZoo_UE5_6/Binaries/Mac/UnrealZoo_UE5_6.app/Contents/MacOS/UnrealZoo_UE5_6",
  "agents": {
    "player": {
      "class_name": ["bp_character_C"],
      "asset_path": ["/Game/SmartLocomotion/Blueprints/BP_Character.BP_Character_C"],
      "internal_nav": true,
      "move_action": [[0, 100], [0, -100], [15, 50], [-15, 50], [30, 0], [-30, 0], [0, 0]],
      "move_action_continuous": { "high": [30, 100], "low": [-30, -100] },
      "head_action": [[0, 0, 0], [0, 30, 0], [0, -30, 0]],
      "animation_action": ["stand", "jump", "crouch", "open_door", "enter_vehicle", "pickup"]
    }
  },
  "targets": { "Point": [] }
}
```

### 8.3 导航目标配置

```json
"targets": { "Point": [] }
```

- 允许为空列表，但会触发 warning
- 目标为空时 reward 始终为 0
- 可动态设置：`env.unwrapped.set_navigation_targets(["TargetName"])`

---

## 9. 常见问题



### Q1: 二进制无法启动

- 确认 `UnrealEnv` 路径正确
- Linux: `chmod +x /path/to/binary`
- 确认平台匹配（不要混用 Linux 二进制在 Windows 上运行）


## 2. 文档索引

| 文档 | 内容 |
|------|------|
| `example/README.md` | Demo 完整索引 |
| `doc/wrapper.md` | 全部 Wrapper API |
| `doc/addEnv.md` | 添加自定义环境 |
| `CHANGELOG.md` | 版本变更记录 |

---

## 附录 A: `move_action` 字段详解

### Player / Animal / Car / Motorbike

2 元素数组: `[angular, velocity]`
- `angular`: 转向角速度（度/秒）
- `velocity`: 线速度（cm/秒）

```json
"move_action": [
  [0, 100],    // 直行前进
  [0, -100],   // 直行后退
  [15, 50],    // 左转前进
  [-15, 50],   // 右转前进
  [30, 0],     // 原地左转
  [-30, 0],    // 原地右转
  [0, 0]       // 停止
]
```

### Drone

4 元素数组: `[vx, vy, vz, vyaw]`
- `vx`: 前后速度
- `vy`: 左右速度
- `vz`: 垂直速度（升降）
- `vyaw`: 偏航角速度

```json
"move_action": [
  [0.5, 0, 0, 0],    // 前进
  [-0.5, 0, 0, 0],   // 后退
  [0, 0.5, 0, 0],    // 右移
  [0, -0.5, 0, 0],   // 左移
  [0, 0, 0.5, 0],    // 上升
  [0, 0, -0.5, 0],   // 下降
  [0, 0, 0, 1],      // 左转
  [0, 0, 0, -1],     // 右转
  [0, 0, 0, 0]       // 悬停
]
```

完整模板定义参考: `generate_env_config.py`
