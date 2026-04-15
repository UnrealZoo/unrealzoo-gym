# UnrealZoo User Guide (v3.0)

**Version**: v3.0-stage  
**Last Updated**: 2026-04-15  
**Code Reference**: [CHANGELOG.md](CHANGELOG.md)

---

## 1. What is UnrealZoo?

UnrealZoo is a reinforcement learning simulation environment based on **Unreal Engine 5**, compatible with the **OpenAI Gym** interface.

### Core Features

| Feature | Description |
|---------|-------------|
| **High-Fidelity Rendering** | UE5's Nanite + Lumen provide photorealistic visual quality |
| **Multi-Agent Support** | Homogeneous/heterogeneous agent mixing, supporting air-ground collaboration |
| **Rich Task Types** | Tracking, Navigation, Rescue, Rendezvous, etc. |
| **Keyboard Interaction** | Human players can directly intervene, supporting human-AI collaboration |
| **Cross-Platform** | Linux / Windows / macOS support |

### Architecture Overview

```
Your Algorithm (Python)  ←→  Gym API  ←→  UnrealCV  ←→  UE5 Simulation Environment
                                              (TCP/UDS)
```

---

## 2. Get Started in 5 Minutes (Shortest Path)

> **Goal of this section**: Verify successful installation and see your first simulation

### 2.1 Set Binary Root Directory

```bash
export UnrealEnv=/your/path/to/UnrealEnv
```

### 2.2 Install the Project

```bash
git clone https://github.com/UnrealZoo/unrealzoo-gym.git
cd unrealzoo-gym
pip install -e .
```

### 2.3 Run Baseline

```bash
python example/multi_agent/baseline/multi_random_baseline.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0
```

When the UE5 window pops up and agents start moving randomly → **Installation successful** ✓

### 2.4 Try Keyboard Control

```bash
python example/navigation/keyboard/navigation_keyboard_human.py \
  -e UnrealNavigation-SuburbNeighborhood_Day-MixedColor-v0
```

Press `I/J/K/L` to move, `F` to open doors, `E` to pick up → **Interaction verified** ✓

> If startup fails, check the `UnrealEnv` path and binary execution permissions (Linux: `chmod +x`)

---

## 3. Environment Setup (Detailed)

> **Goal of this section**: Fully understand the installation process and configuration options

### 3.1 Get Unreal Binaries

Map binary files can be downloaded from the following sources:

| Platform | Download Link |
|----------|---------------|
| ModelScope | [To be added] |
| Baidu Netdisk | [To be added] |

After downloading, extract to your local path, e.g., `/media/wuk/T9/UnrealEnv/`.

### 3.2 Set Environment Variable

```bash
export UnrealEnv=/your/path/to/UnrealEnv
```

Recommended to save permanently in your shell config (e.g., `.bashrc`):

```bash
echo 'export UnrealEnv=/your/path/to/UnrealEnv' >> ~/.bashrc
source ~/.bashrc
```

> **Alternative**: Some demo scripts allow manual path specification at the beginning:
> ```python
> import os
> os.environ['UnrealEnv'] = '/your/path/to/UnrealEnv'
> ```

### 3.3 Environment ID Naming Convention

Unified format:

```
Unreal{Task}-{MapName}-{ActionType}{ObservationType}-v{reset}
```

**Examples**:
- `UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0`
- `UnrealNavigation-Demo_Roof-MixedColor-v0`

**Field Descriptions**:

| Field | Options | Description |
|-------|---------|-------------|
| `Task` | Track / Navigation / NavigationMulti / Rendezvous / Rescue | Task type |
| `ActionType` | Discrete / Continuous / Mixed | Action space type |
| `ObservationType` | Color / Depth / Rgbd / Gray / CG / Mask / Pose / ... | Observation modality |
| `reset` | v0-v6 | Reset variant, **default v0** |

> ⚠️ **Note on reset field**:
> - Only `FlexibleRoom` supports `v1-v6` (randomized backgrounds and obstacles)
> - Other maps use `v0` uniformly

---

## 4. Running Examples (By Task Category)

> **Goal of this section**: Understand how to run different tasks and find the entry point for your research

### 4.1 Tracking (Target Tracking)

Agents track moving targets (e.g., animals, other agents).

```bash
# Automatic tracking (basic)
python example/tracking/basic/tracking_auto_basic.py \
  -e UnrealTrack-Greek_Island-ContinuousColor-v0

# NavMesh auto target + PID following
python example/tracking/navmesh/tracking_auto_navmesh.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0

# Keyboard control tracking animals
python example/tracking/basic/tracking_keyboard_animal.py \
  -e UnrealTrack-Old_Town-MixedColor-v0
```

### 4.2 Navigation (Navigation)

Agents navigate to target points in the environment, supporting interactions (open doors, pick up, enter/exit vehicles).

```bash
# Human keyboard navigation (with interaction actions)
python example/navigation/keyboard/navigation_keyboard_human.py \
  -e UnrealNavigation-SuburbNeighborhood_Day-MixedColor-v0

# Drone keyboard navigation (4D continuous control)
python example/navigation/keyboard/navigation_keyboard_drone.py \
  -e UnrealNavigation-Demo_Roof-ContinuousColor-v0
```

### 4.3 Multi-Agent (Multi-Agent)

Multiple agents collaborate or perform independent tasks.

```bash
# Homogeneous multi-agent random policy baseline
python example/multi_agent/baseline/multi_random_baseline.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0

# Keyboard + random policy mix
python example/multi_agent/baseline/multi_keyboard_plus_random.py \
  -e UnrealTrack-Map_ChemicalPlant_1-MixedColor-v0

# Heterogeneous air-ground collaboration (3 ground + 1 drone)
python example/multi_agent/HeterogeneousCooperation/Aerial-Ground-Cooperative.py \
  -e UnrealTrack-Map_ChemicalPlant_1-ContinuousColor-v0
```

---

## 5. Interaction Control Details

> **Goal of this section**: Master the control methods for human players

### 5.1 Human Navigation (Mixed Action)

Applicable to `player` category, supporting movement + interaction.

#### Basic Movement

| Key | Function | Description |
|-----|----------|-------------|
| `I` | Forward | Move in facing direction |
| `K` | Backward | Move in backward direction |
| `J` | Left | Strafe left |
| `L` | Right | Strafe right |
| `↑` (Up arrow) | Look up | Increase pitch angle |
| `↓` (Down arrow) | Look down | Decrease pitch angle |

#### Interaction Actions

| Key | Function | Trigger Condition | Underlying API |
|-----|----------|-------------------|----------------|
| `F` | Open door | Within **1.5m** of door | `vbp {player} set_open_door 1` |
| `H` | Enter/exit vehicle | Near vehicle | `vbp {obj} enter_exit_car {player_index}` |
| `Ctrl` | Crouch | - | `vbp {player} set_crouch` |
| `E` | Pick up | Within **1.5m** of item | `vbp {player} set_pickup` |
| `Space` | Jump | - | `vbp {player} set_jump` |

> ⚠️ **Interaction Key Point**: Approach the interactive object (within 1.5m) and **press the corresponding key** to trigger, not automatic. Adjust head view to ensure the object is in sight.

#### Custom Keyboard Mapping

Interaction actions are mapped to APIs through `animation_dict`. You can customize keys in your code:

```python
# Example: Modify key mapping in navigation_keyboard_human.py
if event.key == pygame.K_f:  # Original F key for door
    action['animation'] = 'open_door'

# Change to X key for door
if event.key == pygame.K_x:
    action['animation'] = 'open_door'
```

Available `animation` action names:
- `'open_door'` → `vbp {player} set_open_door {state}`
- `'enter_vehicle'` → `vbp {obj} enter_exit_car {player_index}`
- `'pickup'` → `vbp {player} set_pickup`
- `'crouch'` → `vbp {player} set_crouch`
- `'jump'` → `vbp {player} set_jump`
- `'stand'` → Stand state
- `'liedown'` → Lie down
- `'drop'` → Drop

Full mapping definition in: `gym_unrealcv/envs/agent/character.py`, `animation_dict`

### 5.2 Drone Navigation (Continuous Action)

Applicable to `drone` category, 4-DOF continuous control.

| Key | Function | Control Dimension |
|-----|----------|-------------------|
| `W` | Forward | `vx` (forward/backward speed) |
| `S` | Backward | `vx` (forward/backward speed) |
| `A` | Left | `vy` (left/right speed) |
| `D` | Right | `vy` (left/right speed) |
| `E` | Ascend | `vz` (vertical speed) |
| `Q` | Descend | `vz` (vertical speed) |
| `J` | Turn left | `vyaw` (yaw angular velocity) |
| `L` | Turn right | `vyaw` (yaw angular velocity) |

---

## 6. Wrappers Quick Reference

> **Goal of this section**: Understand the role of wrappers in example code and learn to customize environment behavior

Wrappers are used to **dynamically adjust environment behavior without modifying environment source code**.

### 6.1 ConfigUEWrapper — Launch Parameter Configuration

Controls how the Unreal binary launches.

```python
from gym_unrealcv.envs.wrappers import configUE

env = configUE.ConfigUEWrapper(
    env,
    resolution=(160, 160),    # Rendering resolution (width, height)
    gpu_id=0,                 # Specify GPU ID
    offscreen=False,          # Whether to render off-screen (no window)
    display=None,             # Specify display (Linux, e.g., ':0.0')
    use_opengl=False,         # Whether to use OpenGL (default Vulkan)
    nullrhi=False,            # Whether to disable rendering (pure logic simulation)
    render_quality=7,         # Render quality 0-7 (7 is highest)
    use_lumen=False,          # Whether to enable Lumen global illumination
    sleep_time=5,             # Wait time after launch (seconds)
    comm_mode='tcp'           # Communication mode: 'tcp' / 'websocket'
)
```

**Typical Scenarios**:
- Multi-GPU training: `gpu_id=0,1,2...`
- Server deployment: `offscreen=True`
- Quick logic verification: `nullrhi=True` (no rendering, fastest)
- Visual quality priority: `render_quality=7, use_lumen=True`
- Performance priority: `render_quality=3, use_lumen=False`

### 6.2 RandomPopulationWrapper — Dynamic Population

Randomly set the number of agents per episode.

```python
from gym_unrealcv.envs.wrappers import augmentation

# Fixed 4 agents
env = augmentation.RandomPopulationWrapper(env, 4, 4, random_target=False)

# Random 5-10 agents per episode
env = augmentation.RandomPopulationWrapper(env, 5, 10, random_target=False)
```

**Typical Scenario**: Training policy generalization, simulating different crowd densities.

### 6.3 TimeDilationWrapper — Simulation Speed

Controls UE simulation speed.

```python
from gym_unrealcv.envs.wrappers import time_dilation

# Target 30 FPS, update every 60 steps
env = time_dilation.TimeDilationWrapper(env, reference_fps=30, update_steps=60)
```

**Typical Scenario**: Synchronization with external systems, or accelerating/decelerating simulation.

> **Full Wrapper API**: See `doc/wrapper.md`

---

## 7. Agent Spawn Mechanism (v3.0 Core Change)

> **Goal of this section**: Understand the templated spawn mechanism in v3.0 and be able to customize agent configurations

### 7.1 New vs Old Scheme

| Dimension | Old Scheme | New Scheme (v3.0) |
|-----------|------------|-------------------|
| Configuration | Fixed actor name + fixed count | Category template + runtime dynamic generation |
| Extensibility | High cost to extend mixed categories | Decoupled count, category combination, and experiment config |
| Multi-task reuse | Separate JSON for each task | Same map JSON reusable |

### 7.2 Templated Configuration Example

**Old Scheme** (fixed instance names):
```json
"agents": {
  "player_EP0_0": { "class_name": "bp_character_C" },
  "player_EP0_1": { "class_name": "bp_character_C" }
}
```

**New Scheme** (category templates):
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

### 7.3 Script-Side Control

```python
# Key: Set agents_category before reset()
env.unwrapped.agents_category = ["player", "player", "player", "drone"]
env = augmentation.RandomPopulationWrapper(env, 4, 4, random_target=False)
obs = env.reset()
```

### 7.4 Mixed Category Initialization Rules

| `agents_category` Length | Behavior |
|--------------------------|----------|
| 1 | All use same category (e.g., `["player"]` means all players) |
| Equal to population | Initialize per slot |
| Other (e.g., 2 ≤ len < population) | **Fallback to first category with warning** |

> ⚠️ **Critical Timing**: `agents_category` must be set before `reset()`, otherwise it won't take effect!

### 7.5 Migration Benefits

- Decoupled map configuration from experiment parameters
- Support for dynamic population changes
- Support for heterogeneous agent mixing

---

## 8. Configuration and Map JSON

> **Goal of this section**: Understand JSON configuration structure and be able to customize maps and tasks

### 8.1 Platform Auto-Switching

Three platform fields are written when generating JSON, **automatically selected at runtime**:

```json
{
  "env_bin": "Linux path",
  "env_bin_win": "Windows path",
  "env_bin_mac": "macOS path"
}
```

### 8.2 Complete JSON Example

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

### 8.3 Navigation Target Configuration

```json
"targets": { "Point": [] }
```

- Empty list is allowed but will trigger warning
- Reward is always 0 when target is empty
- Can be set dynamically: `env.unwrapped.set_navigation_targets(["TargetName"])`

---

## 9. FAQ



### Q1: Binary fails to start

- Verify `UnrealEnv` path is correct
- Linux: `chmod +x /path/to/binary`
- Verify platform match (don't mix Linux binaries on Windows)


## 2. Document Index

| Document | Content |
|----------|---------|
| `example/README.md` | Complete demo index |
| `doc/wrapper.md` | All Wrapper APIs |
| `doc/addEnv.md` | Adding custom environments |
| `CHANGELOG.md` | Version change log |

---

## Appendix A: `move_action` Field Details

### Player / Animal / Car / Motorbike

2-element array: `[angular, velocity]`
- `angular`: Angular velocity (degrees/second)
- `velocity`: Linear velocity (cm/second)

```json
"move_action": [
  [0, 100],    // Forward
  [0, -100],   // Backward
  [15, 50],    // Left turn forward
  [-15, 50],   // Right turn forward
  [30, 0],     // Turn left in place
  [-30, 0],    // Turn right in place
  [0, 0]       // Stop
]
```

### Drone

4-element array: `[vx, vy, vz, vyaw]`
- `vx`: Forward/backward speed
- `vy`: Left/right speed
- `vz`: Vertical speed (ascend/descend)
- `vyaw`: Yaw angular velocity

```json
"move_action": [
  [0.5, 0, 0, 0],    // Forward
  [-0.5, 0, 0, 0],   // Backward
  [0, 0.5, 0, 0],    // Right
  [0, -0.5, 0, 0],   // Left
  [0, 0, 0.5, 0],    // Ascend
  [0, 0, -0.5, 0],   // Descend
  [0, 0, 0, 1],      // Turn left
  [0, 0, 0, -1],     // Turn right
  [0, 0, 0, 0]       // Hover
]
```

Full template definition reference: `generate_env_config.py`
