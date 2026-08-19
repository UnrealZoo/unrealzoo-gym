# UnrealZoo latest feature demos

This directory is the staging area for newly added UnrealZoo/UnrealCV
capabilities. Each feature has a reproducible entry point and a README-ready
way to show its result. Most Python demos launch a registered
`gym_unrealcv` environment; the Go1 demos intentionally use a direct UnrealCV
connection to an environment started by the user.

## Feature matrix

| Feature | What the demo proves | Entry point | Recommended visual |
|---|---|---|---|
| LiDAR observation + mapping | XYZI observation, scan checks and pose-conditioned street mapping | [`suburb_street_slam.py`](suburb_street_slam.py) | RGB + current scan + accumulated map dashboard |
| MuJoCo | UE rendering driven by MuJoCo state and policy observations | [`../mujoco/go1_parkour.py`](../mujoco/go1_parkour.py) | Keyboard, parkour third-person, and depth-observation GIFs |
| Scene occupancy voxels | LINGO-compatible bool XYZ grid, bounds/mesh modes, camera-relative origin | [`occupancy_voxel_demo.py`](occupancy_voxel_demo.py) | Top-down projection, height slice and 3-D voxels |
| Runtime drone visual customization | Five production-ready models with animated propellers, one customization template, and external Static Mesh support on the same live pawn | [`drone_mesh_switch_demo.py`](drone_mesh_switch_demo.py) | Third-person GIF covering the built-in appearances |
| Character social animations | Runtime selection from the packaged party, everyday, and in-car animation groups | `BP_Character.set_social_anim` | Third-person character montage capture |
| Dynamic 3DGS load | A user-supplied packaged 3DGS level is loaded and reused by UnrealZoo agents | `vset /action/game/level /Game/3dgs/custom_3dgs` in the UE binary | External scene + supported-actor GIF |

## Setup and launch modes

Install this repository in the active Python environment. `UnrealEnv` points
to the directory containing the packaged environment folder referenced by the
generated JSON configuration:

```cmd
conda activate unrealzoo
python -m pip install -e .
set "UnrealEnv=D:\UnrealZoo"
```

Run commands from the repository root.

| Demo | Startup model | Port behavior |
|---|---|---|
| LiDAR, occupancy voxel, drone | Gym resolves `env_bin` below `UnrealEnv`, then launches and closes the environment | The requested port must be unused before launch |
| Go1 keyboard and parkour | The user starts the binary or Editor PIE first | The script connects to the already listening `--host/--port` |
| Dynamic 3DGS level | The user starts the v4.0 binary first | Enter the level-load command in the binary's UnrealCV console |

Do not point a Gym-launched demo at a port occupied by a manually started
binary. Conversely, the Go1 scripts cannot start a binary and will fail if no
UnrealCV server is listening. Wait until a manually started map is fully loaded
before running either Go1 script.

The base installation covers the visualization demos. Basic Go1 policy
inference additionally imports `onnxruntime`; the advanced parkour policy
imports `torch` and uses the bundled `rsl_rl` runtime. Install those optional
inference packages into the same Python environment that runs the scripts.

### Optional README media utility

[`record_feature_gif.py`](record_feature_gif.py) is a maintainer utility for
capturing an UnrealCV camera, visible feature window, or desktop region. It is
not required to run any feature demo.

```cmd
python example\new_features\record_feature_gif.py --source window --window-title "UnrealZoo" --output artifacts\feature.gif
```

## 1. LiDAR observation and pose-conditioned street mapping

![LiDAR street mapping](../../doc/figs/new_features/lidar_street_slam.gif)

Run directly from the repository root:

```cmd
python example\new_features\suburb_street_slam.py --width 640 --height 360 --transport auto
```

The demo spawns a temporary player, normally discovers it as camera 1, and
keeps RGB, camera pose and LiDAR in one diagnostic loop. It displays the
correctly decoded RGB player view, the current sensor-local LiDAR scan, and a
centimetre-to-metre pose-conditioned 2-D voxel map. It also checks hit count,
the expected ray-grid layout, duplicate/order/range errors, pose consistency,
stationary consistency and map overlap.

Controls are `I/K` (forward/back), `J/L` (turn), arrow keys (look), `Space`
(jump), `Ctrl` (crouch), `C` (clear map) and `P` (pause mapping).
Keyboard input is global on Windows, so the Unreal player or Matplotlib window
may remain focused. Close the dashboard, press Esc, or use Ctrl+C to exit.

```cmd
rem Generic live map or one-frame observation check against a running server.
python example\new_features\lidar_visualization.py --camera-id 1
python example\new_features\lidar_visualization.py --camera-id 1 --once --no-show --save artifacts\lidar.png
```

LiDAR points are sensor-local metres; Unreal locations are centimetres and
rotations are `[pitch, yaw, roll]` degrees. The demo combines each observation
with the corresponding camera pose before accumulating the map.

## 2. MuJoCo Go1 control and observations

| Keyboard Control | Parkour: Third-Person View | Parkour: Depth Observation |
|:---:|:---:|:---:|
| ![MuJoCo Go1 keyboard control](../../doc/figs/new_features/mujoco_go1_keyboard.gif) | ![MuJoCo Go1 parkour third-person view](../../doc/figs/new_features/mujoco_go1_parkour_third_person.gif) | ![MuJoCo Go1 parkour depth observation](../../doc/figs/new_features/mujoco_go1_parkour_depth.gif) |

First manually start the v4 packaged binary or enter Editor PIE and wait for
the map and UnrealCV server to become ready. The port below must match that
server; `--binary`, `--env-id`, and `--sleep-time` do not apply to Go1.

```cmd
rem Basic keyboard locomotion policy.
python example\mujoco\go1_keyboard_control.py --host 127.0.0.1 --port 9000

rem Advanced vision-based parkour policy.
python example\mujoco\go1_parkour.py --host 127.0.0.1 --port 9000 --command-mode keyboard
```

The parkour demo combines UE RGB/depth, MuJoCo proprioception and keyboard or
policy actions. Its advanced policy and Go1 checkpoint come from the official
[Robot Parkour Learning repository](https://github.com/ZiwenZhuang/parkour);
cite Zhuang et al., “Robot Parkour Learning,” CoRL 2023, PMLR 229:73–92
([paper](https://proceedings.mlr.press/v229/zhuang23a.html)). See
[`example/mujoco/README.md`](../mujoco/README.md) for full BibTeX, model setup,
action semantics and runtime setup.

Both demos use I/K for forward/backward, J/L for turning, Space to stop, and
X/Esc to exit. They move the acquired Go1 to `(0, 0, 500)` cm by default; use
all three `--spawn-x/--spawn-y/--spawn-z` arguments for another safe road or
platform location. If the scene contains multiple Go1 actors, pass `--actor`
to select one explicitly. Run only one Go1 controller per UnrealCV server.

## 3. Scene occupancy voxel observation

The observation is a C-order bool NPY grid with axis order `[x, y_up, z]`:

| Profile | Bounds in metres `(min -> max)` | Shape | Resolution |
|---|---|---:|---:|
| `lingo_vis` | `(-4, 0, -6) -> (4, 2, 6)` | `400 x 100 x 600` | 2 cm |
| `lingo_train` | `(-3, 0, -4) -> (3, 2, 4)` | `300 x 100 x 400` | 2 cm |

`bounds` rasterizes visible component AABBs and is faster. `mesh` tests render
or cooked collision triangles and falls back to bounds where CPU triangle data
is unavailable. Collision enable state is not the visibility filter.

```cmd
rem Camera-relative accurate grid with three-panel visualization.
python example\new_features\occupancy_voxel_demo.py --camera-id auto --profile lingo_vis --method mesh

rem Reproducible headless artifact.
python example\new_features\occupancy_voxel_demo.py --profile lingo_train --method bounds --max-steps 1 --save artifacts\occupancy_voxel.png --no-show
```

`mesh` is the accurate but heavier mode; use `bounds` for a faster structural
preview. A `--no-show` run is continuous unless `--max-steps` is supplied.

Raw UnrealCV commands:

```text
vget /scene/occupancy/spec lingo_vis mesh
vget /scene/occupancy npy lingo_vis mesh Xcm Ycm Zcm YawDeg 0
```

The specification response describes shape, dtype, layout, grid bounds,
origin, yaw, and voxelization method so downstream models can validate their
expected observation contract.

## 4. Runtime drone visual customization

![Runtime drone mesh switching](../../doc/figs/new_features/drone_mesh_switch.gif)

The built-in mode provides five production-ready drone models with animated
propellers, plus one template appearance for customization. Switching the
visual model does not respawn `BP_Drone_customized`: its controller, movement,
physics, camera, and current task state remain active.

| Index | Appearance | Type |
|---:|---|---|
| 0 | `spy` | Production-ready animated model |
| 1 | `fpv` | Production-ready animated model |
| 2 | `police` | Production-ready animated model |
| 3 | `template` | Customization template |
| 4 | `baba` | Production-ready animated model |
| 5 | `delivery` | Production-ready animated model |

This command launches the default Navigation map with the customized drone as
its Gym agent, cycles the five production-ready models and the template twice,
and restores `spy` on exit:

```cmd
python example\new_features\drone_mesh_switch_demo.py --interval 2 --cycles 2 --render
```

Without `--render`, this finite demo reports appearance changes only in the
terminal. It restores the default `spy` appearance before closing unless
`--no-reset` is supplied.

The underlying built-in API is simply `vbp <actor> set_app 2`. For rapid asset
integration, visual prototyping, and domain customization, the pawn also
exposes `set_app_mesh_path` for compatible external Static Mesh assets. Pass a
cooked Unreal object path, not a filesystem path.

External Static Mesh loading replaces only the drone's visual geometry and
preserves its controller, physics, camera, and task state. The imported mesh is
displayed as authored; component-level animation such as propeller rotation is
not generated automatically for a single Static Mesh.

## 5. Character social animations

![UnrealZoo character social animations](../../doc/figs/new_features/social_animation.gif)

`BP_Character` exposes `set_social_anim` independently from the drone
appearance APIs. It selects social actions from the packaged
`SocialAnimsBundle` content while the drone feature replaces a drone's mesh;
the two systems do not share state or assets.

The argument is the asset name after the `AM_` prefix and is
**case-sensitive**. For example, asset `AM_MakeJokes` is invoked as:

```text
vbp <actor> set_social_anim MakeJokes
```

Use one of the following exact argument strings:

```text
Waiter3
Waiter2
Waiter
StandUp
SowFocus
SmokeHookah
SlippingInBar2
SlippingInBar
ShyDance
SecurityAtEntrance
Security
ReadRap
MakeJokes
ListenRap
Karaoke
In_Bar
Guitarist
GivesAGift
GetingCakeInface
Fans2
Fans
EatingOnparty3
EatingOnparty2
EatingOnParty
DrunkWalk
DrunkMan
Drummer
DrinkCoffe
DJ2
DJ
DancingWalk
Dancing3
Dancing2
Dancing
CoolDude
Conversation_On_Party2
Conversation_On_Party
ConversationInBar2
ConversationInBar
CarryCake
CakeINFace
BlowOutCandlesSittign
BlowOutCandles
Bartender
UnderHoodRT
UnderHoodPL2
UnderHoodPL
TurnOnWindshieldWipers
TurnOnTheTurnSignal
TurnOnAC
TurnOfEngineGetOutOfCar
TalkInCar
Reversing
Passanger
OpenGloveBox
LookAround
GetInCarStartEngine
GearShift
FullThrottle
Fixing
FillingOil
FastenSeatBelt
EmergencyBraking
DrivingTwoHandTurnRight
DrivingTwoHandTurnLeft
DrivingTwoHands
DrivingOneHandTurnLeft
DrivingOneHand
DrinkInCar
ChillDriving
Braking
AtGasStation
AtCarWashRT
AtCarWashPL
AtCarWash2RT
AtCarWash2PL
DrivingOneHandTurnRight
SpeakOnPhone
Smoking
SmallTalk
Sitting3
Sitting
ShyGirl
Selfy
Rest3
Rest2
Rest
Reading
PosingForPhoto
Posing4
Posing2
Posing
Poising3
PickUp
PickedUp
OnTheBeach
OldLady
MakeUpSitting
MakeUp
FixesHairstyle
FemaleLaugh
DrinkCoffy
DrinkAndTalk
Combing2
Combing
AirKiss
```

## 6. Dynamic 3DGS environment load

![External 3DGS environment with an UnrealZoo-supported actor](../../doc/figs/new_features/3dgs_dynamic_load.gif)

Users prepare and package a compatible Gaussian-splat environment ahead of
time. Start the UnrealZoo v4.0 binary, open its UnrealCV command console, and
load the packaged level directly:

```text
vset /action/game/level /Game/3dgs/custom_3dgs
```

This activates the external 3DGS level without rebuilding the complete
UnrealZoo project. Existing cameras, agents, observation APIs, and interaction
APIs remain available in the loaded level. Raw PLY preparation is not repeated
on every environment reset.

The external PAK must be compatible with the target UnrealZoo engine and
plugin build, and its cooked level must resolve to
`/Game/3dgs/custom_3dgs`. Filesystem paths and uncooked source assets are not
valid Unreal level paths.

## Media checklist

Keep README media under `doc/figs/new_features/` and raw captures under
`artifacts/`. Before committing, prefer 8-14 seconds, 8-12 FPS, 640-960 px
width and 64-128 colours; keep each GIF below roughly 10 MB. Crop editor chrome
unless it explains the feature, and never use a static screenshot as evidence
of a runtime transition.
