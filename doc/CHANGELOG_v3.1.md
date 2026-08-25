# UnrealZoo v3.1 Changelog

Status: **Development snapshot**  
Version: **v3.1**
Scope baseline: changes after [`v3.0-stage`](CHANGELOG_v4.md)

This document fixes the feature boundary for the next major UnrealZoo update
before the root README, tutorials, GIFs, and example index are reorganized.
It separates end-user UnrealZoo features from the broader UnrealCV+ platform
improvements included in the release.

The UnrealCV+ label means that a capability is developed and validated first
with UnrealZoo builds. It does not imply a separate UnrealCV plugin or that
every item is already part of the upstream open-source UnrealCV baseline. See
the official [UnrealCV+ documentation](https://docs.unrealcv.org/en/latest/unrealcv_plus/index.html)
and [feature overview](https://docs.unrealcv.org/en/latest/unrealcv_plus/overview.html).

## Release highlights

| Area | User-facing result | Recommended showcase |
|---|---|---|
| UnrealCV rendering and capture | Up to 14.66× standard-camera speedup and up to 4.96× panorama speedup | Benchmark comparison and real-time capture clip |
| LiDAR | XYZI observations and pose-conditioned street mapping | Suburb street-mapping Python demo + GIF |
| MuJoCo | Unreal-rendered Go1 driven by MuJoCo state and policy control | Go1 parkour Python demo + GIF |
| Occupancy voxels | LINGO-compatible scene occupancy grids for embodied-agent observation | Profile/shape documentation + visualization |
| Runtime drone visual customization | Switch five production-ready models with animated propellers, use a customization template, or load an external Static Mesh without respawning the pawn | Appearance-cycle demo + GIF |
| Social animation | Trigger the newly packaged character social-animation library at runtime | Animation-cycle demo + GIF |
| External 3DGS environment | Load a user's packaged 3DGS scene at runtime and reuse UnrealZoo agents and observations in it | External-scene loading workflow + agent demo |

## Added

### UnrealZoo features

#### 1. LiDAR observation and mapping

- Added a camera LiDAR observation that returns an `N x 4` float32 XYZI point
  cloud.
- Added a NumPy observation interface:

  ```text
  vget /camera/[camera_id]/lidar npy
  ```

- Added a live `SuburbNeighborhood_Day` SLAM-style example that combines the
  player RGB view, the current sensor-local scan, the synchronized camera pose,
  and an accumulated top-down voxel map.
- Added scan diagnostics for expected ray-grid layout, duplicate/order/range
  anomalies, synchronized pose span, stationary consistency, and map overlap.

#### 2. MuJoCo Go1 integration

- Added an Unreal-rendered MuJoCo simulation path for the Unitree Go1
  quadruped.
- Added Go1 policy control with a 12-D action, a 3-D velocity command, and a
  48-D policy observation.
- Added velocity and parkour dynamics for Go1, together with keyboard, policy,
  pose-preview, observation-visualization, and GIF-recording examples.

#### 3. Scene occupancy voxel observation

- Added boolean scene occupancy volumes with C-order axis layout
  `[x, y_up, z]`.
- Added `lingo_vis` and `lingo_train` profiles at 2 cm resolution.
- Added `bounds` and `mesh` voxelization methods, optional dynamic actors, and
  camera-relative origin/yaw arguments.
- Added NPY observation and machine-readable specification endpoints:

  ```text
  vget /scene/occupancy npy [profile] [method] [origin_cm x y z yaw include_dynamic]
  vget /scene/occupancy/spec [profile] [method]
  ```

#### 4. Runtime drone visual customization

- Added five production-ready appearances with animated propellers: `spy`,
  `fpv`, `police`, `baba`, and `delivery`, plus the `template` appearance for
  customization.
- Appearance switching on `BP_Drone_customized` does not respawn the pawn or
  replace its controller, movement, physics, camera, or current task state.
- Added `set_app`, `reset_app`, and `get_available_apps` Blueprint-callable
  functions.
- Added custom static-mesh setters, including string-path variants for assets
  supplied by a mounted PAK:
  `set_app_mesh_path` and `set_app_mesh_path_transform`.
- External Static Mesh loading is intended for rapid asset integration and
  visual customization. It preserves the drone systems above and displays the
  imported geometry as authored; component-level animations are not generated
  automatically for a single Static Mesh.

#### 5. Character social animations

- Added the `set_social_anim` Blueprint function for runtime social-animation
  selection.
- Added montage playback and stop handling for social actions.
- Packaged the `SocialAnimsBundle2` content with the UnrealZoo build, including
  party, female, and in-car animation groups.
- Animation-name examples include `AirKiss`, `DrinkAndTalk`, `FemaleLaugh`,
  `Reading`, `Selfy`, `SmallTalk`, `Smoking`, and `SpeakOnPhone`.
- The input is the case-sensitive suffix after `AM_`; see the
  [complete argument list](../example/new_features/README.md#5-character-social-animations).

#### 6. User-supplied 3DGS environments through dynamic PAK loading

- Added a release workflow in which users import and cook their own 3DGS asset
  into an external PAK instead of rebuilding the complete UnrealZoo project.
- Added a guided runtime workflow for discovering and activating compatible
  3DGS content from an external package.
- The loaded 3DGS environment remains a normal UnrealZoo world: existing
  players, drones, cameras, observations, and task logic can be spawned or
  reused on top of it.
- Runtime loading consumes an already imported and cooked Gaussian-splat asset;
  raw PLY import is an editor/cooking step, not an environment-reset operation.

### UnrealCV+ user-facing improvements

#### Rendering and observation performance

- Standard-camera capture reaches `89.77 FPS` at 1080p and `96.20 FPS` at 2K.
- Standard-camera speedup ranges from `1.58×` at 480p to a peak of `14.66×`
  at 4K; 8K capture improves from `1.40 FPS` to `18.10 FPS`.
- Panorama speedup ranges from `1.42×` at 480p to `4.96×` at 8K; 4K panorama
  capture improves from `4.00 FPS` to `12.91 FPS`.
- Improved responsiveness for continuous observation loops, interactive agent
  control, and dataset recording workloads.
- Reduced the practical cost of collecting multiple visual modalities from an
  active UnrealZoo scene.

#### Panoramic capture extensions

- Extended equirectangular panoramic capture beyond lit output to normal,
  object-mask, and depth modalities.
- Added configurable panoramic output resolution.

#### Built-in recording workflow

- Improved the built-in camera and actor recording workflow for sustained
  dataset capture.
- Added user controls for output channels, frame rate, duration, warm-up,
  pause/resume behavior, and output naming.

#### Camera, annotation, and dynamic content workflows

- Added stable camera identifiers while keeping legacy numeric camera scripts
  compatible.
- Expanded scene and object annotation controls for dataset generation.
- Added runtime external-content loading and asset-path spawning so users can
  extend an UnrealZoo environment without rebuilding the complete project.

#### MuJoCo command bridge

- Added UnrealCV object commands for Go1 MuJoCo setup, synchronous stepping,
  observation retrieval, policy action, and policy command.

Several of these UnrealCV+ capabilities first appeared during the v3.0 stage.
They are listed here because this release turns them into complete user-facing
workflows, rather than claiming that every item was first introduced in this
cycle.

## Changed

- New feature examples attach to an already running Unreal Editor/PIE session
  by default instead of silently launching a second Unreal process.
- The street-mapping example defaults to the spawned player camera and treats
  Unreal world positions as centimetres while LiDAR XYZ values are metres.
- Corrected RGB colour reproduction in the new visualization examples.
- Dynamic content examples distinguish filesystem PAK paths, Unreal package
  paths, and full Unreal object paths instead of treating them as
  interchangeable strings.

## Compatibility and limitations

- Camera-to-LiDAR extrinsics must be applied when the sensor is not colocated
  with the camera. A demo may assume identity only when that assumption is
  explicitly shown.
- Unreal Engine locations use centimetres. LiDAR and visualization examples
  expose metric coordinates; conversions must occur exactly once.
- Occupancy profiles are large dense boolean arrays. Select the smallest
  profile that satisfies the task and avoid retaining unnecessary frames.
- `mesh` occupancy accuracy depends on compatible geometry data and may fall
  back to bounds for unsupported scene assets.
- A user 3DGS package must be compatible with the target UnrealZoo engine,
  platform, and content requirements.
- The social-animation command accepts names that exist in the cooked bundle;
  unknown names should be treated as errors rather than silently substituted.
- UnrealCV+ availability is tied to the UnrealZoo-tested plugin build. Scripts
  should query command/help availability when they also target upstream or
  older UnrealCV installations.

## Examples and media

- [`suburb_street_slam.py`](../example/new_features/suburb_street_slam.py)
  demonstrates player-controlled LiDAR street mapping.
- [`lidar_visualization.py`](../example/new_features/lidar_visualization.py)
  visualizes individual LiDAR observations and accumulated maps.
- [`go1_parkour.py`](../example/mujoco/go1_parkour.py) demonstrates MuJoCo Go1
  control in an Unreal-rendered scene.
- [`realtime_scene_occupancy_gpu.py`](../example/new_features/realtime_scene_occupancy_gpu.py)
  renders live shared-memory occupancy data in four GPU-instanced views.
- [`drone_mesh_switch_demo.py`](../example/new_features/drone_mesh_switch_demo.py)
  cycles the available drone appearances on a live pawn.
- The packaged-binary command
  `vset /action/game/level /Game/3dgs/custom_3dgs` demonstrates loading an
  external 3DGS level while retaining the UnrealZoo interfaces.

- [`lidar_street_slam.gif`](figs/new_features/lidar_street_slam.gif): full
  LiDAR street-mapping capture converted from MP4 for README use.
- [`mujoco_go1_keyboard.gif`](figs/new_features/mujoco_go1_keyboard.gif): basic
  MuJoCo Go1 keyboard-control demonstration.
- [`mujoco_go1_parkour_third_person.gif`](figs/new_features/mujoco_go1_parkour_third_person.gif):
  advanced parkour policy shown from a third-person view.
- [`mujoco_go1_parkour_depth.gif`](figs/new_features/mujoco_go1_parkour_depth.gif):
  UnrealCV depth and the depth input used by the parkour policy.
- [`drone_mesh_switch.gif`](figs/new_features/drone_mesh_switch.gif): runtime
  mesh switching on a live drone without respawning the pawn.
- [`3dgs_dynamic_load.gif`](figs/new_features/3dgs_dynamic_load.gif): an
  external 3DGS environment loaded at runtime with an UnrealZoo-supported
  actor reused in the scene.
- [`social_animation.gif`](figs/new_features/social_animation.gif): multiple
  UnrealZoo characters playing runtime-selected social animations.

Environment-specific usage questions and detailed behavior can be discussed in
[GitHub Discussions](https://github.com/UnrealZoo/gym-unrealzoo/discussions).

## TODO

- [ ] MuJoCo Unitree G1 integration and examples.
