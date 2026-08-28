# UnrealZoo Example Index

This directory collects runnable examples for the main UnrealZoo research
workflows. Start with the task-specific guide when one is available. For
folders that do not yet have a dedicated README, the links below point directly
to representative entry scripts.

Before running an example, complete the
[main Quick Start](../README.md#-quick-start), select a compatible code branch
and environment package, and configure the UnrealEnv path.

## Example Categories

| Category | Guide | Representative entry points | Launch mode |
|---|---|---|---|
| **Navigation** | [Navigation examples](navigation/) | [Human keyboard navigation](navigation/keyboard/navigation_keyboard_human.py) · [Drone keyboard navigation](navigation/keyboard/navigation_keyboard_drone.py) | Binary auto-launch |
| **Tracking** | [Tracking examples](tracking/) | [Automatic tracking](tracking/basic/tracking_auto_basic.py) · [Keyboard animal tracking](tracking/basic/tracking_keyboard_animal.py) · [NavMesh tracking](tracking/navmesh/tracking_auto_navmesh.py) | Binary auto-launch |
| **Multi-agent** | [Multi-agent examples](multi_agent/) | [Random baseline](multi_agent/baseline/multi_random_baseline.py) · [Keyboard + random agents](multi_agent/baseline/multi_keyboard_plus_random.py) · [Aerial-ground cooperation](multi_agent/HeterogeneousCooperation/Aerial-Ground-Cooperative.py) | Binary auto-launch |
| **Interaction** | [Interaction examples](interaction/) | [Object interaction](interaction/keyboard/interaction_keyboard_object.py) · [Vehicle interaction](interaction/keyboard/interaction_keyboard_vehicle.py) | Binary auto-launch |
| **Data recording** | [Data-recording examples](DataRecording/) | [Video recording pipeline](DataRecording/VideoRecordingPipeline.py) | Start the binary manually |
| **v3.1 features** | [Latest feature guide](new_features/README.md) | [LiDAR street mapping](new_features/suburb_street_slam.py) · [Occupancy viewer](new_features/realtime_scene_occupancy_gpu.py) · [Drone customization](new_features/drone_mesh_switch_demo.py) | See the feature guide |
| **MuJoCo Go1** | [Go1 setup and demos](mujoco/README.md) | [Keyboard control](mujoco/go1_keyboard_control.py) · [Parkour](mujoco/go1_parkour.py) | Start the binary or Editor manually |
| **VLN baselines** | [VLN baseline guide](VLN_Baseline/README.md) | Uni-NaVid · ViNT / NoMaD · StreamVLN | Follow the model-specific guide |

## Runtime Conventions

- **Binary auto-launch** means the registered Gym environment resolves the
  binary from UnrealEnv and starts it during environment initialization.
- **Manual launch** means the script connects to an already running UnrealZoo
  binary or Unreal Editor session; the host and port must match that session.
- Commands, optional dependencies, controls, and asset requirements for v3.1
  features are documented in the
  [latest feature guide](new_features/README.md).

## Related Documentation

- [Main README](../README.md)
- [Chinese README](../README_ch.md)
- [v3.1 changelog](../doc/CHANGELOG_v3.1.md)
- [UnrealCV+ documentation](https://docs.unrealcv.org/en/latest/unrealcv_plus/index.html)
