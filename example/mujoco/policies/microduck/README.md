# Microduck policies

Official Pollen Robotics 61-D policies pinned to the `daemon-v0.10.0`
release of <https://github.com/pollen-robotics/microduck>.

- `alpha_walking.onnx`: rough-terrain velocity policy.
- `alpha_stand.onnx`: standing and body-pose policy.
- `alpha_sitstand.onnx`: commanded sit and rise.
- `alpha_ground_pick.onnx`: phase-commanded ground pick.
- `ball_kick_left.onnx` / `ball_kick_right.onnx`: leg-specific kicks.
- `roulade.onnx`: one-second forward roll.

The upstream code/policy repository is Apache-2.0. The robot 3D model files
used by the Unreal asset are separately described by `microduck_rl` as
Creative Commons BY-SA-NC; review those terms before redistributing a binary.
