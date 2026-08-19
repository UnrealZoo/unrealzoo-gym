# MuJoCo Go1 examples

| Keyboard Control | Parkour: Third-Person View | Parkour: Depth Observation |
|:---:|:---:|:---:|
| ![MuJoCo Go1 keyboard control](../../doc/figs/new_features/mujoco_go1_keyboard.gif) | ![MuJoCo Go1 parkour third-person view](../../doc/figs/new_features/mujoco_go1_parkour_third_person.gif) | ![MuJoCo Go1 parkour depth observation](../../doc/figs/new_features/mujoco_go1_parkour_depth.gif) |

This directory exposes two user-facing Go1 demos. Both connect directly to an
already running UnrealZoo/UnrealCV session. They reuse an existing
`BP_UnitreeGo1` when one is present; otherwise they spawn a temporary Go1 and
remove only that actor when the script exits.

| Demo | Entry point | Policy input | Intended use |
| --- | --- | --- | --- |
| Basic movement | `go1_keyboard_control.py` | 48D proprioception + velocity command | Flat and moderately rough ground |
| Visual parkour | `go1_parkour.py` | Proprioception + delayed 48x64 depth + recurrent state | Obstacles, ledges, gaps, and parkour experiments |

## Requirements

- Run commands from the `unrealzoo-gym` repository root.
- Install the repository into the active Python environment with
  `python -m pip install -e .`. Basic control additionally requires
  `onnxruntime`; parkour additionally requires `torch`. The Go1 policy and
  parkour checkpoint are already included in this directory.
- Start the Unreal Editor in PIE, or manually launch the packaged v4 UnrealZoo
  environment, before running either script. Wait until the map is loaded and
  UnrealCV is listening. `--port` must match the server's configured `Port=`
  value or the port reported in its startup log.
  These demos do not accept `--binary`, `--env-id`, or `--sleep-time` and do
  not create a Gym task.
- Both demos place the acquired Go1 at the fixed world-space test location
  `(0, 0, 500)` cm before starting MuJoCo. Override it with
  all three `--spawn-x/--spawn-y/--spawn-z` arguments when that point is not a
  safe road or platform in the loaded map.
- If the scene contains multiple Go1 actors, select the intended one with
  `--actor ACTOR_NAME`. Run only one Go1 controller per UnrealCV server.
- Keep detailed collision on stairs, curbs, and platforms. The MuJoCo bridge
  captures environment collision around the robot as it moves.

Minimal launch sequence:

1. Start the v4 binary or enter Editor PIE.
2. Wait for the level to finish loading and note the UnrealCV port.
3. Run one of the commands below with the matching `--port`.

## Basic movement

The basic demo loads the bundled 48D-to-12D ONNX velocity policy and controls
the acquired Go1 at 50 Hz:

```cmd
python .\example\mujoco\go1_keyboard_control.py --host 127.0.0.1 --port 9000 --duration 0
```

Both interactive demos use the same controls: I/K for forward/backward, J/L
for left/right turn, Space to stop, and X/Esc to exit. On Windows, commands
are active only while a movement key is held. Useful tuning arguments are
`--vx`, `--yaw-rate`, `--warmup`, and `--command-ramp`.

The implementation is split between:

- `go1_keyboard_control.py`: actor lifecycle, keyboard input, and the
  interactive control loop.
- `go1_locomotion.py`: shared bridge protocol, observation conversion, ONNX
  policy loading, and action mapping.

## Visual parkour

The parkour demo loads the bundled Robot Parkour Learning Go1 checkpoint. It
runs the recurrent locomotion actor at 50 Hz and supplies delayed depth at the
checkpoint's 10 Hz capture rate:

```cmd
python .\example\mujoco\go1_parkour.py --host 127.0.0.1 --port 9000 --camera-id auto --vx 1.0 --command-ramp 0 --duration 0
```

`--camera-id auto` first binds to the FusionCamSensor on the acquired Go1.
If none is available, the demo creates a temporary fallback camera and removes
it on exit. Use `--preserve-attached-camera-pose` when the sensor pose authored
in the Blueprint should be kept instead of applying checkpoint camera
extrinsics.

Parkour additionally uses R to reset recurrent state. The default
`--keyboard-mode hold` sends a zero command as soon as the key is released.
The live observation window shows raw UnrealCV depth, the processed 48x64
policy depth, proprioception, contacts, and policy actions. Disable it with
`--no-visualize-observation`.

On Windows the movement keys are sampled globally, so the packaged game or
observation window may remain focused. Hold I/K to move forward/backward and
J/L to turn; releasing the key sends a zero command in the default hold mode.

The policy integration is implemented by:

- `go1_parkour.py`: UnrealCV camera binding, synchronized control, delayed
  depth sampling, keyboard commands, and observation visualization.
- `parkour_policy.py`: checkpoint/config loading, depth preprocessing,
  recurrent-state management, joint ordering, and bridge action mapping.
- `go1_observation_visualizer.py`: live observation diagnostics and anomaly
  snapshots under `observation_debug/`.

Run an offline checkpoint smoke test without PIE:

```cmd
python .\example\mujoco\go1_parkour.py --self-test
```

## Advanced policy source and citation

`go1_parkour.py` adapts the official [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour)
Go1 policy and checkpoint at pinned upstream commit
`5e1f4136f7fec9dac2741add63ca58d32090666f`. See the
[project page](https://robot-parkour.github.io/), the
[published CoRL 2023 paper](https://proceedings.mlr.press/v229/zhuang23a.html),
and the checked-in [`SOURCE.json`](policies/robot_parkour_go1/SOURCE.json) for
provenance and file hashes.

If this advanced demo contributes to a publication, cite the upstream work:

```bibtex
@InProceedings{pmlr-v229-zhuang23a,
  title     = {Robot Parkour Learning},
  author    = {Zhuang, Ziwen and Fu, Zipeng and Wang, Jianren and
               Atkeson, Christopher G. and Schwertfeger, S{\"o}ren and
               Finn, Chelsea and Zhao, Hang},
  booktitle = {Proceedings of The 7th Conference on Robot Learning},
  pages     = {73--92},
  year      = {2023},
  volume    = {229},
  series    = {Proceedings of Machine Learning Research}
}
```

Robot Parkour Learning was trained for discrete parkour obstacles; stable
operation on arbitrary continuous staircases is not guaranteed. A valid depth
image is necessary but not sufficient: camera extrinsics, depth scaling and
latency, collision fidelity, joint/action mapping, and the checkpoint's
training distribution must all match closely.

## Diagnostic utilities

`go1_pose_preview.py` and the standalone mode in `go1_locomotion.py` remain
available for bridge diagnostics. They are not the two primary interactive
demos above and may expose spawn-oriented test options.

Both main demos stop MuJoCo but do not delete a reused scene actor during
cleanup. If a demo spawned a fallback Go1, it destroys only that actor during
normal exit or error cleanup. Add `--keep-actor` to retain it. The selected
actor is moved to the configured XYZ start whether it was reused or spawned.
