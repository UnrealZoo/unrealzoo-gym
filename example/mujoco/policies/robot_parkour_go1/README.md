# Robot Parkour Learning — Go1

This directory packages the official Go1 visual-distillation checkpoint used by
`../../go1_parkour.py`. The demo keeps the existing 48D ONNX locomotion example
unchanged; this policy instead consumes 48D proprioception, a 48x64 forward
depth image, and recurrent state.

Upstream: <https://github.com/ZiwenZhuang/parkour>  
Project page: <https://robot-parkour.github.io/>  
Go1 deployment notes: <https://github.com/ZiwenZhuang/parkour/blob/main/onboard_codes/Deploy-Go1.md>  
Pinned source commit: `5e1f4136f7fec9dac2741add63ca58d32090666f`  
License: MIT; see `UPSTREAM_LICENSE`.

Bundled files:

- `model_674000.pt`: official Go1 parkour checkpoint.
- `config.json`: matching official training/deployment configuration.
- `rsl_rl/`: the minimum official Python policy runtime.
- `SOURCE.json`: provenance and SHA-256 hashes.

Offline load/inference check:

```cmd
conda run --no-capture-output -n unrealzoo python .\example\mujoco\go1_parkour.py --self-test
```

Run after starting an UnrealCV-enabled editor session or packaged build. The
demo spawns its own `BP_UnitreeGo1` in front of camera 0 and settles it onto
collision geometry before MuJoCo starts:

```cmd
conda run --no-capture-output -n unrealzoo python ^
  .\example\mujoco\go1_parkour.py ^
  --spawn-camera-id 0 ^
  --vx 1.0 --warmup 0.8 --print-every 5
```

Keyboard controls are I/K forward/backward, J/L left/right turn, Space stop,
R recurrent-state reset, and X/Esc exit. Hold-to-move is the default; use
`--keyboard-mode latch` only when persistent commands are desired. Every
recognized press prints `KEY_EVENT`. The demo prefers the FusionCamSensor on
the spawned Go1 and creates a temporary fallback camera only when necessary.
Use `--keep-actor` to retain the spawned robot for inspection.

The checkpoint was trained for parkour primitives, not a dedicated continuous
staircase curriculum. Validate jump/leap geometry first. Stairs, ledges, and
platforms should be exported to MuJoCo with detailed StaticMesh/Box collision;
a coarse Landscape heightfield can erase treads or create unsupported gaps.
