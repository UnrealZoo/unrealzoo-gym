# Go1 visual Parkour

The Parkour example combines MuJoCo proprioception with depth from the Go1
FusionCamSensor. It runs the recurrent locomotion policy at 50 Hz and supplies
delayed depth at the checkpoint's 10 Hz observation rate.

```cmd
python .\example\mujoco\parkour\demo.py --host 127.0.0.1 --port 9000 --camera-id auto --vx 1.0 --command-ramp 0
```

`--camera-id auto` first uses the sensor attached to the acquired Go1. If no
sensor is available, the demo creates and later removes a temporary camera.
Use `--preserve-attached-camera-pose` to keep Blueprint-authored extrinsics.

Controls are I/K for forward/backward, J/L for turning, R to reset recurrent
state, and X/Esc to exit. Disable the live raw/processed depth dashboard with
`--no-visualize-observation`. Diagnostic captures are stored under
`artifacts/mujoco/parkour_observation_debug`.

Offline checkpoint test:

```cmd
python .\example\mujoco\parkour\demo.py --self-test
```

The example adapts [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour)
at commit `5e1f4136f7fec9dac2741add63ca58d32090666f`. Provenance and license files are
under `policies/parkour/go1`; the required upstream inference runtime is
isolated under `third_party/robot_parkour_rsl_rl`.

This checkpoint was trained for discrete Parkour obstacles. Arbitrary stairs
also depend on matching camera extrinsics, depth scale and latency, collision
fidelity, joint ordering, and the original training distribution.
