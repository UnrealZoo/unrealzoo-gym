# Unitree G1 29-DoF velocity policy

The bundled `policy.onnx` and `deploy.yaml` come from Unitree's official
`unitree_rl_lab` repository, commit
`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`:

`deploy/robots/g1_29dof/config/policy/velocity/v0`

- Input: `obs`, shape `[1, 480]`
- Output: `actions`, shape `[1, 29]`
- Policy SHA-256: `610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f`
- License: Apache-2.0; see `LICENSE.unitree_rl_lab.txt`

After starting PIE with UnrealCV listening on port 9000, run from the
`unrealzoo-gym` repository root:

```cmd
conda run -n unrealzoo python example\mujoco\mujoco_robot_demo.py g1 keyboard --host 127.0.0.1 --port 9000
```

The demo spawns `BP_UnitreeG1` in front of Camera 0, settles it onto the
ground, and starts synchronous policy control. Use I/K for forward/back,
J/L for turn, Space to stop, and X or Escape to exit.
