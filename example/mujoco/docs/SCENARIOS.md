# Multi-robot MuJoCo scenarios

These demonstrations still use the generic batch control command directly.
They will migrate to a Gym batch environment once `UnrealCvMujocoBatchEnv` is
implemented.

## MicroDuck SchoolGym

Open `/Game/SchoolGym/Maps/SchoolGymDay`, start PIE or the matching packaged
map, and run:

```cmd
python .\example\mujoco\scenarios\schoolgym.py --host 127.0.0.1 --port 9000 --ducks 7 --vx 0.36 --duration 0
```

Seven official policies demonstrate walking, standing/body pose, sit/stand,
ground pick, left kick, right kick, and roulade around the four placed balls.
The script reuses those balls and removes only robots it spawned.

## DowntownWest mixed showcase

The placed balls must be named `StaticMeshActor_4`, `StaticMeshActor_6`, and
`StaticMeshActor_8`:

```cmd
python .\example\mujoco\scenarios\downtown_showcase.py --host 127.0.0.1 --port 9000 --duration 0
```

This runs five MicroDucks and two Go1 robots with shared dynamic-ball state and
kinematic cross-world proxies. It is a synchronized multi-world approximation,
not one monolithic seven-robot MuJoCo model.

## DowntownWest follow-the-leader

```cmd
python .\example\mujoco\scenarios\downtown_follow.py --host 127.0.0.1 --port 9000 --duration 0
```

One Go1 follows a counter-clockwise route while three MicroDucks track their
immediate predecessor using live MuJoCo root state. X, Esc, or Ctrl+C ends each
scenario and cleans up the spawned robots.
