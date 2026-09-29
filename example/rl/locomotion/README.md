# Go1 PPO training inside UnrealZoo

`train_go1_ue.py` learns from **MuJoCo transitions computed inside Unreal Engine**.
Fresh runs randomly initialize the actor and critic. There is no teacher,
pretrained initialization or distillation. `--resume` explicitly continues a
trusted checkpoint produced by this UE trainer.

The PPO recipe comes from [MJLab commit
e710cead240b4c0f6f52afaa4f4b2a22c734082c](https://github.com/mujocolab/mjlab/tree/e710cead240b4c0f6f52afaa4f4b2a22c734082c).
Reading its configuration does not load pretrained weights or run its simulator.
Adapted formulas retain their attribution and [Apache license](LICENSE.mjlab).

## Framework

```text
RSL-RL PPO / actor / critic (CUDA or CPU)
                 |
           UEGo1VecEnv
            /       \
 Go1Task (NumPy)    UEGo1Pool -> UEWorker connections
 observations,              -> synchronous UnrealCV RPC
 commands, reward,          -> Go1 components in UE
 termination                -> independent MuJoCo model/data per robot (CPU)
```

The pool pauses the UE world so Actor ticks cannot advance physics between
requests. Each synchronous action advances the selected models by exactly 20 ms
(four 5 ms substeps). Robots reset independently. Batched RPC and parallel CPU
stepping reduce overhead. **NullRHI removes rendering; physics still runs on CPU.**
The policy outputs 12 joint actions.

| Entry | Purpose |
| --- | --- |
| `train_go1_ue.py` | Direct UE PPO, checkpoint/resume and learning budgets |
| `benchmark_go1_ue.py`, `benchmark_go1_reset.py` | Sampling capacity and reset costs |
| `evaluate_go1_ue.py` | Numerical fixed-command evaluation in local PIE |
| `play_go1.py`, `record_go1.py` | Explicit .pt/.onnx playback and video |
| `run_ue_scale_suite.py` | Sequential capacity/short PPO checks on a Linux package |
| `train_go1.py`, `evaluate_go1.py` | Separate native MJLab/MuJoCo Warp baseline |

The reusable `gym_unrealcv.make_mujoco_vector_env()` API supports Go1, G1 and
MicroDuck across processes/agents. It returns raw observations, zero reward and
no fall termination. Go1 PPO uses its own pool and task wrapper; these are
different interfaces.

## Plugin and task compatibility

Build the matching UnrealCV plugin in your UE5.7 project before using v2.
Older released binaries may lack its endpoints. Installing Python code does not
update the plugin embedded in a packaged environment.

| Contract | `ue_v310_velocity` (default) | `ue_keyboard_flat_v2` (explicit) |
| --- | --- | --- |
| Actor / critic dimensions | 48 / 48 | 48 / 72 |
| Telemetry | Legacy observations | Foot heights, air/contact times, world contact forces |
| Rewards | Reduced tracking/posture/action/slip recipe | Also joint limits, clearance, swing and landing |
| Heading commands / randomized reset poses | Disabled | Enabled with capability checks |
| Applied normalized action | Legacy contract | Clipped to ±5; PPO keeps the sample for log probability |
| Physics | Existing runtime | Explicit nominal actuator/model profile, validated against MJCF |

V2 checks training profile, telemetry, reset-pose and model capabilities.
The optional fast path requires `policy_step_train_v2` (117-column binary
state). Missing capabilities fail explicitly rather than padding observations.

This is a **static-scene** mode. Shared dynamic props, robot-to-robot collisions
and visual RL are outside its scope. Collision geometry, domain randomization
(friction, encoder, COM and pushes) and sensor timing differ from native MJLab.
Root sensing can lag one 5 ms substep; height sensing differs from the source's
cached-ray phase. Matching the recipe does not establish native task equivalence
or native-quality gait.

## Install

Use an isolated Python 3.11 environment. The lightweight UE runtime needs a clean
checkout of the pinned MJLab source for configuration, but does not require
MJLab, MuJoCo Warp or another native simulator as installed Python packages.

Windows PowerShell, from this repository; replace paths with your own:

```powershell
py -3.11 -m venv E:/unrealzoo-runtime/go1-ppo
$python = 'E:/unrealzoo-runtime/go1-ppo/Scripts/python.exe'
& $python -m pip install --upgrade pip
# NVIDIA CUDA 12.8 wheels; for CPU use the /whl/cpu index instead.
& $python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r example/rl/locomotion/requirements-ue.txt
& $python -m pip check
git clone https://github.com/mujocolab/mjlab.git E:/unrealzoo-runtime/mjlab
git -C E:/unrealzoo-runtime/mjlab checkout --detach e710cead240b4c0f6f52afaa4f4b2a22c734082c
$env:UNREALZOO_MJLAB_SOURCE = 'E:/unrealzoo-runtime/mjlab'
& $python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use `--device cpu` when CUDA is unavailable. Run the scripts from the source
tree. The base package's optional `mujoco-training` extra alone is not this
pinned PPO runtime. General Gym/vector examples also need the base package.

On Linux/macOS, a Python 3.11 venv plus the same requirements and pinned source
also provides the lightweight runtime; select the appropriate PyTorch wheel.
`setup_ue_macos.sh` provisions an Apple Silicon CPU runtime.
`setup_ue.sh` / `setup_mjlab.sh` provision the fuller Linux x86-64 NVIDIA
runtime, including native baseline dependencies. Override source/runtime/venv
paths using the variables documented in those scripts. The native installer
records its replacement of an unavailable MuJoCo nightly with stable 3.7.0,
without changing the upstream lock.

## Start UE and smoke-test

On Windows, open the rebuilt UE5.7 project, load a static map with sufficient
ground coverage, and start **in-process Play in Editor**. Enable UnrealCV on a
known local port (9000 below). The trainer owns its training actors, not the
editor process. Do not connect another trainer or change maps while it owns the
pool.

```powershell
& $python example/rl/locomotion/train_go1_ue.py --connect 127.0.0.1:9000 --mjlab-source $env:UNREALZOO_MJLAB_SOURCE --task-profile ue_keyboard_flat_v2 --agents-per-process 2 --iterations 2 --seed 42 --device cuda:0 --log-dir runs/go1/ue-smoke
```

On Linux, use `--ue-binary /path/to/UnrealZoo_UE5_7` instead of `--connect`
to launch owned NullRHI processes. Total replicas are
`num-processes * agents-per-process`; an attached PIE uses one process.
Owned processes have port checks and ownership-based cleanup. Attached-session
cleanup restores the previous pause state.

Two updates check integration, not convergence. Use the benchmark entries'
`--help` and measure stepping **and resets** before increasing capacity.
Zero-action throughput is not real PPO throughput. Reserve disk and memory for
initialization, reset models, checkpoints and logs.

## Training configuration

The source actor/critic are 512/256/128 ELU MLPs; PPO uses five epochs and four
minibatches. Default rollout length is 24 unless overridden. Episodes last up to
20 seconds and terminate beyond 70 degrees of tilt. The manifest records the
effective task and optimizer settings.

An explicit larger-batch experiment, after capacity checks:

```powershell
& $python example/rl/locomotion/train_go1_ue.py --connect 127.0.0.1:9000 --mjlab-source $env:UNREALZOO_MJLAB_SOURCE --task-profile ue_keyboard_flat_v2 --agents-per-process 512 --rollout-steps 192 --curriculum-step-scale 8 --training-budget-seconds 43200 --iterations 1000000 --save-interval 25 --seed 42 --device cuda:0 --log-dir runs/go1/ue-large-batch
```

512 × 192 gives 98,304 transitions/update. It is not equivalent to 4,096
independent environments × 24 steps: correlation and GAE horizons differ.
Curriculum scale multiplies vector-step thresholds, not the physics time step.

| Option | Meaning |
| --- | --- |
| `--training-budget-seconds` | Learning-loop wall time; stops after a complete update, excludes initialization/export |
| `--iterations` | Additional updates for this invocation, including on resume |
| `--rollout-steps` | Explicit steps per environment per update |
| `--curriculum-step-scale` | Sampling/heading curriculum threshold multiplier |
| `--command-sampling source` | Original distribution |
| `--command-sampling axis_balanced_v1` | Standing 10%, six single axes 50%, mixed 40% |
| `--command-sampling axis_mixed_v2` | Standing 10%, single axes 30%, mixed 60% |
| `--linear-tracking-reward precision_v1` | Equal mix of exp(-e²/0.25) and exp(-e²/0.0625), retaining peak and weight |
| `--step-mode fast` | Compact binary sampling with deferred render-pose sync |
| `--reset-mode auto` | Reuse cached state for matching spawn/configuration |
| `--reset-mode rebuild` | Regenerate model/collision, including after terrain edits |

Sampling/reward variants are controlled experiments, not guaranteed improvements.
Both defaults remain `source`.

## Checkpoints and continuation

Repeat the original task, rollout, curriculum and sampling/reward options and
use a **new output directory**:

```powershell
& $python example/rl/locomotion/train_go1_ue.py --connect 127.0.0.1:9000 --mjlab-source $env:UNREALZOO_MJLAB_SOURCE --task-profile ue_keyboard_flat_v2 --agents-per-process 512 --rollout-steps 192 --curriculum-step-scale 8 --resume runs/go1/ue-large-batch/model_final.pt --training-budget-seconds 7200 --iterations 1000000 --log-dir runs/go1/ue-continuation
```

Resume restores model, optimizer, adaptive learning rate and curriculum counter.
Changing sampling/reward requires `--allow-command-sampling-change` /
`--allow-linear-tracking-reward-change` respectively. These flags do not
override other checkpoint incompatibilities.

Outputs include initial/periodic/final .pt files, configurations,
`run_manifest.json`, iteration metrics and diagnostics. Fresh runs save their
initial model before learning. ONNX export is optional (`--export-onnx`).
Finite-value guards stop invalid updates. Failure dumps are diagnostic, not
automatic resume candidates. The final checkpoint is not automatically best.

## Evaluation and video

Name a policy explicitly. Validate its contract before connecting:

```powershell
& $python example/rl/locomotion/play_go1.py --checkpoint runs/go1/ue-large-batch/model_final.pt --verify-only
& $python example/rl/locomotion/evaluate_go1_ue.py --connect 127.0.0.1:9000 --checkpoint runs/go1/ue-large-batch/model_final.pt --output runs/go1/eval-a.json --spawn 250 40 35 --spawn-rotation 0 5 0 --episode-seconds 20
```

Choose map-appropriate coordinates (UE cm; pitch/yaw/roll degrees). The example
pose is not safe on every map. Repeat at new positions/orientations. The evaluator
reports standing, forward, backward, positive lateral and positive yaw commands,
all-axis error, falls, contacts and slip. UE v2 checkpoints automatically select
v2 physics; select the same `--environment-profile ue_keyboard_flat_v2` when
comparing an explicit ONNX reference.

Also test opposite lateral/yaw directions and command changes using playback or
an explicit evaluation driver. Freeze candidates after screening, then assess
new spawn poses and command sequences. Survival, reward growth or one fixed
command passing is insufficient evidence of robust locomotion.

```powershell
& $python example/rl/locomotion/play_go1.py --checkpoint runs/go1/ue-large-batch/model_final.pt --host 127.0.0.1 --port 9000 --steps 500 --command-vx 0 --command-vy -0.3 --command-yaw 0
& $python example/rl/locomotion/record_go1.py --checkpoint runs/go1/ue-large-batch/model_final.pt --host 127.0.0.1 --port 9000 --output runs/go1/demo.mp4 --fps 25 --size 960 540
```

Recording needs rendering and FFmpeg (or imageio-ffmpeg). The default sequence
covers standing, forward, turning, backward and stopping. Capture must not
advance physics; video time follows simulation time. The updated plugin refreshes
paused-world bone poses and calibrates the body frame. Render correctness and
numerical policy quality are separate checks. Frames/metadata remain local.

## Tests and output hygiene

From the repo root, set the pinned source variable and run:

```powershell
& $python -B -m unittest discover -s tests -p 'test_*go1*.py'
```

Tests use synthetic states and temporary checkpoints; source-oracle integration
tests read the separately installed pinned source. Missing optional dependencies
are reported as skips. Historical training directories are not required.
Gym/vector tests additionally require the base package dependencies.

Keep models, logs, videos, generated MJCF, virtual environments and experiment
supervisors out of version control. Standard `runs/`, `artifacts/` and local
worktree directories are ignored. Review custom output paths before staging.
Publish source, portable tests, guides and required licenses; this guide ships
no trained weights or experiment results.

## Native baseline

`train_go1.py` / `evaluate_go1.py` require the full native Linux/CUDA setup
from `setup_mjlab.sh`. They run the original MJLab/MuJoCo Warp task independently
of UE collision/rendering. Native checkpoints use `evaluate_go1.py`, and are
not interchangeable with UE trainer states. See `--help` for training/resume
options. Native performance does not establish UE throughput or convergence.
