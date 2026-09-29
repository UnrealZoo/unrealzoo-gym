#!/usr/bin/env python3
"""Train the keyboard policy's original MJLab Go1 task without pretrained weights.

This is UnrealZoo's headless MJLab backend, not the UE Actor step/reset path.
The task, rewards and PPO implementation come from pinned MJLab. Training saves
PyTorch checkpoints; optional ONNX exports retain the upstream policy metadata.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE_COMMIT = "e710cead240b4c0f6f52afaa4f4b2a22c734082c"
SOURCE_REPOSITORY = "https://github.com/mujocolab/mjlab"
TASK_ID = "Mjlab-Velocity-Flat-Unitree-Go1"


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def configure_local_runtime() -> None:
    # These apply even when the invoking shell previously enabled online logging.
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_SILENT"] = "true"
    os.environ.setdefault("MUJOCO_GL", "egl")


def inspect_source() -> dict:
    import mjlab

    package = Path(mjlab.__file__).resolve().parent
    root = package.parent.parent
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "MJLab must be installed from the pinned Git checkout so its source "
            f"can be verified; discovered package {package}"
        )
    commit = result.stdout.strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"Expected MJLab {SOURCE_COMMIT}, found {commit}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"Pinned MJLab has tracked modifications: {dirty}")
    versions = {}
    for distribution in ("mjlab", "rsl-rl-lib", "mujoco", "mujoco-warp", "warp-lang", "torch", "onnx"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "repository": SOURCE_REPOSITORY, "commit": commit,
        "source_root": str(root), "package": str(package),
        "tracked_files_clean": True, "versions": versions,
    }


def load_task(num_envs: int, seed: int):
    import mjlab.tasks  # noqa: F401 -- registers the original task.
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(TASK_ID)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    agent_cfg.seed = seed
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    agent_cfg.resume = False
    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
        raise RuntimeError("The pinned Go1 task must provide VelocityOnPolicyRunner")
    return env_cfg, agent_cfg, runner_cls


def validate_trained_policy(runner, wrapped, sample_path: Path | None = None) -> dict:
    """Reject non-finite final updates before publishing final training artifacts.

    RSL-RL 5.0.1 PPO exposes actor and critic separately. Its rollout NaN check
    runs before the optimizer update, so it cannot catch a bad final update.
    Checking a checkpoint's structure alone also cannot reject invalid weights.
    """
    import torch

    checked = {}
    for module_name in ("actor", "critic"):
        module = getattr(runner.alg, module_name)
        parameters = dict(module.named_parameters())
        if not parameters:
            raise RuntimeError(f"Final validation found no {module_name} parameters")
        buffers = dict(module.named_buffers())
        for kind, tensors in (("parameter", parameters), ("buffer", buffers)):
            for name, tensor in tensors.items():
                if not bool(torch.isfinite(tensor).all()):
                    raise RuntimeError(
                        f"Non-finite final {module_name} {kind}: {name}"
                    )
        checked[module_name] = {
            "parameter_tensors": len(parameters),
            "parameter_elements": sum(value.numel() for value in parameters.values()),
            "buffer_tensors": len(buffers),
        }

    observations = wrapped.get_observations().to(runner.device)
    observation_shapes = {}
    for name, observation in observations.items():
        if not isinstance(observation, torch.Tensor):
            raise RuntimeError(f"Unexpected non-tensor Go1 observation group: {name}")
        if not bool(torch.isfinite(observation).all()):
            raise RuntimeError(f"Non-finite final environment observation: {name}")
        observation_shapes[name] = list(observation.shape)
    if not observation_shapes:
        raise RuntimeError("Final validation found no environment observations")

    policy = runner.get_inference_policy(device=runner.device)
    with torch.inference_mode():
        actions = policy(observations, stochastic_output=False)
        values = runner.alg.critic(observations)
    if not bool(torch.isfinite(actions).all()):
        raise RuntimeError("Final deterministic actor produced non-finite actions")
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("Final critic produced non-finite values")
    if tuple(actions.shape) != (wrapped.num_envs, 12):
        raise RuntimeError(f"Unexpected final Go1 action shape: {list(actions.shape)}")
    result = {
        "status": "passed", "modules": checked,
        "observation_shapes": observation_shapes,
        "deterministic_action_shape": list(actions.shape),
        "deterministic_action_abs_max": float(actions.abs().max().item()),
        "critic_value_abs_max": float(values.abs().max().item()),
        "runtime_parity": "not_checked_here; validate in the target playback runtime",
    }
    if sample_path is not None:
        import numpy as np

        count = min(4, wrapped.num_envs)
        np.savez_compressed(
            sample_path,
            observations=observations["actor"][:count].detach().cpu().numpy(),
            actions=actions[:count].detach().cpu().numpy(),
        )
        result["policy_validation_samples"] = {
            "path": str(sample_path), "sha256": sha256(sample_path), "count": count,
            "reference": "deterministic PyTorch actor on the training device",
            "note": "Compare corresponding observation/action rows and allow device rounding",
        }
    return result


def make_checkpoint_only_runner_class(runner_cls):
    """Keep the source runner while suppressing its automatic ONNX save hook.

    Pinned VelocityOnPolicyRunner.save adds only ONNX export to its parent save.
    Calling that parent directly preserves model/optimizer/iteration and the
    environment's common_step_counter, without exporting at every checkpoint.
    The inherited exporter remains available for an explicit final export.
    """
    from mjlab.rl.runner import MjlabOnPolicyRunner

    if not issubclass(runner_cls, MjlabOnPolicyRunner):
        raise RuntimeError("Expected the pinned MJLab checkpoint runner")

    class CheckpointOnlyRunner(runner_cls):
        def save(self, path: str, infos=None) -> None:
            MjlabOnPolicyRunner.save(self, path, infos)

    return CheckpointOnlyRunner


def export_policy(runner, env, output_dir: Path, run_name: str) -> dict:
    """Perform the requested final ONNX export and reject invalid artifacts."""
    import onnx
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    runner.export_policy_to_onnx(str(output_dir), "policy.onnx")
    path = output_dir / "policy.onnx"
    policy_metadata = get_base_metadata(env, run_name)
    policy_metadata.update({
        "unrealzoo_backend": "mjlab", "mjlab_source_commit": SOURCE_COMMIT,
        "task_id": TASK_ID, "control_dt": env.step_dt,
    })
    attach_metadata_to_onnx(str(path), policy_metadata)
    model = onnx.load(path)
    onnx.checker.check_model(model)
    shapes = [
        [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in (*model.graph.input, *model.graph.output)
    ]
    if len(model.graph.input) != 1 or len(model.graph.output) != 1 or shapes != [[1, 48], [1, 12]]:
        raise RuntimeError(f"Keyboard requires ONNX [1,48] -> [1,12], found {shapes}")
    required = {"joint_names", "observation_names", "default_joint_pos", "action_scale", "joint_stiffness", "joint_damping"}
    missing = required - {item.key for item in model.metadata_props}
    if missing:
        raise RuntimeError(f"Missing keyboard policy metadata: {sorted(missing)}")
    return {"path": str(path), "sha256": sha256(path), "shapes": shapes,
            "metadata": policy_metadata}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=positive_int, default=4096)
    parser.add_argument("--iterations", type=positive_int, default=10000,
                        help="Additional PPO iterations, including when resuming")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-dir", type=Path, required=True,
                        help="New, empty run directory; never overwrites an existing run")
    parser.add_argument("--resume", type=Path,
                        help="Explicit trusted local training checkpoint; default is random initialization")
    parser.add_argument("--save-interval", type=positive_int, default=None)
    parser.add_argument("--export-onnx", action="store_true",
                        help="Also export the final policy to ONNX; default saves .pt only")
    parser.add_argument("--threads", type=positive_int, default=4,
                        help="CPU intra-op threads; does not change task/PPO configuration")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative for reproducible initialization")
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
        if not args.resume.is_file():
            parser.error(f"Checkpoint does not exist: {args.resume}")
    log_dir = args.log_dir.expanduser().resolve()
    if log_dir.exists() and any(log_dir.iterdir()):
        parser.error(f"--log-dir must be empty: {log_dir}")
    configure_local_runtime()
    source = inspect_source()
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("This headless training backend requires an available NVIDIA CUDA device")
    torch.set_num_threads(args.threads)
    configure_torch_backends()
    env_cfg, agent_cfg, runner_cls = load_task(args.num_envs, args.seed)
    agent_cfg.max_iterations = args.iterations
    agent_cfg.resume = args.resume is not None
    if args.save_interval is not None:
        agent_cfg.save_interval = args.save_interval
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))
    manifest = {
        "schema_version": 1, "status": "initializing", "backend": "mjlab",
        "ue_actor_backend": False, "render_mode": None,
        "task_id": TASK_ID, "source": source,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv if argv is None else argv,
        "seed": args.seed, "device": args.device,
        "initialization": "explicit_resume" if args.resume else "random",
        "resume_checkpoint": str(args.resume) if args.resume else None,
        "resume_sha256": sha256(args.resume) if args.resume else None,
        "external_pretrained_teacher": False,
        "onnx_export_requested": args.export_onnx,
        "logging": {"logger": "tensorboard", "upload_model": False, "wandb_mode": "disabled"},
        "num_envs": args.num_envs,
        "steps_per_env_per_iteration": agent_cfg.num_steps_per_env,
        "samples_per_iteration": args.num_envs * agent_cfg.num_steps_per_env,
        "additional_iterations_requested": args.iterations,
        "command_curriculum": {
            "counter": "common_step_counter (vector steps, not aggregate samples)",
            "original_threshold_vector_steps": [120000, 240000],
            "original_threshold_iterations_at_24_steps": [5000, 10000],
        },
        "config_sha256": {
            name: sha256(log_dir / "params" / name) for name in ("env.yaml", "agent.yaml")
        },
    }
    manifest_path = log_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    env = None
    start = time.perf_counter()
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = make_checkpoint_only_runner_class(runner_cls)(
            wrapped, asdict(agent_cfg), str(log_dir), args.device
        )
        if args.resume is not None:
            runner.load(str(args.resume), map_location=args.device)
        else:
            # Keep the exact initial network for an honest before/after evaluation.
            runner.save(str(log_dir / "model_initial.pt"))
        manifest.update(status="training", initial_iteration=int(runner.current_learning_iteration),
                        initial_common_step_counter=int(env.common_step_counter),
                        physics_dt=env.physics_dt, control_dt=env.step_dt)
        write_json(manifest_path, manifest)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        manifest["final_validation"] = {"status": "running"}
        write_json(manifest_path, manifest)
        manifest["final_validation"] = validate_trained_policy(
            runner, wrapped, log_dir / "policy_validation.npz"
        )
        write_json(manifest_path, manifest)
        final_checkpoint = log_dir / "model_final.pt"
        runner.save(str(final_checkpoint))
        exported = (
            export_policy(runner, env, log_dir, log_dir.name)
            if args.export_onnx else None
        )
        manifest.update(
            status="completed", elapsed_seconds=time.perf_counter() - start,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            final_iteration=int(runner.current_learning_iteration),
            final_common_step_counter=int(env.common_step_counter),
            final_checkpoint={"path": str(final_checkpoint), "sha256": sha256(final_checkpoint)},
            exported_policy=exported,
        )
        write_json(manifest_path, manifest)
        outputs = {"checkpoint": str(final_checkpoint), "manifest": str(manifest_path)}
        if exported is not None:
            outputs["policy"] = exported["path"]
        print(json.dumps(outputs, indent=2))
    except BaseException as error:
        if manifest.get("final_validation", {}).get("status") == "running":
            manifest["final_validation"] = {
                "status": "failed", "error": f"{type(error).__name__}: {error}"
            }
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}",
                        elapsed_seconds=time.perf_counter() - start)
        write_json(manifest_path, manifest)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
