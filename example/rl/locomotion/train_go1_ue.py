#!/usr/bin/env python3
"""Train a Go1 velocity policy from real UE/MuJoCo transitions.

The pinned keyboard source supplies the PPO implementation and MLP settings.
UE supplies every physics transition. Profiles explicitly distinguish the old
reduced task from keyboard-aligned v2; neither claims identical native terrain
or full domain randomization.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

# Temporary shared utilities; none of the native backend's training loop runs.
from train_go1 import (
    SOURCE_COMMIT,
    TASK_ID,
    configure_local_runtime,
    positive_int,
    sha256,
    validate_trained_policy,
    write_json,
)
from ue_ppo_config import inspect_ppo_source, load_ppo_runner_cfg, load_source_utility
from ue_training_numerics import (
    NUMERICAL_CONTRACT, V2_PROFILE, PPONumericalGuard, applied_actions, checked_array,
)


BACKEND = "ue_mujoco"
TASK_PROFILE = "ue_v310_velocity"


def task_profile(task: Any) -> str:
    return task.task_manifest.get("profile", TASK_PROFILE)


class UEGo1VecEnv:
    """Adapt the numpy task and UE RPC pool to the RSL-RL VecEnv contract."""

    def __init__(self, pool: Any, task: Any, device: str) -> None:
        import numpy as np
        import torch
        from tensordict import TensorDict

        self._np = np
        self._torch = torch
        self._tensor_dict = TensorDict
        self.pool = pool
        self.task = task
        self.v2 = task_profile(task) == V2_PROFILE
        self.device = device
        self.num_envs = pool.num_envs
        self.num_actions = 12
        self.max_episode_length = task.max_episode_length
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=device
        )
        self.vector_steps = 0
        self.reset_episodes = 0
        self.fallen_episodes = 0
        self.timeout_episodes = 0
        self.step_seconds = 0.0
        self.reset_seconds = 0.0
        self.last_metrics: dict[str, float] = {}
        self.metric_sums: dict[str, float] = {}
        self.metric_counts: dict[str, int] = {}
        self.completed_return_sum = 0.0
        self.completed_seconds_sum = 0.0
        start = time.perf_counter()
        states = pool.reset()
        observations = task.reset(np.arange(self.num_envs), states)
        self.reset_seconds += time.perf_counter() - start
        self.initial_reset_seconds = self.reset_seconds
        self._observations = self._to_tensor_dict(observations)
        if self.v2:
            self.episode_length_buf.copy_(torch.as_tensor(
                task.initialize_episode_lengths(), dtype=torch.long, device=device
            ))
        self.cfg = copy.deepcopy(task.task_manifest)

    def _to_tensor_dict(self, observations: dict) -> Any:
        tensors = {
            key: self._torch.as_tensor(
                value, dtype=self._torch.float32, device=self.device
            )
            for key, value in observations.items()
        }
        for name, value in tensors.items():
            expected_dim = getattr(self.task, f"{name}_dim", 48)
            if tuple(value.shape) != (self.num_envs, expected_dim):
                raise ValueError(f"Unexpected {name} shape: {tuple(value.shape)}")
            if not bool(self._torch.isfinite(value).all()):
                raise ValueError(f"Non-finite {name} observations from UE")
            if self.v2:
                checked_array(observations[name], f"{name} observations", max_abs=1e4)
        if set(tensors) != {"actor", "critic"}:
            raise ValueError("UE task must supply actor and critic observations")
        return self._tensor_dict(tensors, batch_size=[self.num_envs])

    def get_observations(self) -> Any:
        return self._observations

    def restore_common_step_counter(self, value: int) -> None:
        if self.vector_steps:
            raise RuntimeError("Resume requires freshly reset UE environments")
        observations = self.task.restore_common_step_counter(value)
        self._observations = self._to_tensor_dict(observations)

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict]:
        start = time.perf_counter()
        array = actions.detach().cpu().numpy()
        if array.shape != (self.num_envs, 12) or not self._np.isfinite(array).all():
            raise ValueError("PPO supplied invalid Go1 actions")
        action_metrics = {}
        if self.v2:
            array, action_metrics = applied_actions(array)
        states = self.pool.step(array, self.task.commands)
        observations, rewards, terminated, truncated, metrics = self.task.step(
            states, array
        )
        terminated = self._np.asarray(terminated, dtype=bool)
        # A fall at the time limit is a true terminal state, not a timeout.
        truncated = self._np.asarray(truncated, dtype=bool) & ~terminated
        dones = terminated | truncated
        terminal_observations = self._to_tensor_dict(observations)
        reset_ids = self._np.flatnonzero(dones)
        if reset_ids.size:
            reset_start = time.perf_counter()
            states = self.pool.reset_indices(reset_ids.tolist())
            observations = self.task.reset(reset_ids, states)
            self.reset_seconds += time.perf_counter() - reset_start
            self._observations = self._to_tensor_dict(observations)
        else:
            self._observations = terminal_observations
        self.episode_length_buf.copy_(
            self._torch.as_tensor(
                self.task.episode_lengths,
                dtype=self._torch.long,
                device=self.device,
            )
        )
        self.vector_steps += 1
        self.reset_episodes += int(reset_ids.size)
        self.fallen_episodes += int(terminated.sum())
        self.timeout_episodes += int(truncated.sum())
        self.step_seconds += time.perf_counter() - start
        logs = dict(metrics.get("log", {}))
        logs.update(action_metrics)
        logs.update({
            str(key): float(value)
            for key, value in metrics.items()
            if self._np.isscalar(value)
        })
        if "tracking_mae" in metrics:
            for index, axis in enumerate(("vx", "vy", "yaw")):
                logs[f"Tracking/{axis}_mae"] = float(metrics["tracking_mae"][index])
        completed = metrics.get("completed_episodes", [])
        self.completed_return_sum += sum(float(x["return"]) for x in completed)
        self.completed_seconds_sum += sum(float(x["seconds"]) for x in completed)
        self.last_metrics = {key: float(value) for key, value in logs.items()}
        for key, value in self.last_metrics.items():
            self.metric_sums[key] = self.metric_sums.get(key, 0.0) + value
            self.metric_counts[key] = self.metric_counts.get(key, 0) + 1
        extras = {
            "time_outs": self._torch.as_tensor(
                truncated, dtype=self._torch.bool, device=self.device
            ),
            "terminal_observations": terminal_observations,
            "log": dict(self.last_metrics),
        }
        reward_tensor = self._torch.as_tensor(
            rewards, dtype=self._torch.float32, device=self.device
        )
        if not bool(self._torch.isfinite(reward_tensor).all()):
            raise ValueError("Non-finite float32 rewards from UE task")
        return (
            self._observations,
            reward_tensor,
            self._torch.as_tensor(dones, dtype=self._torch.long, device=self.device),
            extras,
        )

    def metrics(self, include_pool: bool = True) -> dict:
        transitions = self.vector_steps * self.num_envs
        return {
            "vector_steps": self.vector_steps,
            "transitions": transitions,
            "step_seconds_including_auto_reset": self.step_seconds,
            "reset_seconds_including_initial_reset": self.reset_seconds,
            "initial_reset_seconds": self.initial_reset_seconds,
            "auto_reset_seconds": self.reset_seconds - self.initial_reset_seconds,
            "auto_reset_fraction_of_environment_step_time": (
                (self.reset_seconds - self.initial_reset_seconds) / self.step_seconds
                if self.step_seconds else None
            ),
            "environment_transitions_per_second_including_auto_reset": (
                transitions / self.step_seconds if self.step_seconds else None
            ),
            "completed_episodes": self.reset_episodes,
            "fallen_episodes": self.fallen_episodes,
            "timeout_episodes": self.timeout_episodes,
            "mean_completed_episode_return": (
                self.completed_return_sum / self.reset_episodes
                if self.reset_episodes else None
            ),
            "mean_completed_episode_seconds": (
                self.completed_seconds_sum / self.reset_episodes
                if self.reset_episodes else None
            ),
            "fall_fraction_completed_episodes": (
                self.fallen_episodes / self.reset_episodes
                if self.reset_episodes
                else None
            ),
            "latest_task_metrics": self.last_metrics,
            "mean_logged_task_metrics": {
                key: value / self.metric_counts[key]
                for key, value in self.metric_sums.items()
            },
            "metric_averaging": "mean over vector steps reporting each metric",
            **({"pool": self.pool.metrics()} if include_pool else {}),
        }

    def iteration_metrics(self) -> dict:
        # Static per-robot XML/configuration belongs in the run manifest, not in
        # every iteration (the old 256-robot run produced 650 MB of duplicate JSON).
        return self.metrics(include_pool=False)

    def close(self) -> None:
        self.pool.close()


def prepare_runner_cfg(agent_cfg: Any) -> dict:
    """Match the pinned MJLab runner's removal of absent optional model fields."""
    cfg = asdict(agent_cfg)
    for name in ("actor", "critic"):
        model = cfg[name]
        for optional in ("cnn_cfg", "distribution_cfg"):
            if model.get(optional) is None:
                model.pop(optional, None)
        if model.get("rnn_type") is None:
            for optional in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                model.pop(optional, None)
    return cfg


def checkpoint_learning_rate(saved: dict) -> float:
    """Recover the scalar PPO schedule state from its saved optimizer groups."""
    groups = saved.get("optimizer_state_dict", {}).get("param_groups", [])
    if not groups:
        raise ValueError("Resume checkpoint has no optimizer learning rate")
    rates = [group.get("lr") for group in groups]
    if any(isinstance(rate, bool) or not isinstance(rate, (int, float))
           or not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Resume checkpoint optimizer learning rates must be positive and finite")
    if any(rate != rates[0] for rate in rates[1:]):
        raise ValueError("UE PPO requires one shared optimizer learning rate")
    # Existing UE checkpoints already carry the authoritative optimizer rate.
    # New checkpoints also record the adaptive scheduler's scalar explicitly.
    declared = (saved.get("infos") or {}).get("ppo_learning_rate", rates[0])
    if isinstance(declared, bool) or declared != rates[0]:
        raise ValueError("Checkpoint PPO learning rate differs from its optimizer")
    return float(rates[0])


def make_runner_class() -> type:
    import torch
    from rsl_rl.runners import OnPolicyRunner

    class LearningBudgetReached(Exception):
        """Internal stop at a completed rollout/update boundary."""

    class UEOnPolicyRunner(OnPolicyRunner):
        """Original PPO loop with UE checkpoint provenance and timing records."""

        def __init__(
            self, env: Any, train_cfg: dict, log_dir: str, device: str = "cpu"
        ) -> None:
            # RSL consumes parts of its configuration during construction. Keep
            # the exact actor recipe so every checkpoint can also run inference.
            self.inference_actor_cfg = copy.deepcopy(train_cfg["actor"])
            self.inference_actor_class = self.inference_actor_cfg.pop("class_name")
            self.inference_obs_groups = {
                "actor": list(train_cfg["obs_groups"]["actor"])
            }
            super().__init__(env, train_cfg, log_dir, device)
            self.numerical_guard = (
                PPONumericalGuard(self.alg, Path(log_dir))
                if task_profile(env.task) == V2_PROFILE else None
            )
            self.completed_iterations = 0
            self.total_collection_seconds = 0.0
            self.total_update_seconds = 0.0
            self.learning_seconds_this_invocation = 0.0
            self.training_budget_seconds = None
            self.training_stop_reason = None
            self._learning_started = None
            original_log = self.logger.log

            def log_iteration(**values: Any) -> None:
                self.completed_iterations = int(values["it"]) + 1
                record = {
                    "completed_iterations": self.completed_iterations,
                    "collection_seconds": float(values["collect_time"]),
                    "update_seconds": float(values["learn_time"]),
                    "learning_rate": float(values["learning_rate"]),
                    "action_std_mean": float(values["action_std"].detach().mean().cpu()),
                    "losses": {
                        key: float(value)
                        for key, value in values["loss_dict"].items()
                    },
                    "environment": getattr(self.env, "iteration_metrics", self.env.metrics)(),
                    "numerics": self.numerical_guard.maxima.copy() if self.numerical_guard else None,
                    "minibatch_kl": self.numerical_guard.update_kl.copy() if self.numerical_guard else None,
                }
                self.total_collection_seconds += record["collection_seconds"]
                self.total_update_seconds += record["update_seconds"]
                original_log(**values)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                record["learning_seconds_this_invocation"] = self.learning_elapsed()
                path = Path(self.logger.log_dir) / "iteration_metrics.jsonl"
                with path.open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                if (self.training_budget_seconds is not None
                        and self.learning_elapsed() >= self.training_budget_seconds):
                    raise LearningBudgetReached()

            self.logger.log = log_iteration

        def learning_elapsed(self) -> float:
            if self._learning_started is None:
                return self.learning_seconds_this_invocation
            return time.perf_counter() - self._learning_started

        def learn(
            self, num_learning_iterations: int, init_at_random_ep_len: bool = False,
            training_budget_seconds: float | None = None,
        ) -> None:
            if training_budget_seconds is not None and (
                not math.isfinite(training_budget_seconds) or training_budget_seconds <= 0
            ):
                raise ValueError("Training budget must be positive and finite")
            self.training_budget_seconds = training_budget_seconds
            self.training_stop_reason = "iteration_limit"
            self._learning_started = time.perf_counter()
            try:
                try:
                    super().learn(num_learning_iterations, init_at_random_ep_len)
                except LearningBudgetReached:
                    # The unchanged RSL loop already completed collection, returns,
                    # optimizer update and logging. No partial rollout is committed.
                    self.training_stop_reason = "budget"
                    if self.logger.writer is not None:
                        self.save(str(Path(self.logger.log_dir) / "model_budget.pt"))
                        self.logger.stop_logging_writer()
            except BaseException as error:
                if self.numerical_guard is not None:
                    try:
                        self.numerical_guard.record_failure(error, self.env)
                    except Exception as diagnostic_error:
                        print(f"Numerical diagnostic save failed: {diagnostic_error}", file=sys.stderr)
                raise
            finally:
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                self.learning_seconds_this_invocation = self.learning_elapsed()
                self._learning_started = None

        def save(self, path: str, infos: dict | None = None) -> None:
            saved = self.alg.save()
            saved["iter"] = self.current_learning_iteration
            saved["infos"] = {
                **(infos or {}),
                "backend": BACKEND,
                "task_profile": task_profile(self.env.task),
                "numerical_contract": NUMERICAL_CONTRACT if task_profile(self.env.task) == V2_PROFILE else None,
                "source_commit": SOURCE_COMMIT,
                "completed_iterations": self.completed_iterations,
                "learning_seconds_this_invocation": self.learning_elapsed(),
                "ppo_learning_rate": float(self.alg.learning_rate),
                "common_step_counter": int(self.env.task.common_step_counter),
                "task_manifest": self.env.task.task_manifest,
                "inference": {
                    "schema_version": 1,
                    "rsl_rl_version": metadata.version("rsl-rl-lib"),
                    "actor_class": self.inference_actor_class,
                    "actor_cfg": self.inference_actor_cfg,
                    "obs_groups": self.inference_obs_groups,
                    "obs_set": "actor",
                    "observation_dim": 48,
                    "action_dim": 12,
                    "policy_metadata": make_policy_metadata(
                        self.env.task, Path(self.logger.log_dir)
                    ),
                },
            }
            checkpoint = Path(path)
            temporary = checkpoint.with_name(checkpoint.name + ".tmp")
            torch.save(saved, temporary)
            temporary.replace(checkpoint)

        def load(
            self,
            path: str,
            load_cfg: dict | None = None,
            strict: bool = True,
            map_location: str | None = None,
        ) -> dict:
            saved = torch.load(path, weights_only=True, map_location=map_location)
            infos = saved.get("infos") or {}
            expected = {
                "backend": BACKEND,
                "task_profile": task_profile(self.env.task) if hasattr(self.env, "task") else TASK_PROFILE,
                "source_commit": SOURCE_COMMIT,
            }
            for key, value in expected.items():
                if infos.get(key) != value:
                    raise ValueError(f"Resume checkpoint has incompatible {key}")
            saved_task = infos.get("task_manifest") or {}
            saved_scale = saved_task.get("curriculum_step_scale", 1)
            current_scale = getattr(getattr(self.env, "task", None), "curriculum_step_scale", 1)
            if saved_scale != current_scale:
                raise ValueError("Resume checkpoint has incompatible curriculum_step_scale")
            validate_command_sampling_resume(
                saved_task.get("command_sampling", "source"),
                getattr(getattr(self.env, "task", None), "command_sampling", "source"),
                getattr(self, "allow_command_sampling_change", False),
            )
            validate_linear_tracking_reward_resume(
                saved_task.get("linear_tracking_reward", "source"),
                getattr(getattr(self.env, "task", None), "linear_tracking_reward", "source"),
                getattr(self, "allow_linear_tracking_reward_change", False),
            )
            if expected["task_profile"] == V2_PROFILE and infos.get("numerical_contract") != NUMERICAL_CONTRACT:
                raise ValueError("Resume checkpoint has incompatible numerical/action contract")
            restore_optimizer = load_cfg is None or bool(load_cfg.get("optimizer", False))
            learning_rate = checkpoint_learning_rate(saved) if restore_optimizer else None
            load_iteration = self.alg.load(saved, load_cfg, strict)
            if learning_rate is not None:
                # RSL-RL 5.0.1 restores the optimizer groups but leaves this
                # scalar at its constructor value. The next adaptive-KL update
                # would otherwise overwrite the restored groups with that rate.
                self.alg.learning_rate = learning_rate
            if load_iteration:
                self.completed_iterations = int(infos["completed_iterations"])
                self.current_learning_iteration = self.completed_iterations
                self.env.restore_common_step_counter(int(infos["common_step_counter"]))
            return infos

    return UEOnPolicyRunner


def make_policy_metadata(task: Any, output_dir: Path) -> dict:
    return {
        **task.policy_metadata,
        "run_path": output_dir.name,
        "unrealzoo_backend": BACKEND,
        "mjlab_source_commit": SOURCE_COMMIT,
        "task_id": task_profile(task),
        "control_dt": "0.02",
    }


def export_policy(runner: Any, task: Any, output_dir: Path) -> dict:
    import onnx
    import torch

    # Match MJLab's legacy ONNX exporter, avoiding Torch 2.9 dynamo defaults.
    model = runner.alg.get_policy().as_onnx(verbose=False).cpu().eval()
    path = output_dir / "policy.onnx"
    torch.onnx.export(
        model,
        model.get_dummy_inputs(),
        str(path),
        export_params=True,
        opset_version=18,
        input_names=model.input_names,
        output_names=model.output_names,
        dynamic_axes={},
        dynamo=False,
    )
    policy_metadata = make_policy_metadata(task, output_dir)
    exported = onnx.load(path)
    for key, value in policy_metadata.items():
        entry = exported.metadata_props.add()
        entry.key, entry.value = str(key), str(value)
    onnx.checker.check_model(exported)
    shapes = [
        [dimension.dim_value for dimension in item.type.tensor_type.shape.dim]
        for item in (*exported.graph.input, *exported.graph.output)
    ]
    if shapes != [[1, 48], [1, 12]]:
        raise ValueError(f"Unexpected exported policy shapes: {shapes}")
    onnx.save(exported, path)
    return {"path": str(path), "sha256": sha256(path), "metadata": policy_metadata}


def validate_command_sampling_resume(saved, current, allow_change=False):
    allowed = ("source", "axis_balanced_v1", "axis_mixed_v2")
    if saved not in allowed or current not in allowed:
        raise ValueError("Unknown saved/current command sampling")
    if saved != current and not allow_change:
        raise ValueError("Resume command sampling changed without explicit experiment opt-in")


def validate_linear_tracking_reward_resume(saved, current, allow_change=False):
    if saved not in ("source", "precision_v1") or current not in ("source", "precision_v1"):
        raise ValueError("Unknown saved/current linear tracking reward")
    if saved != current and not allow_change:
        raise ValueError("Resume linear tracking reward changed without explicit experiment opt-in")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    connection = parser.add_mutually_exclusive_group(required=True)
    connection.add_argument("--ue-binary", type=Path, help="Launch and own a Linux packaged UE process")
    connection.add_argument("--connect", help="Attach to existing local UE PIE at HOST:PORT; never owns or exits UE")
    parser.add_argument("--mjlab-source", type=Path, help="Pinned MJLab Git checkout; read for PPO configuration without importing its simulator")
    parser.add_argument("--num-processes", type=positive_int, default=1)
    parser.add_argument("--agents-per-process", type=positive_int, default=1)
    parser.add_argument("--base-port", type=int, default=19200)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument(
        "--training-budget-seconds", type=float,
        help="Stop after this much learning-loop wall time at a full PPO iteration; excludes initialization and final evaluation/export",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-profile", choices=(TASK_PROFILE, V2_PROFILE), default=TASK_PROFILE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument(
        "--reset-mode", choices=("auto", "rebuild"), default="auto",
        help="Reuse matching terrain/model state automatically, or always rebuild collision",
    )
    parser.add_argument("--save-interval", type=positive_int, default=50)
    parser.add_argument("--rollout-steps", type=positive_int, default=None,
                        help="Explicit experiment override of source's 24 steps per environment")
    parser.add_argument("--curriculum-step-scale", type=positive_int, default=1,
                        help="Multiply command curriculum thresholds; default retains source vector-step clock")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Explicit trusted UE checkpoint")
    parser.add_argument("--command-sampling", choices=("source", "axis_balanced_v1", "axis_mixed_v2"), default="source")
    parser.add_argument("--allow-command-sampling-change", action="store_true",
                        help="Explicit continuation experiment; restores weights/optimizer but changes command distribution")
    parser.add_argument("--linear-tracking-reward", choices=("source", "precision_v1"), default="source")
    parser.add_argument("--allow-linear-tracking-reward-change", action="store_true",
                        help="Explicit reward experiment; restore own weights/optimizer while changing the linear tracking kernel")
    parser.add_argument(
        "--export-onnx", action="store_true",
        help="Also export ONNX; PyTorch checkpoints are always saved and directly playable",
    )
    parser.add_argument("--disable-observation-noise", action="store_true")
    parser.add_argument("--runtime-diagnostics", choices=("enabled", "disabled"), default="disabled",
                        help="Request per-step UE diagnostic writes; old servers report control unavailable")
    parser.add_argument("--step-mode", choices=("auto", "single", "batch_serial", "batch_parallel", "fast"), default="auto",
                        help="auto keeps full-state compatibility; fast requires binary training API and skips per-step Actor pose sync")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    durations = (args.episode_seconds, args.request_timeout)
    if args.training_budget_seconds is not None:
        durations += (args.training_budget_seconds,)
    if args.seed < 0 or any(not math.isfinite(x) or x <= 0 for x in durations):
        parser.error("Seed must be non-negative; durations must be positive")
    if not 1024 <= args.base_port <= 65535 - args.num_processes + 1:
        parser.error("UE port range must lie between 1024 and 65535")
    if args.connect is not None:
        from ue_go1_env import parse_connect
        try:
            parse_connect(args.connect)
        except ValueError as error:
            parser.error(str(error))
        if args.num_processes != 1:
            parser.error("--connect requires --num-processes 1")
    else:
        args.ue_binary = args.ue_binary.expanduser().resolve(strict=True)
    if args.resume:
        args.resume = args.resume.expanduser().resolve(strict=True)
    log_dir = args.log_dir.expanduser().resolve()
    if log_dir.exists() and any(log_dir.iterdir()):
        parser.error(f"--log-dir must be empty: {log_dir}")
    configure_local_runtime()
    source = inspect_ppo_source(args.mjlab_source)
    import numpy as np
    import torch
    from ue_go1_env import UEGo1Pool
    from ue_go1_task import Go1Task

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        load_source_utility(source, "src/mjlab/utils/torch.py").configure_torch_backends()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("Requested CUDA device is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    num_envs = args.num_processes * args.agents_per_process
    agent_cfg = load_ppo_runner_cfg(source)
    agent_cfg.seed = args.seed
    agent_cfg.logger = "tensorboard"
    agent_cfg.upload_model = False
    agent_cfg.max_iterations = args.iterations
    agent_cfg.save_interval = args.save_interval
    agent_cfg.resume = args.resume is not None
    source_rollout_steps = agent_cfg.num_steps_per_env
    if args.rollout_steps is not None:
        agent_cfg.num_steps_per_env = args.rollout_steps
    task = Go1Task(
        num_envs,
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        observation_noise=not args.disable_observation_noise,
        profile=args.task_profile,
        curriculum_step_scale=args.curriculum_step_scale,
        command_sampling=args.command_sampling,
        linear_tracking_reward=args.linear_tracking_reward,
    )
    runner_cfg = prepare_runner_cfg(agent_cfg)
    log_dir.mkdir(parents=True, exist_ok=True)
    load_source_utility(source, "src/mjlab/utils/os.py").dump_yaml(
        log_dir / "params" / "agent.yaml", runner_cfg
    )
    write_json(log_dir / "params" / "ue_task.json", task.task_manifest)
    manifest = {
        "schema_version": 1,
        "status": "initializing",
        "backend": BACKEND,
        "physics_sampling": "UE Actor MuJoCo synchronous RPC only",
        "render_mode": "existing_editor_settings" if args.connect else "NullRHI",
        "connection_mode": "attach" if args.connect else "launch",
        "connect_endpoint": args.connect,
        "owns_ue_process": not bool(args.connect),
        "source": source,
        "source_task": TASK_ID,
        "task_profile": args.task_profile,
        "numerical_contract": NUMERICAL_CONTRACT if args.task_profile == V2_PROFILE else None,
        "task": task.task_manifest,
        "initialization": "explicit_resume" if args.resume else "random",
        "resume_checkpoint": str(args.resume) if args.resume else None,
        "resume_sha256": sha256(args.resume) if args.resume else None,
        "resume_semantics": "network/optimizer/curriculum; UE episodes restart",
        "allow_command_sampling_change": args.allow_command_sampling_change,
        "allow_linear_tracking_reward_change": args.allow_linear_tracking_reward_change,
        "external_pretrained_teacher": False,
        "onnx_export_requested": args.export_onnx,
        "default_policy_artifact": "model_final.pt",
        "num_envs": num_envs,
        "num_processes": args.num_processes,
        "agents_per_process": args.agents_per_process,
        "reset_mode": args.reset_mode,
        "runtime_diagnostics_requested": args.runtime_diagnostics,
        "step_mode_requested": args.step_mode,
        "additional_iterations_requested": args.iterations,
        "training_budget_seconds_requested": args.training_budget_seconds,
        "training_budget_scope": "Learning loop including sampling, auto-reset, PPO, logging and periodic checkpoints; excludes initialization, final validation/export and cleanup; stops at a complete iteration",
        "steps_per_env_per_iteration": agent_cfg.num_steps_per_env,
        "sampling_experiment": {
            "source_rollout_steps": source_rollout_steps,
            "rollout_steps_override": args.rollout_steps,
            "batch_transitions": num_envs * agent_cfg.num_steps_per_env,
            "curriculum_step_scale": args.curriculum_step_scale,
            "note": "Longer trajectories match batch count, not independent environment diversity or GAE horizon",
        },
        "seed": args.seed,
        "device": args.device,
        "configuration_sha256": {
            "agent.yaml": sha256(log_dir / "params" / "agent.yaml"),
            "ue_task.json": sha256(log_dir / "params" / "ue_task.json"),
        },
        "implementation_sha256": {
            name: sha256(Path(__file__).resolve().parent / name)
            for name in ("train_go1_ue.py", "ue_go1_env.py", "ue_go1_task.py", "train_go1.py",
                         "ue_training_numerics.py", "ue_go1_checkpoint.py")
        },
        "timeout_bootstrap": "unchanged RSL-RL 5.0.1: gamma * V(s_t) on time_outs",
        "initial_episode_length_randomization": args.task_profile == V2_PROFILE,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv if argv is None else argv,
    }
    manifest_path = log_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    pool = None
    wrapped = None
    runner = None
    start = time.perf_counter()
    try:
        pool = UEGo1Pool(
            ue_binary=args.ue_binary,
            num_processes=args.num_processes,
            agents_per_process=args.agents_per_process,
            base_port=args.base_port,
            output_dir=log_dir / "ue",
            request_timeout=args.request_timeout,
            reset_mode=args.reset_mode,
            connect=args.connect,
            runtime_diagnostics=args.runtime_diagnostics == "enabled",
            step_mode=args.step_mode,
            training_telemetry="keyboard_v2" if args.task_profile == V2_PROFILE else "disabled",
            randomize_reset_pose=args.task_profile == V2_PROFILE,
            seed=args.seed,
        )
        wrapped = UEGo1VecEnv(pool, task, args.device)
        # Now include the reference actually reported by each reset UE actor.
        write_json(log_dir / "params" / "ue_task.json", task.task_manifest)
        manifest["task"] = task.task_manifest
        manifest["configuration_sha256"]["ue_task.json"] = sha256(
            log_dir / "params" / "ue_task.json"
        )
        runner = make_runner_class()(wrapped, runner_cfg, str(log_dir), args.device)
        if args.resume:
            runner.allow_command_sampling_change = args.allow_command_sampling_change
            runner.allow_linear_tracking_reward_change = args.allow_linear_tracking_reward_change
            resumed_info = runner.load(str(args.resume), map_location=args.device)
            manifest["resumed_command_sampling"] = (resumed_info.get("task_manifest") or {}).get("command_sampling", "source")
            manifest["resumed_linear_tracking_reward"] = (resumed_info.get("task_manifest") or {}).get("linear_tracking_reward", "source")
        else:
            runner.save(str(log_dir / "model_initial.pt"))
        manifest.update(
            status="training",
            initial_completed_iterations=runner.completed_iterations,
            initial_common_step_counter=int(task.common_step_counter),
            initial_command_distribution={
                "standing_envs": int((np.count_nonzero(task.commands, axis=1) == 0).sum()),
                "single_axis_envs": int((np.count_nonzero(task.commands, axis=1) == 1).sum()),
                "mixed_envs": int((np.count_nonzero(task.commands, axis=1) > 1).sum()),
                "heading_envs": int(task._is_heading_env.sum()),
                "command_min": task.commands.min(axis=0).tolist(),
                "command_max": task.commands.max(axis=0).tolist(),
            },
            startup_seconds=time.perf_counter() - start,
        )
        write_json(manifest_path, manifest)
        manifest["learning_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        runner.learn(
            num_learning_iterations=args.iterations, init_at_random_ep_len=False,
            training_budget_seconds=args.training_budget_seconds,
        )
        training_seconds = runner.learning_seconds_this_invocation
        manifest["final_validation"] = validate_trained_policy(
            runner, wrapped, log_dir / "policy_validation.npz"
        )
        checkpoint = log_dir / "model_final.pt"
        runner.save(str(checkpoint))
        exported = export_policy(runner, task, log_dir) if args.export_onnx else None
        completed_iterations = runner.completed_iterations - manifest["initial_completed_iterations"]
        transitions = completed_iterations * num_envs * agent_cfg.num_steps_per_env
        manifest.update(
            status="completed",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - start,
            training_seconds=training_seconds,
            additional_iterations_completed=completed_iterations,
            training_stop_reason=runner.training_stop_reason,
            training_transitions=transitions,
            end_to_end_training_transitions_per_second=transitions / training_seconds,
            final_completed_iterations=runner.completed_iterations,
            final_common_step_counter=int(task.common_step_counter),
            environment=wrapped.metrics(),
            collection_seconds=runner.total_collection_seconds,
            update_seconds=runner.total_update_seconds,
            final_checkpoint={"path": str(checkpoint), "sha256": sha256(checkpoint)},
            exported_policy=exported,
        )
        write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2))
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter() - start,
        )
        if runner is not None and "learning_started_at_utc" in manifest:
            manifest.update(
                training_seconds=runner.learning_elapsed(),
                additional_iterations_completed=runner.completed_iterations - manifest["initial_completed_iterations"],
                training_stop_reason="interrupted",
            )
        if wrapped is not None:
            try:
                manifest["environment"] = wrapped.metrics()
            except Exception as metrics_error:
                manifest["environment_metrics_error"] = str(metrics_error)
        elif pool is not None:
            try:
                manifest["pool_at_failure"] = pool.metrics()
            except Exception as metrics_error:
                manifest["pool_metrics_error"] = str(metrics_error)
        write_json(manifest_path, manifest)
        raise
    finally:
        if pool is not None:
            pool.close()
            manifest["ue_cleanup"] = {
                "owned_process_exit_codes": [worker.proc.returncode for worker in pool.workers if worker.proc is not None],
                "external_editor_left_running": bool(args.connect),
                "errors": pool.cleanup_errors,
            }
            write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
