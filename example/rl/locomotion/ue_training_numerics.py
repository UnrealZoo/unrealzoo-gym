"""Fail early on invalid UE/PPO data without changing the pinned PPO equations.

The v2 action bound is an environment contract: PPO retains the sampled action
and its log probability, while physics, last-action observations and action-rate
rewards all use the same bounded applied action. Diagnostic limits below abort;
they never replace corrupt observations, rewards or gradients with zeros.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


V2_PROFILE = "ue_keyboard_flat_v2"
ACTION_CLIP = 5.0
NUMERICAL_CONTRACT = {
    "version": 1,
    "applied_action_clip": ACTION_CLIP,
    "raw_action_abort_above": 1e6,
    "observation_abort_above": 1e4,
    "ppo_value_or_return_abort_above": 1e6,
    "optimizer_gradient_check": "before every optimizer step, after source gradient clipping",
    "failure_behavior": "abort with diagnostics; no automatic restart or NaN substitution",
}


class NumericalFailure(RuntimeError):
    pass


def checked_array(value, name, *, max_abs=None):
    """Return a finite array or identify the actual offending element."""
    value = np.asarray(value)
    bad = ~np.isfinite(value)
    if max_abs is not None:
        bad |= np.abs(value) > max_abs
    if np.any(bad):
        index = tuple(int(x) for x in np.argwhere(bad)[0])
        raise NumericalFailure(
            f"{name}: invalid value {value[index]!r} at {index}; abs limit={max_abs}"
        )
    return value


def applied_actions(raw):
    raw = checked_array(raw, "sampled actions", max_abs=1e6)
    applied = np.clip(raw, -ACTION_CLIP, ACTION_CLIP)
    clipped = np.abs(raw) > ACTION_CLIP
    metrics = {
        "Actions/raw_abs_max": float(np.max(np.abs(raw))),
        "Actions/applied_abs_max": float(np.max(np.abs(applied))),
        "Actions/clipped_fraction": float(np.mean(clipped)),
    }
    # np.clip allocates: the sampled tensor retained by PPO must not be mutated.
    return applied, metrics


class PPONumericalGuard:
    """Instance-local hooks around the unchanged installed RSL-RL algorithm."""

    def __init__(self, algorithm, output_dir: Path):
        import torch

        self.torch = torch
        self.algorithm = algorithm
        self.output_dir = Path(output_dir)
        self.last_failure = None
        self.maxima = {}
        self.update_kl = {"count": 0, "min": None, "max": None, "mean": None}
        self._collecting_kl = False
        self._kl_sum = 0.0
        self.handles = []
        for name, model in (("actor", algorithm.actor), ("critic", algorithm.critic)):
            def check_output(module, inputs, output, label=name):
                self.check_tensor(output, f"{label}_output", 1e6)
            self.handles.append(model.register_forward_hook(check_output))
        self.handles.append(algorithm.optimizer.register_step_pre_hook(self._before_step))
        original_returns = algorithm.compute_returns
        original_update = algorithm.update
        original_kl = algorithm.actor.get_kl_divergence

        def get_kl_divergence(*args, **kwargs):
            # Observe the source PPO's actual call and return its tensor intact.
            result = original_kl(*args, **kwargs)
            if self._collecting_kl:
                mean = float(result.detach().mean().cpu())
                if not math.isfinite(mean):
                    self.fail(f"non-finite PPO minibatch KL mean: {mean!r}")
                stats = self.update_kl
                stats["count"] += 1
                stats["min"] = mean if stats["min"] is None else min(stats["min"], mean)
                stats["max"] = mean if stats["max"] is None else max(stats["max"], mean)
                self._kl_sum += mean
                stats["mean"] = self._kl_sum / stats["count"]
            return result

        def compute_returns(obs):
            storage = algorithm.storage
            for name in ("rewards", "values", "actions", "actions_log_prob"):
                self.check_tensor(getattr(storage, name), f"rollout_{name}", 1e6)
            original_returns(obs)
            for name in ("returns", "advantages"):
                self.check_tensor(getattr(storage, name), f"rollout_{name}", 1e6)

        def update():
            self.update_kl = {"count": 0, "min": None, "max": None, "mean": None}
            self._kl_sum = 0.0
            self._collecting_kl = True
            try:
                losses = original_update()
            finally:
                self._collecting_kl = False
            for name, value in losses.items():
                if not math.isfinite(float(value)):
                    self.fail(f"non-finite PPO {name} loss: {value!r}")
            self.check_model()
            return losses

        algorithm.compute_returns = compute_returns
        algorithm.update = update
        algorithm.actor.get_kl_divergence = get_kl_divergence

    def fail(self, message):
        self.last_failure = message
        raise NumericalFailure(message)

    def check_tensor(self, value, name, max_abs=None):
        if not self.torch.is_tensor(value):
            self.fail(f"{name} is not a tensor")
        with self.torch.no_grad():
            maximum = float(value.detach().abs().amax().cpu()) if value.numel() else 0.0
        if not math.isfinite(maximum) or (max_abs is not None and maximum > max_abs):
            self.fail(f"{name}: abs max={maximum!r}, limit={max_abs}")
        self.maxima[name] = max(self.maxima.get(name, 0.0), maximum)

    def _before_step(self, optimizer, args, kwargs):
        # Batch the GPU reductions into one synchronization per optimizer step.
        gradients = [parameter.grad.detach() for group in optimizer.param_groups
                     for parameter in group["params"] if parameter.grad is not None]
        if gradients:
            maxima = self.torch.stack([gradient.abs().amax() for gradient in gradients])
            self.check_tensor(maxima, "optimizer_gradients")

    def check_model(self):
        state = []
        for model in (self.algorithm.actor, self.algorithm.critic):
            state.extend(tensor.detach() for tensor in model.state_dict().values() if tensor.numel())
        for values in self.algorithm.optimizer.state.values():
            state.extend(value.detach() for value in values.values()
                         if self.torch.is_tensor(value) and value.numel())
        by_device = {}
        for value in state:
            by_device.setdefault(str(value.device), []).append(value)
        # Adam keeps its scalar step counters on CPU even for CUDA parameters.
        for device, values in by_device.items():
            self.check_tensor(self.torch.stack([value.abs().amax() for value in values]),
                              f"model_and_adam/{device}")

    def record_failure(self, error, environment):
        """Keep evidence separate from playable checkpoints, even on non-guard errors."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record = {"error": f"{type(error).__name__}: {error}",
                  "guard_failure": self.last_failure, "maxima_before_failure": self.maxima,
                  "update_kl": self.update_kl,
                  "contract": NUMERICAL_CONTRACT,
                  "vector_steps": getattr(environment, "vector_steps", None)}
        (self.output_dir / "numerical_failure.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # Explicitly not resumable: this is the failing state, not a good checkpoint.
        self.torch.save({"diagnostic_only_do_not_resume": True,
                         "algorithm": self.algorithm.save(),
                         "observations": environment.get_observations().to("cpu"),
                         "sampled_actions": getattr(self.algorithm.transition, "actions", None),
                         "ue_states": getattr(getattr(environment, "pool", None), "states", None)},
                        self.output_dir / "numerical_failure_state.pt")
