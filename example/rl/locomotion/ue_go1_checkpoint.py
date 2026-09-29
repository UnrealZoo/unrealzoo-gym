"""Deterministic playback of self-described UE Go1 RSL-RL checkpoints."""
from __future__ import annotations

import copy
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path

import numpy as np


SOURCE_COMMIT = "e710cead240b4c0f6f52afaa4f4b2a22c734082c"
BACKEND = "ue_mujoco"
TASK_PROFILE = "ue_v310_velocity"
V2_PROFILE = "ue_keyboard_flat_v2"
RSL_RL_VERSION = "5.0.1"


def legacy_inference_recipe(infos: dict, checkpoint: Path) -> dict:
    """Recover an old final checkpoint's recipe from hash-bound local run files."""
    import yaml

    manifest_path = checkpoint.parent / "run_manifest.json"
    agent_path = checkpoint.parent / "params" / "agent.yaml"
    try:
        manifest = json.loads(manifest_path.read_text())
        agent_bytes = agent_path.read_bytes()
    except (OSError, ValueError) as error:
        raise ValueError(
            "Old UE checkpoint requires its run_manifest.json and params/agent.yaml "
            "to recover the exact actor recipe"
        ) from error
    if (manifest.get("backend") != BACKEND or manifest.get("task_profile") != TASK_PROFILE
            or manifest.get("source", {}).get("commit") != SOURCE_COMMIT
            or manifest.get("source", {}).get("versions", {}).get("rsl-rl-lib") != RSL_RL_VERSION):
        raise ValueError("Old checkpoint run manifest has incompatible provenance")
    if manifest.get("final_checkpoint", {}).get("sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError("Old checkpoint does not match the run manifest's final checkpoint SHA256")
    if manifest.get("configuration_sha256", {}).get("agent.yaml") != hashlib.sha256(agent_bytes).hexdigest():
        raise ValueError("Old checkpoint agent.yaml does not match its recorded SHA256")

    class AgentConfigLoader(yaml.SafeLoader):
        pass

    # MJLab serializes tuples; allow that one data-only tag, never object loading.
    AgentConfigLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: loader.construct_sequence(node),
    )
    cfg = yaml.load(agent_bytes, Loader=AgentConfigLoader)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("actor"), dict):
        raise ValueError("Old checkpoint agent.yaml has no actor configuration")
    actor_cfg = copy.deepcopy(cfg["actor"])
    actor_class = actor_cfg.pop("class_name", None)
    if actor_cfg.get("cnn_cfg") is None:
        actor_cfg.pop("cnn_cfg", None)
    if actor_cfg.get("rnn_type") is None:
        for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            actor_cfg.pop(key, None)
    if cfg.get("clip_actions") is not None:
        raise ValueError("Old checkpoint declares unsupported action clipping")
    task = infos.get("task_manifest") or {}
    policy_metadata = dict(task.get("policy_metadata") or {})
    if (task.get("profile") != TASK_PROFILE or task.get("source_commit") != SOURCE_COMMIT
            or task.get("actor_dim") != 48 or task.get("control_dt") != 0.02):
        raise ValueError("Old checkpoint task manifest is incompatible")
    policy_metadata.update({
        "unrealzoo_backend": BACKEND, "task_id": TASK_PROFILE,
        "mjlab_source_commit": SOURCE_COMMIT, "control_dt": str(task["control_dt"]),
        "run_path": checkpoint.parent.name,
    })
    return {
        "schema_version": 1, "rsl_rl_version": RSL_RL_VERSION,
        "actor_class": actor_class, "actor_cfg": actor_cfg,
        "obs_groups": {"actor": cfg.get("obs_groups", {}).get("actor")},
        "obs_set": "actor", "observation_dim": 48, "action_dim": 12,
        "policy_metadata": policy_metadata,
    }


def checkpoint_recipe(saved: dict, checkpoint: Path | None = None) -> tuple[dict, dict]:
    """Require provenance and an explicit actor recipe; never infer it from weights."""
    if not isinstance(saved, dict):
        raise ValueError("UE .pt checkpoint must contain an RSL-RL state dictionary")
    infos = saved.get("infos")
    if not isinstance(infos, dict):
        raise ValueError("Not a UE-trained checkpoint: missing infos provenance")
    profile = infos.get("task_profile")
    if profile not in (TASK_PROFILE, V2_PROFILE):
        raise ValueError("UE .pt checkpoint has incompatible task_profile")
    for key, expected in (
        ("backend", BACKEND),
        ("source_commit", SOURCE_COMMIT),
    ):
        if infos.get(key) != expected:
            raise ValueError(f"UE .pt checkpoint has incompatible {key}")
    recipe = copy.deepcopy(infos.get("inference"))
    if recipe is None and checkpoint is not None:
        recipe = legacy_inference_recipe(infos, checkpoint)
    if not isinstance(recipe, dict):
        raise ValueError("UE checkpoint lacks its inference recipe; do not guess the actor architecture")
    for key, expected in (
        ("schema_version", 1), ("rsl_rl_version", RSL_RL_VERSION),
        ("actor_class", "MLPModel"), ("observation_dim", 48), ("action_dim", 12),
        ("obs_set", "actor"), ("obs_groups", {"actor": ["actor"]}),
    ):
        if recipe.get(key) != expected:
            raise ValueError(f"Unsupported checkpoint inference {key}: {recipe.get(key)!r}")
    cfg = recipe.get("actor_cfg")
    if not isinstance(cfg, dict) or set(cfg) != {
        "hidden_dims", "activation", "obs_normalization", "distribution_cfg",
    }:
        raise ValueError("Actor recipe must provide the complete supported MLPModel configuration")
    if (not isinstance(cfg["hidden_dims"], (tuple, list)) or not cfg["hidden_dims"]
            or any(type(width) is not int or width <= 0 for width in cfg["hidden_dims"])):
        raise ValueError("Actor hidden_dims must be positive integers")
    if cfg["activation"] not in {"elu", "relu", "selu", "crelu", "lrelu", "tanh", "sigmoid"}:
        raise ValueError("Unsupported actor activation")
    if type(cfg["obs_normalization"]) is not bool:
        raise ValueError("Actor obs_normalization must be Boolean")
    distribution = cfg["distribution_cfg"]
    if (not isinstance(distribution, dict)
            or set(distribution) != {"class_name", "init_std", "std_type"}
            or distribution["class_name"] != "GaussianDistribution"
            or distribution["std_type"] != "scalar"
            or not isinstance(distribution["init_std"], (int, float))
            or not np.isfinite(distribution["init_std"])
            or distribution["init_std"] <= 0):
        raise ValueError("Unsupported actor distribution recipe")
    policy_metadata = recipe.get("policy_metadata")
    if not isinstance(policy_metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in policy_metadata.items()
    ):
        raise ValueError("Checkpoint policy_metadata must map strings to strings")
    for key, expected in (
        ("unrealzoo_backend", BACKEND), ("task_id", profile),
        ("ue_task_profile", profile), ("mjlab_source_commit", SOURCE_COMMIT),
    ):
        if policy_metadata.get(key) != expected:
            raise ValueError(f"Checkpoint policy metadata has incompatible {key}")
    if profile == V2_PROFILE:
        from ue_training_numerics import ACTION_CLIP, NUMERICAL_CONTRACT
        if (policy_metadata.get("action_clip") != str(ACTION_CLIP)
                or infos.get("numerical_contract") != NUMERICAL_CONTRACT):
            raise ValueError("v2 checkpoint requires its exact bounded-action numerical contract")
    if not isinstance(saved.get("actor_state_dict"), dict) or not saved["actor_state_dict"]:
        raise ValueError("Checkpoint has no actor_state_dict")
    return recipe, policy_metadata


class UECheckpointPolicy:
    """Same observation adapter as legacy Go1 playback, with a direct Torch actor."""

    def __init__(self, actor, policy_metadata: dict, device: str, generic_bridge_default):
        self.actor = actor
        self.metadata = policy_metadata
        self.device = device
        self.generic_bridge_default = np.asarray(generic_bridge_default, dtype=np.float32)
        self.go1_default = np.asarray(policy_metadata["default_joint_pos"].split(","), dtype=np.float32)
        if self.generic_bridge_default.shape != (12,) or self.go1_default.shape != (12,):
            raise ValueError("Go1 joint references must contain 12 values")
        self.last_action = np.zeros(12, dtype=np.float32)

    def reset(self) -> None:
        self.last_action.fill(0)
        self.actor.reset()

    def act(self, observation, command) -> np.ndarray:
        import torch
        from tensordict import TensorDict

        # play_go1.prepare_policy_observation already replaces the command and
        # compensates the measured reset reference relative to this generic one.
        value = np.asarray(observation, dtype=np.float32).copy()
        if value.shape != (48,) or not np.isfinite(value).all():
            raise ValueError("Policy input must be a finite 48-vector")
        value[9:21] += self.generic_bridge_default - self.go1_default
        value[33:45] = self.last_action
        with torch.inference_mode():
            observations = TensorDict(
                {"actor": torch.as_tensor(value[None], device=self.device)}, batch_size=[1],
            )
            output = self.actor(observations, stochastic_output=False)
            if output.shape != (1, 12) or not torch.isfinite(output).all():
                raise RuntimeError("Checkpoint actor returned invalid actions")
            action = output[0].detach().cpu().numpy().astype(np.float32, copy=True)
        if self.metadata.get("ue_task_profile") == V2_PROFILE:
            from ue_training_numerics import applied_actions
            action, _ = applied_actions(action)
        self.last_action = action.copy()
        return action


def load_ue_checkpoint(path: Path, device: str, generic_bridge_default, threads: int = 1) -> tuple[UECheckpointPolicy, dict]:
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    if type(threads) is not int or threads <= 0:
        raise ValueError("Torch inference threads must be a positive integer")
    torch.set_num_threads(threads)
    version = package_metadata.version("rsl-rl-lib")
    if version != RSL_RL_VERSION:
        raise RuntimeError(f"UE checkpoint playback requires rsl-rl-lib {RSL_RL_VERSION}; found {version}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    recipe, policy_metadata = checkpoint_recipe(saved, Path(path))
    observations = TensorDict({"actor": torch.zeros(1, 48)}, batch_size=[1])
    actor = MLPModel(
        observations, recipe["obs_groups"], recipe["obs_set"], recipe["action_dim"],
        **copy.deepcopy(recipe["actor_cfg"]),
    )
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    for name, tensor in actor.state_dict().items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Checkpoint actor contains non-finite state: {name}")
    actor.to(device).eval()
    actor.requires_grad_(False)
    policy = UECheckpointPolicy(actor, policy_metadata, device, generic_bridge_default)
    # A real finite forward pass is part of --verify-only, without any UE access.
    policy.act(np.zeros(48, dtype=np.float32), np.zeros(3, dtype=np.float32))
    policy.reset()
    return policy, policy_metadata
