"""Check safe actor reconstruction and deterministic UE calibration without Torch."""
import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


DIRECTORY = Path(__file__).resolve().parents[1] / "example/rl/locomotion"
sys.path.insert(0, str(DIRECTORY))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checkpoint = load_module("checkpoint_playback_test", DIRECTORY / "ue_go1_checkpoint.py")
playback = load_module("pt_playback_test", DIRECTORY / "play_go1.py")


def saved_checkpoint():
    metadata = {
        "unrealzoo_backend": checkpoint.BACKEND,
        "task_id": checkpoint.TASK_PROFILE,
        "ue_task_profile": checkpoint.TASK_PROFILE,
        "mjlab_source_commit": checkpoint.SOURCE_COMMIT,
        "joint_names": ",".join(playback.JOINT_NAMES),
        "observation_names": ",".join(playback.OBSERVATION_NAMES),
        "command_names": "twist", "control_dt": "0.02",
        "ue_observation_reference": "reset_control_targets",
        "ue_bridge_default_joint_pos": ",".join(map(str, [0, 0.9, -1.8] * 4)),
        **{key: ",".join(map(str, values)) for key, values in playback.PROFILE.items()},
    }
    return {
        "actor_state_dict": {"test_weight": np.zeros((12, 48), dtype=np.float32)},
        "infos": {
            "backend": checkpoint.BACKEND, "task_profile": checkpoint.TASK_PROFILE,
            "source_commit": checkpoint.SOURCE_COMMIT,
            "inference": {
                "schema_version": 1, "rsl_rl_version": "5.0.1",
                "actor_class": "MLPModel", "obs_groups": {"actor": ["actor"]},
                "obs_set": "actor", "observation_dim": 48, "action_dim": 12,
                "actor_cfg": {
                    "hidden_dims": [512, 256, 128], "activation": "elu",
                    "obs_normalization": False,
                    "distribution_cfg": {
                        "class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar",
                    },
                },
                "policy_metadata": metadata,
            },
        },
    }


class FakeTensor(np.ndarray):
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class FakeTensorDict(dict):
    def __init__(self, data, batch_size):
        super().__init__(data)
        self.batch_size = batch_size


def fake_torch():
    return types.SimpleNamespace(
        inference_mode=contextlib.nullcontext,
        as_tensor=lambda value, device: np.asarray(value),
        isfinite=np.isfinite,
        zeros=lambda *shape: np.zeros(shape, dtype=np.float32),
        set_num_threads=Mock(),
    )


class CheckpointPlaybackTests(unittest.TestCase):
    def test_v2_checkpoint_keeps_bounded_action_inference_contract(self):
        from ue_training_numerics import NUMERICAL_CONTRACT
        saved = saved_checkpoint()
        saved["infos"].update(task_profile=checkpoint.V2_PROFILE,
                              numerical_contract=copy.deepcopy(NUMERICAL_CONTRACT))
        metadata = saved["infos"]["inference"]["policy_metadata"]
        metadata.update(task_id=checkpoint.V2_PROFILE, ue_task_profile=checkpoint.V2_PROFILE,
                        action_clip="5.0")
        _, metadata = checkpoint.checkpoint_recipe(saved)
        playback.validate_policy_metadata(metadata)
        captured = []

        def actor(obs, stochastic_output):
            captured.append(obs["actor"].copy())
            return np.full((1, 12), 8.0, dtype=np.float32).view(FakeTensor)

        policy = checkpoint.UECheckpointPolicy(actor, metadata, "cpu", playback.PROFILE["default_joint_pos"])
        with patch.dict(sys.modules, {
            "torch": fake_torch(), "tensordict": types.SimpleNamespace(TensorDict=FakeTensorDict),
        }):
            for _ in range(2):
                np.testing.assert_array_equal(policy.act(np.zeros(48), [0, 0, 0]), np.full(12, 5))
        np.testing.assert_array_equal(captured[1][0, 33:45], np.full(12, 5))
        del saved["infos"]["inference"]["policy_metadata"]["action_clip"]
        with self.assertRaisesRegex(ValueError, "bounded-action"):
            checkpoint.checkpoint_recipe(saved)

    def test_valid_recipe_is_copied_and_semantics_validated(self):
        saved = saved_checkpoint()
        recipe, metadata = checkpoint.checkpoint_recipe(saved)
        playback.validate_policy_metadata(metadata)
        recipe["actor_cfg"]["distribution_cfg"].pop("class_name")
        self.assertEqual(saved["infos"]["inference"]["actor_cfg"]["distribution_cfg"]["class_name"], "GaussianDistribution")

    def test_native_and_ambiguous_checkpoints_are_rejected(self):
        for section, key, value in (
            ("infos", "backend", "mjlab"),
            ("infos", "task_profile", "parkour"),
            ("infos", "source_commit", "unverified"),
            ("inference", "actor_class", "ArbitraryPythonClass"),
            ("inference", "rsl_rl_version", "unknown"),
            ("inference", "obs_groups", {"actor": ["critic"]}),
        ):
            with self.subTest(key=key):
                saved = saved_checkpoint()
                target = saved["infos"] if section == "infos" else saved["infos"]["inference"]
                target[key] = value
                with self.assertRaises(ValueError):
                    checkpoint.checkpoint_recipe(saved)
        saved = saved_checkpoint()
        del saved["infos"]["inference"]
        with self.assertRaisesRegex(ValueError, "do not guess"):
            checkpoint.checkpoint_recipe(saved)

    def test_actor_recipe_cannot_silently_omit_activation_or_add_dynamic_imports(self):
        for change in ("missing_activation", "wrong_distribution", "extra_field"):
            saved = saved_checkpoint()
            cfg = saved["infos"]["inference"]["actor_cfg"]
            if change == "missing_activation":
                del cfg["activation"]
            elif change == "wrong_distribution":
                cfg["distribution_cfg"]["class_name"] = "some.module:callable"
            else:
                cfg["cnn_cfg"] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                checkpoint.checkpoint_recipe(saved)

    def test_torch_load_is_weights_only_actor_load_is_strict_and_actor_is_eval(self):
        torch = fake_torch()
        torch.load = Mock(return_value=saved_checkpoint())
        actor = Mock()
        actor.return_value = np.zeros((1, 12), dtype=np.float32).view(FakeTensor)
        actor.state_dict.return_value = {"weight": np.array([1.0])}
        actor.to.return_value = actor
        models = types.ModuleType("rsl_rl.models")
        models.MLPModel = Mock(return_value=actor)
        with patch.dict(sys.modules, {
            "torch": torch, "rsl_rl": types.ModuleType("rsl_rl"),
            "rsl_rl.models": models, "tensordict": types.SimpleNamespace(TensorDict=FakeTensorDict),
        }), patch.object(checkpoint.package_metadata, "version", return_value="5.0.1"):
            policy, _ = checkpoint.load_ue_checkpoint(Path("model.pt"), "cpu", playback.PROFILE["default_joint_pos"])
        torch.load.assert_called_once_with(Path("model.pt"), map_location="cpu", weights_only=True)
        self.assertTrue(actor.load_state_dict.call_args.kwargs["strict"])
        actor.to.assert_called_once_with("cpu")
        actor.eval.assert_called_once()
        actor.requires_grad_.assert_called_once_with(False)
        torch.set_num_threads.assert_called_once_with(1)
        self.assertFalse(actor.call_args.kwargs["stochastic_output"])
        self.assertIs(policy.actor, actor)

    def test_deterministic_actor_receives_calibrated_noise_free_observation(self):
        _, metadata = checkpoint.checkpoint_recipe(saved_checkpoint())
        captured = []

        def actor(obs, stochastic_output):
            self.assertFalse(stochastic_output)
            captured.append(obs["actor"].copy())
            return np.full((1, 12), 0.25, dtype=np.float32).view(FakeTensor)

        actor.reset = Mock()
        policy = checkpoint.UECheckpointPolicy(actor, metadata, "cpu", playback.PROFILE["default_joint_pos"])
        raw = np.linspace(-0.2, 0.2, 48, dtype=np.float32)
        original = raw.copy()
        command = [0.5, -0.1, 0.2]
        with patch.dict(sys.modules, {
            "torch": fake_torch(), "tensordict": types.SimpleNamespace(TensorDict=FakeTensorDict),
        }):
            for _ in range(2):
                prepared = playback.prepare_policy_observation(raw, command, metadata, playback.PROFILE["default_joint_pos"])
                np.testing.assert_array_equal(policy.act(prepared, command), np.full(12, 0.25))
        expected = raw.copy()
        expected[9:21] += np.asarray([0, 0.9, -1.8] * 4) - np.asarray(playback.PROFILE["default_joint_pos"])
        expected[45:48] = command
        for index, value in enumerate(captured):
            expected[33:45] = 0 if index == 0 else 0.25
            np.testing.assert_allclose(value[0], expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(raw, original)
        policy.reset()
        actor.reset.assert_called_once()
        np.testing.assert_array_equal(policy.last_action, np.zeros(12))

    def test_pt_dispatch_does_not_use_legacy_onnx_adapter(self):
        _, metadata = checkpoint.checkpoint_recipe(saved_checkpoint())
        helper = types.ModuleType("ue_go1_checkpoint")
        helper.load_ue_checkpoint = Mock(return_value=(object(), metadata))
        with patch.dict(sys.modules, {"ue_go1_checkpoint": helper, "onnxruntime": None, "common.pretrained_policy": None}):
            playback.load_policy(Path("explicit.pt"), "cpu")
        helper.load_ue_checkpoint.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "Legacy YAML compatibility needs PyYAML in the PPO runtime")
    def test_legacy_recipe_requires_both_checkpoint_and_config_hashes(self):
        saved = saved_checkpoint()
        inference = saved["infos"].pop("inference")
        saved["infos"]["task_manifest"] = {
            "profile": checkpoint.TASK_PROFILE, "source_commit": checkpoint.SOURCE_COMMIT,
            "actor_dim": 48, "control_dt": 0.02, "policy_metadata": inference["policy_metadata"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_final.pt"
            path.write_bytes(b"mock checkpoint bytes")
            agent = path.parent / "params" / "agent.yaml"
            agent.parent.mkdir()
            agent.write_text(
                'obs_groups:\n  actor: !!python/tuple [actor]\nclip_actions: null\nactor:\n'
                '  class_name: MLPModel\n  hidden_dims: !!python/tuple [512, 256, 128]\n'
                '  activation: elu\n  obs_normalization: false\n  cnn_cfg: null\n'
                '  rnn_type: null\n  rnn_hidden_dim: 256\n  rnn_num_layers: 1\n'
                '  distribution_cfg: {class_name: GaussianDistribution, init_std: 1.0, std_type: scalar}\n'
            )
            manifest = {
                "backend": checkpoint.BACKEND, "task_profile": checkpoint.TASK_PROFILE,
                "source": {"commit": checkpoint.SOURCE_COMMIT, "versions": {"rsl-rl-lib": "5.0.1"}},
                "final_checkpoint": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                "configuration_sha256": {"agent.yaml": hashlib.sha256(agent.read_bytes()).hexdigest()},
            }
            (path.parent / "run_manifest.json").write_text(json.dumps(manifest))
            recipe, metadata = checkpoint.checkpoint_recipe(saved, path)
            self.assertEqual(recipe["actor_cfg"], inference["actor_cfg"])
            playback.validate_policy_metadata(metadata)
            original = agent.read_bytes()
            agent.write_bytes(original + b'\n# edited\n')
            with self.assertRaisesRegex(ValueError, "agent.yaml"):
                checkpoint.checkpoint_recipe(saved, path)
            agent.write_bytes(original)
            path.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "final checkpoint SHA256"):
                checkpoint.checkpoint_recipe(saved, path)


if __name__ == "__main__":
    unittest.main()
