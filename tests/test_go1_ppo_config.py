"""Pinned source provenance and simulator-free UE PPO configuration loading."""
import builtins
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example/rl/locomotion"))
import ue_ppo_config as module
from train_go1_ue import prepare_runner_cfg


def available_source():
    candidates = [
        Path(os.environ["UNREALZOO_MJLAB_SOURCE"]) if os.environ.get("UNREALZOO_MJLAB_SOURCE") else None,
        Path.home() / ".local/share/unrealzoo/mjlab-e710cead",
    ]
    return next((path for path in candidates if path is not None and (path / module.CONFIG_FILE).is_file()), None)


class PPOConfigTests(unittest.TestCase):
    def test_wrong_commit_is_rejected_before_source_execution(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            module.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="wrong-commit\n")
        ):
            with self.assertRaisesRegex(RuntimeError, "Expected MJLab"):
                module.inspect_ppo_source(Path(directory))

    def test_tracked_modifications_are_rejected(self):
        replies = [SimpleNamespace(returncode=0, stdout=module.SOURCE_COMMIT),
                   SimpleNamespace(returncode=0, stdout=" M src/mjlab/rl/config.py\n")]
        with tempfile.TemporaryDirectory() as directory, patch.object(module.subprocess, "run", side_effect=replies):
            with self.assertRaisesRegex(RuntimeError, "tracked modifications"):
                module.inspect_ppo_source(Path(directory))

    def test_changed_source_is_rejected_after_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / module.CONFIG_FILE
            path.parent.mkdir(parents=True)
            initial = b'raise AssertionError("must never run changed source")\n'
            path.write_bytes(initial + b'# changed\n')
            source = {"source_root": directory, "configuration_files_sha256": {
                module.CONFIG_FILE: hashlib.sha256(initial).hexdigest(),
            }}
            with self.assertRaisesRegex(RuntimeError, "Source changed after verification"):
                module.load_source_utility(source, module.CONFIG_FILE)

    def test_unrelated_source_module_cannot_be_loaded(self):
        with self.assertRaisesRegex(ValueError, "Only the pinned pure"):
            module.load_source_utility({}, "src/mjlab/tasks/__init__.py")

    def test_actual_pinned_source_loads_without_any_native_simulator_import(self):
        source_path = available_source()
        if source_path is None:
            self.skipTest("Pinned source checkout is not available")
        import_original = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"mjlab", "warp", "mujoco", "mujoco_warp"}:
                raise AssertionError(f"Native simulator imported: {name}")
            return import_original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded_import):
            source = module.inspect_ppo_source(source_path)
            cfg = module.load_ppo_runner_cfg(source)
        self.assertEqual(source["commit"], module.SOURCE_COMMIT)
        self.assertEqual(set(source["configuration_files_sha256"]), {
            module.CONFIG_FILE, module.GO1_FILE, *module.UTILITY_FILES,
        })
        self.assertEqual(cfg.num_steps_per_env, 24)
        self.assertEqual(cfg.actor.hidden_dims, (512, 256, 128))
        self.assertEqual(cfg.algorithm.class_name, "PPO")



if __name__ == "__main__":
    unittest.main()
