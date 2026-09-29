"""Actual RSL rollout/update budget boundaries; no UE connection or simulator."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example/rl/locomotion"))
import train_go1_ue as module


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None for name in ("torch", "tensordict", "rsl_rl")),
    "Needs the isolated PPO runtime",
)
class TrainingBudgetTests(unittest.TestCase):
    def run_case(self, directory, budget, initial_iteration=0, iterations=10):
        import torch
        from tensordict import TensorDict

        clock = [1000.0]  # Initialization time is deliberately large and excluded.
        observations = TensorDict({"actor": torch.zeros(4, 48), "critic": torch.zeros(4, 48)}, [4])
        task = SimpleNamespace(common_step_counter=0, task_manifest={}, policy_metadata={})

        class Environment:
            device, num_envs, num_actions, max_episode_length = "cpu", 4, 12, 100
            cfg = {}
            episode_length_buf = torch.zeros(4, dtype=torch.long)

            def __init__(self):
                self.task = task
                self.steps = 0

            def get_observations(self):
                return observations

            def step(self, actions):
                clock[0] += 1.0
                self.steps += 1
                task.common_step_counter += 1
                return observations, torch.ones(4), torch.zeros(4, dtype=torch.long), {}

            def metrics(self):
                return {"steps": self.steps}

        cfg = {
            "num_steps_per_env": 2, "save_interval": 1, "logger": "tensorboard",
            "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
            "actor": {"class_name": "MLPModel", "hidden_dims": [8], "activation": "elu",
                      "obs_normalization": False,
                      "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}},
            "critic": {"class_name": "MLPModel", "hidden_dims": [8], "activation": "elu", "obs_normalization": False},
            "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 1,
                          "learning_rate": 1e-3, "rnd_cfg": None, "symmetry_cfg": None},
        }
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with contextlib.redirect_stdout(io.StringIO()):
            runner = module.make_runner_class()(Environment(), cfg, str(directory), "cpu")
            runner.current_learning_iteration = initial_iteration
            runner.completed_iterations = initial_iteration
            # Avoid code-repository snapshots in the test's logger, retaining the
            # real RSL learning loop, real optimizer, and real checkpoint saves.
            runner.logger._store_code_state = lambda: []
            with patch.object(module.time, "perf_counter", side_effect=lambda: clock[0]):
                runner.learn(iterations, training_budget_seconds=budget)
        return runner

    def test_budget_stops_after_whole_update_and_preserves_native_checkpoint(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            runner = self.run_case(path, 3.0)
            self.assertEqual(runner.env.steps, 4)  # 2 complete PPO updates.
            self.assertEqual(runner.completed_iterations, 2)
            self.assertEqual(runner.training_stop_reason, "budget")
            self.assertEqual(runner.learning_seconds_this_invocation, 4.0)
            saved = torch.load(path / "model_budget.pt", weights_only=True)
            self.assertEqual(saved["infos"]["completed_iterations"], 2)
            self.assertEqual(saved["infos"]["learning_seconds_this_invocation"], 4.0)
            self.assertIn("optimizer_state_dict", saved)
            self.assertFalse(list(path.glob("*.tmp")))
            self.assertFalse(list(path.glob("*.onnx")))
            records = [json.loads(line) for line in (path / "iteration_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["learning_seconds_this_invocation"] for r in records], [2.0, 4.0])

    def test_resume_counts_new_iterations_and_does_not_charge_previous_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.run_case(Path(directory), 1.0, initial_iteration=64)
            self.assertEqual(runner.completed_iterations, 65)
            self.assertEqual(runner.learning_seconds_this_invocation, 2.0)

    def test_iteration_limit_without_budget_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.run_case(Path(directory), None, iterations=2)
            self.assertEqual(runner.completed_iterations, 2)
            self.assertEqual(runner.training_stop_reason, "iteration_limit")
            self.assertEqual(runner.learning_seconds_this_invocation, 4.0)


if __name__ == "__main__":
    unittest.main()
