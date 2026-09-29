"""Resume the real PPO adaptive learning rate without a UE connection."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example/rl/locomotion"))
from train_go1_ue import (
    BACKEND, SOURCE_COMMIT, TASK_PROFILE, checkpoint_learning_rate, make_runner_class,
)


class LearningRateContractTests(unittest.TestCase):
    def test_old_checkpoint_recovers_rate_and_invalid_rates_fail(self):
        saved = {"optimizer_state_dict": {"param_groups": [{"lr": 1e-5}]}}
        self.assertEqual(checkpoint_learning_rate(saved), 1e-5)
        for value in (0, -1, float("inf"), float("nan"), True, "0.001"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checkpoint_learning_rate({"optimizer_state_dict": {"param_groups": [{"lr": value}]}})
        for invalid in (
            {},
            {"optimizer_state_dict": {"param_groups": [{"lr": 1e-5}, {"lr": 1e-3}]}},
            {**saved, "infos": {"ppo_learning_rate": 1e-3}},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                checkpoint_learning_rate(invalid)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None for name in ("torch", "tensordict", "rsl_rl")),
    "Resume integration needs the isolated PPO runtime",
)
class PPOResumeTests(unittest.TestCase):
    def test_resumed_adaptive_update_matches_uninterrupted_optimizer(self):
        import torch
        from rsl_rl.algorithms import PPO
        from rsl_rl.models import MLPModel
        from rsl_rl.storage import RolloutStorage
        from tensordict import TensorDict

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        observations = TensorDict({"actor": torch.zeros(4, 48), "critic": torch.zeros(4, 48)}, batch_size=[4])
        groups = {"actor": ["actor"], "critic": ["critic"]}

        def algorithm():
            actor = MLPModel(
                observations, groups, "actor", 12, hidden_dims=(8,), activation="elu",
                obs_normalization=False,
                distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
            )
            critic = MLPModel(observations, groups, "critic", 1, hidden_dims=(8,), activation="elu", obs_normalization=False)
            return PPO(actor, critic, RolloutStorage("rl", 4, 2, observations, [12], "cpu"),
                       learning_rate=1e-3, num_learning_epochs=2, num_mini_batches=1)

        def update(alg, seed):
            torch.manual_seed(seed)
            with torch.inference_mode():
                for step in range(2):
                    obs = observations + 0.01 * (step + 1)
                    alg.act(obs)
                    alg.process_env_step(obs, torch.arange(4, dtype=torch.float32) + 0.1,
                                         torch.zeros(4, dtype=torch.long), {})
                alg.compute_returns(obs)
            return alg.update()

        torch.manual_seed(42)
        uninterrupted = algorithm()
        update(uninterrupted, 10)  # Populate Adam moments before the checkpoint.
        uninterrupted.learning_rate = 1e-5
        for group in uninterrupted.optimizer.param_groups:
            group["lr"] = 1e-5
        saved = copy.deepcopy(uninterrupted.save())
        saved["infos"] = {
            "backend": BACKEND, "task_profile": TASK_PROFILE, "source_commit": SOURCE_COMMIT,
            "completed_iterations": 64, "common_step_counter": 64 * 24,
        }  # Deliberately use the pre-fix checkpoint format from stage 01.
        resumed_alg = algorithm()
        restored_steps = []
        runner_class = make_runner_class()
        resumed = runner_class.__new__(runner_class)
        resumed.alg = resumed_alg
        resumed.env = SimpleNamespace(restore_common_step_counter=restored_steps.append)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage-01.pt"
            torch.save(saved, checkpoint)
            resumed.load(str(checkpoint), map_location="cpu")
        self.assertEqual(resumed_alg.learning_rate, 1e-5)
        self.assertEqual(resumed.completed_iterations, 64)
        self.assertEqual(resumed.current_learning_iteration, 64)
        self.assertEqual(restored_steps, [64 * 24])
        expected_losses = update(uninterrupted, 20)
        actual_losses = update(resumed_alg, 20)
        self.assertEqual(actual_losses, expected_losses)
        self.assertEqual(resumed_alg.learning_rate, uninterrupted.learning_rate)
        for key, tensor in uninterrupted.actor.state_dict().items():
            torch.testing.assert_close(resumed_alg.actor.state_dict()[key], tensor, rtol=0, atol=0)
        for key, tensor in uninterrupted.critic.state_dict().items():
            torch.testing.assert_close(resumed_alg.critic.state_dict()[key], tensor, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
