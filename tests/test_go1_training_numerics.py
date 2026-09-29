"""Exercise the actual action boundary and Adam/PPO failure paths without UE."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example/rl/locomotion"))
from ue_training_numerics import NumericalFailure, PPONumericalGuard, applied_actions


class AppliedActionTests(unittest.TestCase):
    def test_applied_action_is_bounded_without_changing_sampled_action(self):
        raw = np.array([[-7.0, 0.5, 8.0]], dtype=np.float32)
        original = raw.copy()
        result, metrics = applied_actions(raw)
        np.testing.assert_array_equal(result, [[-5, .5, 5]])
        np.testing.assert_array_equal(raw, original)
        self.assertAlmostEqual(metrics["Actions/clipped_fraction"], 2 / 3)
        self.assertFalse(np.shares_memory(raw, result))

    def test_catastrophic_action_is_diagnosed_instead_of_hidden_by_clip(self):
        for bad in (float("nan"), float("inf"), 1e15):
            with self.subTest(value=bad), self.assertRaisesRegex(NumericalFailure, r"\(1, 2\)"):
                raw = np.zeros((3, 12))
                raw[1, 2] = bad
                applied_actions(raw)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "tensordict", "rsl_rl")),
    "Needs the isolated PPO runtime",
)
class PPONumericalTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tensordict import TensorDict
        from rsl_rl.algorithms import PPO
        from rsl_rl.models import MLPModel
        from rsl_rl.storage import RolloutStorage

        self.torch = torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.obs = TensorDict({"actor": torch.zeros(4, 48), "critic": torch.zeros(4, 72)}, [4])
        groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = MLPModel(self.obs, groups, "actor", 12, hidden_dims=(8,), activation="elu",
                         obs_normalization=False, distribution_cfg={
                             "class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})
        critic = MLPModel(self.obs, groups, "critic", 1, hidden_dims=(8,), activation="elu",
                          obs_normalization=False)
        self.alg = PPO(actor, critic, RolloutStorage("rl", 4, 2, self.obs, [12], "cpu"),
                       num_learning_epochs=1, num_mini_batches=1)
        self.guard = PPONumericalGuard(self.alg, Path(self.directory.name))

    def test_infinite_gradient_cannot_mutate_parameters_or_adam_state(self):
        torch = self.torch
        before = copy.deepcopy(self.alg.actor.state_dict())
        parameter = next(self.alg.actor.parameters())
        parameter.grad = torch.full_like(parameter, float("inf"))
        with self.assertRaisesRegex(NumericalFailure, "optimizer_gradients"):
            self.alg.optimizer.step()
        self.assertEqual(len(self.alg.optimizer.state), 0)
        for key, expected in before.items():
            torch.testing.assert_close(self.alg.actor.state_dict()[key], expected, rtol=0, atol=0)

    def test_failure_snapshot_retains_sample_and_last_physics_state(self):
        self.alg.transition.actions = self.torch.full((4, 12), 8.0)
        env = SimpleNamespace(vector_steps=17, get_observations=lambda: self.obs,
                              pool=SimpleNamespace(states=[{"sim_time": .34, "outlier_robot": 3}]))
        self.guard.record_failure(NumericalFailure("test failure"), env)
        # This diagnostic is generated inside this test, not an untrusted checkpoint.
        state = self.torch.load(Path(self.directory.name) / "numerical_failure_state.pt", weights_only=False)
        self.assertTrue(state["diagnostic_only_do_not_resume"])
        self.assertTrue((state["sampled_actions"] == 8).all())
        self.assertEqual(state["ue_states"], env.pool.states)

    def test_real_ppo_update_accepts_72_dim_critic_and_rejects_bad_rollout(self):
        torch = self.torch
        with torch.inference_mode():
            for _ in range(2):
                self.alg.act(self.obs)
                self.alg.process_env_step(self.obs, torch.ones(4), torch.zeros(4, dtype=torch.long), {})
            self.alg.compute_returns(self.obs)
        losses = self.alg.update()
        self.assertTrue(all(np.isfinite(x) for x in losses.values()))
        self.assertIn("optimizer_gradients", self.guard.maxima)
        self.alg.storage.rewards[0, 2, 0] = float("inf")
        with self.assertRaisesRegex(NumericalFailure, "rollout_rewards"):
            self.alg.compute_returns(self.obs)

    def test_cuda_update_checks_adam_cpu_step_counters(self):
        torch = self.torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable on this host")
        from rsl_rl.storage import RolloutStorage
        self.alg.actor.to("cuda:0")
        self.alg.critic.to("cuda:0")
        self.alg.device = "cuda:0"
        obs = self.obs.to("cuda:0")
        self.alg.storage = RolloutStorage("rl", 4, 2, obs, [12], "cuda:0")
        with torch.inference_mode():
            for _ in range(2):
                self.alg.act(obs)
                self.alg.process_env_step(obs, torch.ones(4, device="cuda:0"),
                                          torch.zeros(4, dtype=torch.long, device="cuda:0"), {})
            self.alg.compute_returns(obs)
        self.alg.update()
        self.assertIn("model_and_adam/cpu", self.guard.maxima)
        self.assertIn("model_and_adam/cuda:0", self.guard.maxima)

    def test_kl_logging_observes_twenty_source_batches_without_changing_update(self):
        from rsl_rl.algorithms import PPO
        from rsl_rl.models import MLPModel
        from rsl_rl.storage import RolloutStorage

        torch = self.torch

        def run(with_guard):
            torch.manual_seed(123)
            groups = {"actor": ["actor"], "critic": ["critic"]}
            actor = MLPModel(self.obs, groups, "actor", 12, hidden_dims=(8,), activation="elu",
                             obs_normalization=False, distribution_cfg={
                                 "class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})
            critic = MLPModel(self.obs, groups, "critic", 1, hidden_dims=(8,), activation="elu",
                              obs_normalization=False)
            algorithm = PPO(actor, critic, RolloutStorage("rl", 4, 2, self.obs, [12], "cpu"),
                            num_learning_epochs=5, num_mini_batches=4,
                            schedule="adaptive", desired_kl=.01, learning_rate=.001)
            original_kl = actor.get_kl_divergence
            actual_kl_tensors = []

            def retain_actual_kl(*args, **kwargs):
                result = original_kl(*args, **kwargs)
                actual_kl_tensors.append(result)
                return result

            actor.get_kl_divergence = retain_actual_kl
            guard = PPONumericalGuard(algorithm, Path(self.directory.name)) if with_guard else None
            with torch.inference_mode():
                for _ in range(2):
                    algorithm.act(self.obs)
                    algorithm.process_env_step(self.obs, torch.tensor([.25, 1., 2., .5]),
                                               torch.zeros(4, dtype=torch.long), {})
                algorithm.compute_returns(self.obs)
            losses = algorithm.update()
            return algorithm, guard, losses, torch.get_rng_state(), actual_kl_tensors

        baseline, _, expected_losses, expected_rng, expected_kl = run(False)
        observed, guard, actual_losses, actual_rng, actual_kl = run(True)
        self.assertEqual(len(expected_kl), 20)
        self.assertEqual(len(actual_kl), 20)
        means = [float(value.mean()) for value in expected_kl]
        self.assertEqual(guard.update_kl, {
            "count": 20, "min": min(means), "max": max(means), "mean": sum(means) / 20,
        })
        self.assertEqual(actual_losses, expected_losses)
        self.assertEqual(observed.learning_rate, baseline.learning_rate)
        torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)
        for expected, actual in zip(expected_kl, actual_kl):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name in ("actor", "critic"):
            expected = getattr(baseline, name).state_dict()
            actual = getattr(observed, name).state_dict()
            for key, value in expected.items():
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        expected_optimizer = baseline.optimizer.state_dict()
        actual_optimizer = observed.optimizer.state_dict()
        self.assertEqual(actual_optimizer["param_groups"], expected_optimizer["param_groups"])
        for parameter, state in expected_optimizer["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(actual_optimizer["state"][parameter][key], value, rtol=0, atol=0)

    def test_vec_env_shares_applied_actions_between_physics_reward_and_history(self):
        from train_go1_ue import UEGo1VecEnv

        class Task:
            actor_dim, critic_dim, max_episode_length = 48, 72, 1000
            task_manifest = {"profile": "ue_keyboard_flat_v2"}
            commands = np.zeros((4, 3))
            episode_lengths = np.zeros(4, dtype=int)

            def reset(self, ids, states):
                return {"actor": np.zeros((4, 48)), "critic": np.zeros((4, 72))}

            def initialize_episode_lengths(self):
                self.episode_lengths[:] = [3, 9, 12, 42]
                return self.episode_lengths

            def step(self, states, actions):
                self.applied = actions.copy()
                obs = self.reset(None, None)
                obs["actor"][:, 33:45] = actions
                return obs, np.ones(4), np.zeros(4, bool), np.zeros(4, bool), {}

        task = Task()
        received = []
        pool = SimpleNamespace(num_envs=4, reset=lambda: [],
                               step=lambda action, command: received.append(action.copy()))
        env = UEGo1VecEnv(pool, task, "cpu")
        self.assertEqual(env.episode_length_buf.tolist(), [3, 9, 12, 42])
        raw = self.torch.full((4, 12), 8.0)
        observations, _, _, _ = env.step(raw)
        np.testing.assert_array_equal(received[0], np.full((4, 12), 5.0))
        np.testing.assert_array_equal(task.applied, received[0])
        self.assertTrue((observations["actor"][:, 33:45] == 5).all())
        self.assertTrue((raw == 8).all())


if __name__ == "__main__":
    unittest.main()
