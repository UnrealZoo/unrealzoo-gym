"""Guard task causality and the real UE-to-policy observation boundary."""

import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ue_go1_task", ROOT / "example/rl/locomotion/ue_go1_task.py"
)
task_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task_module)
Go1Task = task_module.Go1Task


def fresh_state():
    """Minimal v3.1.0 reset contract; its keyframe has zero hip angles."""
    obs = np.zeros(48)
    obs[8] = -1.0
    return {
        "obs": obs.tolist(),
        "sim_time": 0.0,
        "synchronous": True,
        "policy_profile": "velocity",
        "control_targets": [0.0, 0.9, -1.8] * 4,
        "foot_velocities": [0.0] * 12,
        "foot_contacts": [True] * 4,
    }


def advanced(state, action=None):
    state = copy.deepcopy(state)
    state["sim_time"] += 0.02
    state["obs"][33:45] = [0.0] * 12 if action is None else list(action)
    return state


def fresh_keyboard_state():
    state = fresh_state()
    state.update(
        control_targets=task_module.DEFAULT_JOINT_POS.tolist(),
        training_telemetry_version=2,
        foot_heights=[0.03] * 4,
        foot_current_air_time=[0.0] * 4,
        foot_current_contact_time=[0.0] * 4,
        foot_contact_forces_world=[0.0, 0.0, -25.0] * 4,
        root_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        soft_joint_pos_limits=[-1.0, 1.0, -1.0, 3.0, -3.0, 0.0] * 4,
    )
    return state


def advance_keyboard(state, action=None):
    state = advanced(state, action)
    state["foot_current_air_time"] = [
        0.0 if contact else value + 0.02
        for contact, value in zip(state["foot_contacts"], state["foot_current_air_time"])
    ]
    state["foot_current_contact_time"] = [
        value + 0.02 if contact else 0.0
        for contact, value in zip(state["foot_contacts"], state["foot_current_contact_time"])
    ]
    return state


class Go1TaskTests(unittest.TestCase):
    def test_v310_reset_contract_has_a_different_joint_reference(self):
        state = fresh_state()
        task = Go1Task(1, observation_noise=False)
        obs = task.reset([0], [state])
        # UE-relative zero does not mean the MJLab default hip pose.
        expected = [-0.1, 0, 0, 0.1, 0, 0] * 2
        np.testing.assert_allclose(obs["actor"][0, 9:21], expected, atol=1e-7)
        self.assertEqual(task.policy_metadata["ue_observation_reference"], "reset_control_targets")
        self.assertFalse(task.task_manifest["equivalent_to_source_task"])
        self.assertEqual(task.task_manifest["critic_dim"], 48)

    def test_reward_at_source_default_pose_is_dt_scaled_and_action_history_is_raw(self):
        task = Go1Task(1, observation_noise=False)
        initial = fresh_state()
        task.reset([0], [initial])
        task.set_commands([[0, 0, 0]])
        state = advanced(initial, np.ones(12))
        state["obs"][9:21] = [0.1, 0, 0, -0.1, 0, 0] * 2
        obs, rewards, terminated, truncated, metrics = task.step([state], np.ones((1, 12)))
        # Four positive terms = 6; action difference cost = 12 * 0.1.
        self.assertAlmostEqual(float(rewards[0]), (6.0 - 1.2) * 0.02, places=7)
        np.testing.assert_array_equal(obs["actor"][0, 33:45], np.ones(12))
        self.assertFalse(terminated[0] or truncated[0])
        state = advanced(state, np.ones(12))
        _, rewards, _, _, metrics = task.step([state], np.ones((1, 12)))
        self.assertAlmostEqual(float(rewards[0]), 6.0 * 0.02, places=7)
        self.assertEqual(float(metrics["reward_terms"]["action_rate_l2"][0]), 0.0)

    def test_linear_reward_uses_root_velocity_not_offset_imu_velocity(self):
        task = Go1Task(1, observation_noise=False)
        initial = fresh_state()
        task.reset([0], [initial])
        task.set_commands([[0.5, 0.0, 0.5]])
        state = advanced(initial)
        omega = np.array([0.0, 0.0, 0.5])
        state["obs"][:3] = (np.array([0.5, 0.0, 0.0]) + np.cross(omega, task.imu_offset)).tolist()
        state["obs"][3:6] = omega.tolist()
        _, _, _, _, metrics = task.step([state], np.zeros((1, 12)))
        self.assertAlmostEqual(metrics["reward_terms"]["track_linear_velocity"][0], 0.04)
        self.assertAlmostEqual(metrics["reward_terms"]["track_angular_velocity"][0], 0.04)
        np.testing.assert_allclose(metrics["tracking_mae"], 0.0, atol=1e-10)

    def test_fall_precedes_timeout_and_partial_reset_does_not_erase_other_episode(self):
        task = Go1Task(2, observation_noise=False, episode_seconds=0.04)
        initial = [fresh_state(), fresh_state()]
        task.reset([0, 1], initial)
        states = [advanced(item) for item in initial]
        task.step(states, np.zeros((2, 12)))
        states = [advanced(item) for item in states]
        tilt = math.radians(75)
        states[0]["obs"][6:9] = [math.sin(tilt), 0.0, -math.cos(tilt)]
        _, _, terminated, truncated, metrics = task.step(states, np.zeros((2, 12)))
        np.testing.assert_array_equal(terminated, [True, False])
        np.testing.assert_array_equal(truncated, [False, True])
        self.assertEqual(len(metrics["completed_episodes"]), 2)
        self.assertEqual(metrics["completed_episodes"][0]["length"], 2)
        with self.assertRaisesRegex(RuntimeError, "must be reset"):
            task.step(states, np.zeros((2, 12)))
        previous_return = task.episode_returns[1]
        task.reset([0], [fresh_state()])
        self.assertEqual(task.episode_lengths.tolist(), [0, 2])
        self.assertEqual(task.episode_returns[1], previous_return)
        self.assertTrue(task._terminal[1])

    def test_reward_uses_old_command_when_next_command_is_resampled(self):
        task = Go1Task(1, observation_noise=False)
        state = fresh_state()
        task.reset([0], [state])
        task.set_commands([[0.5, 0, 0]])
        task._command_remaining[:] = 0.01
        state = advanced(state)
        state["obs"][:3] = [0.5, 0, 0]
        observation, _, _, _, metrics = task.step([state], np.zeros((1, 12)))
        self.assertAlmostEqual(metrics["reward_terms"]["track_linear_velocity"][0], 0.04)
        self.assertFalse(np.array_equal(task.commands, [[0.5, 0, 0]]))
        np.testing.assert_array_equal(observation["actor"][:, 45:48], task.commands)

    def test_noise_only_changes_actor_and_never_previous_actions_or_commands(self):
        task = Go1Task(2, seed=9, observation_noise=True)
        obs = task.reset([0, 1], [fresh_state(), fresh_state()])
        difference = obs["actor"] - obs["critic"]
        self.assertGreater(np.linalg.norm(difference[:, :33]), 0)
        np.testing.assert_array_equal(difference[:, 33:], 0.0)
        self.assertTrue((np.abs(difference) <= task_module.NOISE_AMPLITUDE + 1e-7).all())
        self.assertTrue(all(value.dtype == np.float32 for value in obs.values()))

    def test_rejects_missing_telemetry_stale_steps_and_mismatched_actions(self):
        initial = fresh_state()
        task = Go1Task(1, observation_noise=False)
        task.reset([0], [initial])
        with self.assertRaisesRegex(RuntimeError, "20 ms"):
            task.step([initial], np.zeros((1, 12)))
        with self.assertRaisesRegex(RuntimeError, "different last action"):
            task.step([advanced(initial)], np.ones((1, 12)))
        state = advanced(initial)
        del state["foot_velocities"]
        with self.assertRaises(KeyError):
            task.step([state], np.zeros((1, 12)))
        state = advanced(initial)
        state["obs"][0] = float("nan")
        with self.assertRaises(ValueError):
            task.step([state], np.zeros((1, 12)))
        self.assertEqual(task.common_step_counter, 0)

    def test_standing_disables_foot_slip_and_airborne_feet_do_not_count(self):
        task = Go1Task(1, observation_noise=False)
        state = fresh_state()
        task.reset([0], [state])
        task.set_commands([[0.5, 0, 0]])
        state = advanced(state)
        state["foot_velocities"] = [3, 4, 99] * 4
        state["foot_contacts"] = [True, False, False, False]
        _, _, _, _, metrics = task.step([state], np.zeros((1, 12)))
        self.assertAlmostEqual(metrics["reward_terms"]["foot_slip"][0], -0.05)
        task.set_commands([[0, 0, 0]])
        _, _, _, _, metrics = task.step([advanced(state)], np.zeros((1, 12)))
        self.assertEqual(metrics["reward_terms"]["foot_slip"][0], 0)

    def test_command_curriculum_uses_vector_steps_not_environment_count(self):
        task = Go1Task(4000, seed=1)
        ids = np.arange(task.num_envs)
        task._resample_commands(ids)
        self.assertLessEqual(np.max(np.abs(task.commands[:, 0])), 1.0)
        self.assertLessEqual(np.max(np.abs(task.commands[:, 2])), 0.5)
        self.assertTrue(((task._command_remaining >= 3) & (task._command_remaining <= 8)).all())
        task.common_step_counter = 240000
        task._resample_commands(ids)
        self.assertGreater(task.commands[:, 0].max(), 2.5)
        self.assertLessEqual(task.commands[:, 0].max(), 3.0)
        self.assertLessEqual(np.max(np.abs(task.commands[:, 2])), 0.7)

    def test_reset_rejects_advanced_physics_without_partially_changing_other_envs(self):
        task = Go1Task(2)
        with self.assertRaisesRegex(ValueError, "sim_time=0"):
            task.reset([0, 1], [fresh_state(), advanced(fresh_state())])
        self.assertFalse(task._initialized.any())
        self.assertTrue(np.isnan(task._sim_times).all())

    def test_resume_resamples_commands_at_restored_stage_before_first_step(self):
        task = Go1Task(128, seed=3, observation_noise=False)
        with self.assertRaises(RuntimeError):
            task.restore_common_step_counter(240000)
        states = [fresh_state() for _ in range(task.num_envs)]
        task.reset(np.arange(task.num_envs), states)
        self.assertLessEqual(task.commands[:, 0].max(), 1.0)
        obs = task.restore_common_step_counter(240000)
        self.assertEqual(task.common_step_counter, 240000)
        self.assertGreater(task.commands[:, 0].max(), 2.5)
        np.testing.assert_array_equal(obs["actor"][:, 45:48], task.commands)
        self.assertFalse(task.episode_lengths.any())
        for invalid in (-1, 1.5, True):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                task.restore_common_step_counter(invalid)
        task.step([advanced(state) for state in states], np.zeros((task.num_envs, 12)))
        with self.assertRaises(RuntimeError):
            task.restore_common_step_counter(0)

    def test_float32_reward_overflow_is_rejected_before_history_changes(self):
        task = Go1Task(1, observation_noise=False)
        state = fresh_state()
        task.reset([0], [state])
        actions = np.full((1, 12), 1e22)
        with self.assertRaisesRegex(FloatingPointError, "rewards outside finite float32 range"):
            task.step([advanced(state, actions[0])], actions)
        self.assertEqual(task.common_step_counter, 0)
        np.testing.assert_array_equal(task._last_actions, 0)


class KeyboardTaskTests(unittest.TestCase):
    def task(self, count=1, **kwargs):
        return Go1Task(count, profile="ue_keyboard_flat_v2", observation_noise=False, **kwargs)

    def test_profile_is_explicit_and_legacy_observations_stay_48(self):
        legacy = Go1Task(1, observation_noise=False)
        self.assertEqual(legacy.profile, "ue_v310_velocity")
        self.assertEqual(legacy.critic_dim, 48)
        self.assertIsNone(legacy.action_clip)
        task = self.task()
        state = fresh_keyboard_state()
        observation = task.reset([0], [state])
        self.assertEqual(observation["actor"].shape, (1, 48))
        self.assertEqual(observation["critic"].shape, (1, 72))
        self.assertEqual(task.action_clip, 5.0)
        self.assertEqual(task.policy_metadata["ue_action_clip"], "5.0")
        self.assertEqual(task.policy_metadata["action_clip"], "5.0")
        self.assertFalse(task.task_manifest["equivalent_to_source_task"])
        self.assertEqual(task.task_manifest["disabled_source_rewards"], {})
        with self.assertRaises(ValueError):
            Go1Task(1, profile="unknown")

    def test_critic_telemetry_order_and_signed_log_force_have_no_noise(self):
        task = Go1Task(1, profile="ue_keyboard_flat_v2", observation_noise=True, seed=3)
        state = fresh_keyboard_state()
        state["foot_heights"] = [0.1, 0.2, 0.3, 0.4]
        state["foot_contact_forces_world"] = [-3, 0, -24, 0, 4, 8, 1, -1, 0, 2, -2, 0]
        observation = task.reset([0], [state])
        critic = observation["critic"][0]
        np.testing.assert_allclose(critic[48:52], state["foot_heights"])
        np.testing.assert_array_equal(critic[52:56], 0)
        np.testing.assert_array_equal(critic[56:60], 1)
        force = np.asarray(state["foot_contact_forces_world"])
        np.testing.assert_allclose(critic[60:72], np.sign(force) * np.log1p(np.abs(force)), rtol=1e-6)
        difference = observation["actor"][0] - critic[:48]
        self.assertGreater(np.linalg.norm(difference[:33]), 0)
        np.testing.assert_array_equal(difference[33:], 0)

    def test_missing_real_v2_fields_and_invalid_clock_or_limits_fail_closed(self):
        for field in ("training_telemetry_version", "foot_heights", "foot_current_air_time",
                      "foot_current_contact_time", "foot_contact_forces_world", "root_quat_wxyz", "soft_joint_pos_limits"):
            task = self.task()
            state = fresh_keyboard_state()
            del state[field]
            with self.subTest(field=field), self.assertRaises((ValueError, KeyError)):
                task.reset([0], [state])
            self.assertFalse(task._initialized.any())
        for change in ({"training_telemetry_version": 1}, {"foot_current_air_time": [0.1] * 4},
                       {"foot_current_contact_time": [-1.0] * 4}, {"root_quat_wxyz": [0.0] * 4},
                       {"soft_joint_pos_limits": [1.0, -1.0] * 12}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.task().reset([0], [{**fresh_keyboard_state(), **change}])

    def test_restored_costs_use_world_velocity_soft_limits_and_force_norm(self):
        task = self.task()
        state = fresh_keyboard_state()
        task.reset([0], [state])
        task.set_commands([[0.5, 0, 0]])
        state["foot_contacts"] = [True, False, False, False]
        state = advance_keyboard(state)
        state["foot_heights"] = [0.05, 0.06, 0.07, 0.08]
        state["foot_velocities"] = [3.0, 4.0, 999.0] * 4
        state["foot_contact_forces_world"] = [-3, -4, -12] + [0] * 9
        state["obs"][9] = 1.25 - task_module.DEFAULT_JOINT_POS[0]
        _, _, _, _, metrics = task.step([state], np.zeros((1, 12)))
        terms = metrics["reward_terms"]
        self.assertAlmostEqual(terms["dof_pos_limits"][0], -0.25 * 0.02)
        self.assertAlmostEqual(terms["foot_clearance"][0], -2 * 0.7 * 0.02)
        self.assertAlmostEqual(terms["soft_landing"][0], -1e-5 * 13 * 0.02)
        self.assertAlmostEqual(terms["foot_swing_height"][0], -0.25 * 0.02)
        self.assertAlmostEqual(terms["foot_slip"][0], -0.1 * 25 * 0.02)

    def test_first_contact_uses_final_substep_clock_not_any_contact_flag(self):
        task = self.task(3)
        initial = [fresh_keyboard_state() for _ in range(3)]
        task.reset([0, 1, 2], initial)
        task.set_commands([[0.5, 0, 0]] * 3)
        states = [advance_keyboard(state) for state in initial]
        for state, clock in zip(states, (0.02, 0.04, 0.0)):
            state["foot_current_contact_time"] = [clock] * 4
        _, _, _, _, metrics = task.step(states, np.zeros((3, 12)))
        np.testing.assert_allclose(metrics["reward_terms"]["soft_landing"], [-0.00002, 0, 0])

    def test_swing_peak_is_sampled_while_airborne_charged_on_landing_then_reset(self):
        task = self.task()
        state = fresh_keyboard_state()
        task.reset([0], [state])
        task.set_commands([[0.5, 0, 0]])
        for height in (0.04, 0.12, 0.08):
            state["foot_contacts"][0] = False
            state = advance_keyboard(state)
            state["foot_heights"][0] = height
            task.step([state], np.zeros((1, 12)))
        state["foot_contacts"][0] = True
        state = advance_keyboard(state)
        state["foot_heights"][0] = 0.0
        _, _, _, _, metrics = task.step([state], np.zeros((1, 12)))
        self.assertAlmostEqual(metrics["reward_terms"]["foot_swing_height"][0], -0.25 * 0.2**2 * 0.02)
        self.assertAlmostEqual(metrics["log"]["Metrics/peak_height_mean"], 0.12)
        np.testing.assert_array_equal(task._peak_heights, 0)
        _, _, _, _, metrics = task.step([advance_keyboard(state)], np.zeros((1, 12)))
        self.assertEqual(metrics["reward_terms"]["foot_swing_height"][0], 0)

    def test_partial_reset_clears_only_selected_swing_history(self):
        task = self.task(2)
        initial = [fresh_keyboard_state(), fresh_keyboard_state()]
        task.reset([0, 1], initial)
        states = []
        for state, height in zip(initial, (0.2, 0.3)):
            state["foot_contacts"] = [False] * 4
            state = advance_keyboard(state)
            state["foot_heights"] = [height] * 4
            states.append(state)
        task.step(states, np.zeros((2, 12)))
        task.reset([0], [fresh_keyboard_state()])
        np.testing.assert_array_equal(task._peak_heights[0], 0)
        np.testing.assert_allclose(task._peak_heights[1], 0.3)

    def test_public_observe_revalidates_changed_input_after_a_valid_step(self):
        task = self.task()
        state = fresh_keyboard_state()
        task.reset([0], [state])
        state = advance_keyboard(state)
        task.step([state], np.zeros((1, 12)))
        for changes in (
            {"obs": [float("nan")] * 48},
            {"foot_heights": [float("nan")] * 4},
            {"foot_contacts": [2] * 4},
            {"foot_current_air_time": [.02] * 4},
            {"soft_joint_pos_limits": [1., -1.] * 12},
            {"root_quat_wxyz": [0.] * 4},
            {"training_telemetry_version": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                task.observe([{**state, **changes}])

    def test_task_snapshots_isolate_caller_mutations_during_partial_reset(self):
        for after_step in (False, True):
            for arrays in (False, True):
                with self.subTest(after_step=after_step, arrays=arrays):
                    task, reference = self.task(2), self.task(2)
                    initial = [fresh_keyboard_state(), fresh_keyboard_state()]
                    if arrays:
                        for state in initial:
                            for key, value in list(state.items()):
                                if isinstance(value, list):
                                    state[key] = np.asarray(value)
                    task.reset([0, 1], initial)
                    reference.reset([0, 1], copy.deepcopy(initial))
                    supplied = initial
                    if after_step:
                        supplied = [advance_keyboard(state) for state in initial]
                        task.step(supplied, np.zeros((2, 12)))
                        reference.step(copy.deepcopy(supplied), np.zeros((2, 12)))
                    # Corrupt caller-owned arrays/lists after task accepted them.
                    # Resetting only env 0 must still observe the saved env 1.
                    for key in ("obs", "control_targets", "foot_velocities", "foot_heights",
                                "foot_current_air_time", "foot_current_contact_time",
                                "foot_contact_forces_world", "root_quat_wxyz", "soft_joint_pos_limits"):
                        supplied[1][key][0] = float("nan")
                    supplied[1]["foot_contacts"][0] = False
                    supplied[1]["sim_time"] = -1.
                    supplied[1]["training_telemetry_version"] = 1
                    actual = task.reset([0], [fresh_keyboard_state()])
                    expected = reference.reset([0], [fresh_keyboard_state()])
                    for key in ("actor", "critic"):
                        np.testing.assert_array_equal(actual[key], expected[key])
                    np.testing.assert_array_equal(task._sim_times, reference._sim_times)
                    np.testing.assert_array_equal(task.episode_returns, reference.episode_returns)
                    self.assertEqual(task.rng.bit_generator.state, reference.rng.bit_generator.state)

    def test_heading_wrap_gain_clipping_and_standing_precedence(self):
        task = self.task(3)
        task.reset([0, 1, 2], [fresh_keyboard_state() for _ in range(3)])
        task.commands[:] = [0.5, 0.2, 0]
        task._is_heading_env[:] = True
        task._is_standing_env[:] = [False, False, True]
        task._heading_target[:] = [math.pi - 0.1, 2.0, 2.0]
        task._root_heading[:] = [-math.pi + 0.1, 0, 0]
        task._update_heading_commands(np.arange(3))
        np.testing.assert_allclose(task.commands[:, 2], [-0.1, 0.5, 0], atol=1e-7)
        np.testing.assert_array_equal(task.commands[2], 0)
        task.common_step_counter = 120000
        task._update_heading_commands(np.arange(3))
        self.assertAlmostEqual(float(task.commands[1, 2]), 0.7, places=6)
        task.set_commands([[0.6, 0, 0.1]] * 3)
        task._update_heading_commands(np.arange(3))
        np.testing.assert_allclose(task.commands, [[0.6, 0, 0.1]] * 3)

    def test_heading_updates_from_new_quaternion_after_old_command_reward(self):
        task = self.task()
        state = fresh_keyboard_state()
        task.reset([0], [state])
        task._is_heading_env[:] = True
        task._is_standing_env[:] = False
        task._heading_target[:] = 1.0
        task._command_remaining[:] = 10
        task._update_heading_commands([0])
        state = advance_keyboard(state)
        state["obs"][3:6] = [0, 0, 0.5]
        state["root_quat_wxyz"] = [math.cos(0.5), 0, 0, math.sin(0.5)]
        observation, _, _, _, metrics = task.step([state], np.zeros((1, 12)))
        self.assertAlmostEqual(metrics["reward_terms"]["track_angular_velocity"][0], 0.04)
        self.assertAlmostEqual(float(observation["actor"][0, 47]), 0.0, places=6)

    def test_initial_episode_phases_only_change_timeout_bookkeeping_once(self):
        task = self.task(4, episode_seconds=0.2)
        initial = [fresh_keyboard_state() for _ in range(4)]
        task.reset(np.arange(4), initial)
        commands = task.commands.copy()
        lengths = task.initialize_episode_lengths([0, 1, 9, 5])
        np.testing.assert_array_equal(lengths, [0, 1, 9, 5])
        np.testing.assert_array_equal(task.commands, commands)
        np.testing.assert_array_equal(task.episode_returns, 0)
        np.testing.assert_array_equal(task._sim_times, 0)
        with self.assertRaises(RuntimeError):
            task.initialize_episode_lengths([0] * 4)
        _, _, _, truncated, _ = task.step([advance_keyboard(state) for state in initial], np.zeros((4, 12)))
        np.testing.assert_array_equal(truncated, [False, False, True, False])

    def test_v2_rejects_unexecuted_unclipped_actions(self):
        task = self.task()
        state = fresh_keyboard_state()
        task.reset([0], [state])
        actions = np.full((1, 12), 5.1)
        with self.assertRaisesRegex(ValueError, "clipped"):
            task.step([advance_keyboard(state, actions[0])], actions)
        self.assertEqual(task.common_step_counter, 0)

    def test_resume_preserves_random_timeout_phases_before_first_physics_step(self):
        task = self.task(4)
        states = [fresh_keyboard_state() for _ in range(4)]
        task.reset(np.arange(4), states)
        lengths = task.initialize_episode_lengths([4, 23, 42, 999])
        observation = task.restore_common_step_counter(240000)
        np.testing.assert_array_equal(task.episode_lengths, lengths)
        np.testing.assert_array_equal(task._sim_times, 0)
        self.assertEqual(task.common_step_counter, 240000)
        self.assertEqual(observation["critic"].shape, (4, 72))
        task.step([advance_keyboard(state) for state in states], np.zeros((4, 12)))
        with self.assertRaisesRegex(RuntimeError, "immediately after"):
            task.restore_common_step_counter(0)

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ("torch", "tensordict", "rsl_rl")),
        "Needs the isolated PPO runtime",
    )
    def test_real_ppo_checkpoint_resume_preserves_timeout_phase_and_critic_shape(self):
        with patch.object(sys, "path", [str(ROOT / "example/rl/locomotion"), *sys.path]):
            from train_go1_ue import UEGo1VecEnv, make_runner_class, prepare_runner_cfg
            from ue_ppo_config import inspect_ppo_source, load_ppo_runner_cfg
        candidates = [
            Path(os.environ["UNREALZOO_MJLAB_SOURCE"]) if os.environ.get("UNREALZOO_MJLAB_SOURCE") else None,
            Path.home() / ".local/share/unrealzoo/mjlab-e710cead",
        ]
        source = next((path for path in candidates if path and (path / ".git").exists()), None)
        if source is None:
            self.skipTest("Pinned source checkout is unavailable")
        cfg = prepare_runner_cfg(load_ppo_runner_cfg(inspect_ppo_source(source)))
        cfg["actor"]["hidden_dims"] = cfg["critic"]["hidden_dims"] = (8,)
        cfg["num_steps_per_env"] = 2

        def make_env(seed):
            pool = SimpleNamespace(num_envs=4, reset=lambda: [fresh_keyboard_state() for _ in range(4)])
            return UEGo1VecEnv(pool, self.task(4, seed=seed), "cpu")

        import torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as directory:
            runner_type = make_runner_class()
            original = runner_type(make_env(3), copy.deepcopy(cfg), directory)
            original.env.restore_common_step_counter(240000)
            original.completed_iterations = 10000
            checkpoint = Path(directory) / "resume.pt"
            original.save(str(checkpoint))
            resumed_env = make_env(7)
            phases = resumed_env.task.episode_lengths.copy()
            self.assertTrue(phases.any())
            resumed = runner_type(resumed_env, copy.deepcopy(cfg), directory)
            resumed.load(str(checkpoint), map_location="cpu")
            np.testing.assert_array_equal(resumed_env.task.episode_lengths, phases)
            np.testing.assert_array_equal(resumed_env.episode_length_buf.numpy(), phases)
            self.assertEqual(resumed_env.task.common_step_counter, 240000)
            self.assertEqual(resumed.completed_iterations, 10000)
            self.assertEqual(tuple(resumed_env.get_observations()["critic"].shape), (4, 72))
            for key, value in original.alg.actor.state_dict().items():
                torch.testing.assert_close(resumed.alg.actor.state_dict()[key], value)
            # A command-distribution experiment must not silently reinterpret a
            # resume. Once explicitly opted in, actor and critic remain exact.
            balanced_env = make_env(19)
            balanced_env.task.command_sampling = 'axis_balanced_v1'
            balanced = runner_type(balanced_env, copy.deepcopy(cfg), directory)
            with self.assertRaisesRegex(ValueError, 'sampling changed'):
                balanced.load(str(checkpoint), map_location='cpu')
            balanced.allow_command_sampling_change = True
            balanced.load(str(checkpoint), map_location='cpu')
            self.assertEqual(balanced.completed_iterations, 10000)
            self.assertEqual(balanced_env.task.common_step_counter, 240000)
            for model in ['actor', 'critic']:
                for key, value in getattr(original.alg, model).state_dict().items():
                    torch.testing.assert_close(getattr(balanced.alg, model).state_dict()[key], value, rtol=0, atol=0)
            balanced_checkpoint = Path(directory)/'balanced.pt'
            balanced.save(str(balanced_checkpoint))
            mixed_env = make_env(23)
            mixed_env.task.command_sampling = 'axis_mixed_v2'
            mixed = runner_type(mixed_env,copy.deepcopy(cfg),directory)
            with self.assertRaisesRegex(ValueError, 'sampling changed'):
                mixed.load(str(balanced_checkpoint),map_location='cpu')
            mixed.allow_command_sampling_change = True
            mixed.load(str(balanced_checkpoint),map_location='cpu')
            self.assertEqual(mixed.completed_iterations,balanced.completed_iterations)
            self.assertEqual(mixed_env.task.common_step_counter,balanced_env.task.common_step_counter)
            self.assertEqual(mixed.alg.learning_rate,balanced.alg.learning_rate)
            for model in ['actor', 'critic']:
                for key,value in getattr(balanced.alg,model).state_dict().items():
                    torch.testing.assert_close(getattr(mixed.alg,model).state_dict()[key],value,rtol=0,atol=0)
            precise_env = make_env(29)
            precise_env.task.command_sampling = 'axis_balanced_v1'
            precise_env.task.linear_tracking_reward = 'precision_v1'
            precise = runner_type(precise_env,copy.deepcopy(cfg),directory)
            with self.assertRaisesRegex(ValueError, 'tracking reward changed'):
                precise.load(str(balanced_checkpoint),map_location='cpu')
            precise.allow_linear_tracking_reward_change = True
            precise.load(str(balanced_checkpoint),map_location='cpu')
            self.assertEqual(precise.completed_iterations,balanced.completed_iterations)
            self.assertEqual(precise_env.task.common_step_counter,balanced_env.task.common_step_counter)
            self.assertEqual(precise.alg.learning_rate,balanced.alg.learning_rate)
            for model in ['actor','critic']:
                for key,value in getattr(balanced.alg,model).state_dict().items():
                    torch.testing.assert_close(getattr(precise.alg,model).state_dict()[key],value,rtol=0,atol=0)


class KeyboardSourceOracleTests(unittest.TestCase):
    """Execute pinned source formulas on synthetic telemetry, without loading its simulator."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Source formula oracle requires torch")
        candidates = [
            Path(os.environ["UNREALZOO_MJLAB_SOURCE"]) if "UNREALZOO_MJLAB_SOURCE" in os.environ else None,
            Path.home() / ".local/share/unrealzoo/mjlab-e710cead",
        ]
        source = next((path for path in candidates if path and (path / "src/mjlab/tasks/velocity/mdp/rewards.py").is_file()), None)
        if source is None:
            raise unittest.SkipTest("Pinned keyboard source snapshot is not present")
        commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        if commit != task_module.SOURCE_COMMIT:
            raise RuntimeError("Source formula oracle requires the exact pinned keyboard commit")

        class HeightSensor:
            num_frames = 4

        cls.torch = torch
        cls.HeightSensor = HeightSensor
        cls.asset_cfg = SimpleNamespace(name="robot", joint_ids=slice(None), site_ids=slice(None))
        namespace = {"torch": torch, "TerrainHeightSensor": HeightSensor, "_DEFAULT_ASSET_CFG": cls.asset_cfg}
        selections = {
            "src/mjlab/tasks/velocity/mdp/rewards.py": ("feet_clearance", "feet_swing_height", "feet_slip", "soft_landing"),
            "src/mjlab/envs/mdp/rewards.py": ("joint_pos_limits",),
            "src/mjlab/tasks/velocity/mdp/observations.py": ("foot_height", "foot_air_time", "foot_contact", "foot_contact_forces"),
        }
        for relative, names in selections.items():
            path = source / relative
            original = subprocess.run(["git", "-C", str(source), "show", f"HEAD:{relative}"], capture_output=True, check=True).stdout
            if path.read_bytes() != original:
                raise RuntimeError(f"Source oracle file differs from pinned Git object: {relative}")
            parsed = ast.parse(original.decode(), filename=str(path))
            selected = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
            if {node.name for node in selected} != set(names):
                raise RuntimeError(f"Pinned source formula set changed: {relative}")
            module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        cls.source = namespace

    def test_new_rewards_and_critic_extras_match_unchanged_source_functions(self):
        torch = self.torch
        count = 4
        task = Go1Task(count, profile="ue_keyboard_flat_v2", observation_noise=False)
        states = [fresh_keyboard_state() for _ in range(count)]
        task.reset(np.arange(count), states)
        commands = np.array([[0.5, 0, 0], [0, 0, 0], [0, 0, 0.4], [1, 0.2, 0.2]], dtype=np.float32)
        task.set_commands(commands)
        height_sensor = self.HeightSensor()
        contact_sensor = SimpleNamespace()
        contact_sensor.compute_first_contact = lambda dt: (
            (contact_sensor.data.current_contact_time > 0)
            & (contact_sensor.data.current_contact_time < dt + 1e-6)
        )
        robot = SimpleNamespace()
        env = SimpleNamespace(num_envs=count, device="cpu", step_dt=0.02, extras={"log": {}},
                              scene={"robot": robot, "heights": height_sensor, "contacts": contact_sensor},
                              command_manager=SimpleNamespace(get_command=lambda _: torch.tensor(commands)))
        swing = self.source["feet_swing_height"](SimpleNamespace(params={"height_sensor_name": "heights"}), env)
        rng = np.random.default_rng(17)
        for tick in range(8):
            for index, state in enumerate(states):
                state["foot_contacts"] = (rng.random(4) > 0.5).tolist()
                states[index] = state = advance_keyboard(state)
                state["foot_heights"] = rng.uniform(0.01, 0.22, 4).tolist()
                state["foot_velocities"] = rng.uniform(-2, 2, 12).tolist()
                state["foot_contact_forces_world"] = rng.uniform(-50, 50, 12).tolist()
                state["obs"][9:21] = rng.uniform(-1.1, 1.1, 12).tolist()
            tensor = lambda key: torch.tensor(np.asarray([state[key] for state in states]), dtype=torch.float32)
            height_sensor.data = SimpleNamespace(heights=tensor("foot_heights"))
            contact_sensor.data = SimpleNamespace(found=tensor("foot_contacts"), force=tensor("foot_contact_forces_world").reshape(count, 4, 3),
                                                  current_air_time=tensor("foot_current_air_time"),
                                                  current_contact_time=tensor("foot_current_contact_time"))
            robot.data = SimpleNamespace(joint_pos=tensor("obs")[:, 9:21] + torch.tensor(task_module.DEFAULT_JOINT_POS, dtype=torch.float32),
                                         soft_joint_pos_limits=tensor("soft_joint_pos_limits").reshape(count, 12, 2),
                                         site_lin_vel_w=tensor("foot_velocities").reshape(count, 4, 3))
            expected_costs = {
                "dof_pos_limits": self.source["joint_pos_limits"](env, self.asset_cfg),
                "foot_clearance": self.source["feet_clearance"](env, 0.1, "heights", "twist", 0.05, self.asset_cfg),
                "foot_swing_height": swing(env, "contacts", "heights", 0.1, "twist", 0.05),
                "soft_landing": self.source["soft_landing"](env, "contacts", "twist", 0.05),
                "foot_slip": self.source["feet_slip"](env, "contacts", "twist", 0.05, self.asset_cfg),
            }
            observation, _, _, _, metrics = task.step(states, np.zeros((count, 12)))
            for name, expected in expected_costs.items():
                with self.subTest(tick=tick, reward=name):
                    np.testing.assert_allclose(metrics["reward_terms"][name], expected.numpy() * task.reward_weights[name] * 0.02,
                                               rtol=2e-5, atol=1e-7)
            expected_extra = torch.cat([
                self.source["foot_height"](env, "heights"), self.source["foot_air_time"](env, "contacts"),
                self.source["foot_contact"](env, "contacts"), self.source["foot_contact_forces"](env, "contacts"),
            ], dim=1)
            np.testing.assert_allclose(observation["critic"][:, 48:], expected_extra.numpy(), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
