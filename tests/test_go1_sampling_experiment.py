"""Verify experiment settings preserve old defaults and delayed command stages."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example/rl/locomotion"))
from ue_go1_task import Go1Task
from train_go1_ue import build_parser, validate_command_sampling_resume


class SamplingExperimentTests(unittest.TestCase):
    def test_defaults_and_explicit_batch(self):
        base = ["--connect", "127.0.0.1:23920", "--log-dir", "unused"]
        args = build_parser().parse_args(base)
        self.assertIsNone(args.rollout_steps)
        self.assertEqual(args.curriculum_step_scale, 1)
        args = build_parser().parse_args(base + ["--rollout-steps", "384", "--curriculum-step-scale", "16"])
        self.assertEqual(args.rollout_steps * 256, 98304)
        self.assertEqual(args.curriculum_step_scale, 16)

    def test_scaled_stages_and_heading_limits(self):
        task = Go1Task(1000, seed=9, profile="ue_keyboard_flat_v2", curriculum_step_scale=16)
        ids = np.arange(1000)
        self.assertEqual(task.task_manifest["command_curriculum_vector_steps"], [0, 1920000, 3840000])
        task.common_step_counter = 120000
        task._resample_commands(ids)
        self.assertLessEqual(np.max(np.abs(task.commands[:, 0])), 1)
        self.assertLessEqual(np.max(np.abs(task.commands[:, 2])), .5)
        task.common_step_counter = 1920000
        task._resample_commands(ids)
        self.assertGreater(np.max(task.commands[:, 0]), 1.5)
        self.assertGreater(np.max(np.abs(task.commands[:, 2])), .5)
        task.common_step_counter = 3840000
        task._resample_commands(ids)
        self.assertGreater(np.max(task.commands[:, 0]), 2.5)

    def test_invalid_scale_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Go1Task(1, curriculum_step_scale=value)

    def test_axis_balanced_covers_six_directions_without_heading_override(self):
        count = 24000
        task = Go1Task(count, seed=71, profile='ue_keyboard_flat_v2', command_sampling='axis_balanced_v1')
        ids = np.arange(count)
        task._root_heading[:] = 1.7
        task._resample_commands(ids)
        commands = task.commands.copy()
        nonzero = np.count_nonzero(commands, axis=1)
        self.assertAlmostEqual(np.mean(nonzero == 0), .1, delta=.015)
        self.assertAlmostEqual(np.mean(nonzero == 1), .5, delta=.02)
        self.assertAlmostEqual(np.mean(nonzero == 3), .4, delta=.02)
        single = nonzero == 1
        for axis in range(3):
            for sign in [-1, 1]:
                selected = single & (sign * commands[:, axis] > 0)
                self.assertAlmostEqual(np.mean(selected), .5 / 6, delta=.012)
                values = np.abs(commands[selected, axis])
                self.assertGreaterEqual(values.min(), .25 if axis == 2 else .2)
                self.assertLessEqual(values.max(), .5 if axis == 2 else .7)
        self.assertFalse(task._is_heading_env[single].any())
        task._root_heading[:] = -2.2
        task._update_heading_commands(ids)
        np.testing.assert_array_equal(task.commands[single], commands[single])
        self.assertFalse(task.commands[nonzero == 0].any())
        self.assertEqual(task.task_manifest['command_sampling_experiment']['mixed_fraction'], .4)

    def test_default_command_rng_and_subset_resampling_remain_unchanged(self):
        first = Go1Task(100, seed=18, profile='ue_keyboard_flat_v2')
        explicit = Go1Task(100, seed=18, profile='ue_keyboard_flat_v2', command_sampling='source')
        for _ in range(3):
            first._resample_commands(np.arange(100))
            explicit._resample_commands(np.arange(100))
            np.testing.assert_array_equal(first.commands, explicit.commands)
        balanced = Go1Task(100, profile='ue_keyboard_flat_v2', command_sampling='axis_balanced_v1')
        balanced.commands[:] = 9
        balanced._resample_commands([2, 5, 7])
        self.assertTrue((balanced.commands[[0, 1, 3, 4, 6, 8]] == 9).all())

    def test_sampling_resume_changes_require_explicit_opt_in(self):
        validate_command_sampling_resume('source', 'source')
        validate_command_sampling_resume('source', 'axis_balanced_v1', True)
        with self.assertRaises(ValueError):
            validate_command_sampling_resume('source', 'axis_balanced_v1')
        with self.assertRaises(ValueError):
            validate_command_sampling_resume('invalid', 'axis_balanced_v1', True)
        with self.assertRaises(ValueError):
            Go1Task(1, command_sampling='axis_balanced_v1')

    def test_mixed_v2_restores_mixed_exposure_and_preserves_pure_targets(self):
        task = Go1Task(24000, seed=71, profile='ue_keyboard_flat_v2', command_sampling='axis_mixed_v2')
        ids = np.arange(task.num_envs)
        task._resample_commands(ids)
        commands = task.commands.copy()
        nonzero = np.count_nonzero(commands, axis=1)
        for count, fraction in [(0, .1), (1, .3), (3, .6)]:
            self.assertAlmostEqual(np.mean(nonzero == count), fraction, delta=.02)
        for axis in range(3):
            for sign in [-1, 1]:
                self.assertAlmostEqual(np.mean((nonzero == 1) & (sign * commands[:, axis] > 0)), .05, delta=.012)
        self.assertAlmostEqual(np.mean(task._is_heading_env), .18, delta=.015)
        task._root_heading[:] = 2.1
        task._update_heading_commands(ids)
        np.testing.assert_array_equal(task.commands[nonzero <= 1], commands[nonzero <= 1])
        manifest = task.task_manifest
        self.assertEqual(manifest['command_sampling_experiment']['mixed_fraction'], .6)
        original = Go1Task(1, profile='ue_keyboard_flat_v2').task_manifest
        for key in ['reward_weights', 'restored_physics_contract', 'action_clip', 'command_resampling_seconds']:
            self.assertEqual(manifest[key], original[key])
        with self.assertRaises(ValueError):
            validate_command_sampling_resume('axis_balanced_v1', 'axis_mixed_v2')
        validate_command_sampling_resume('axis_balanced_v1', 'axis_mixed_v2', True)


if __name__ == "__main__":
    unittest.main()
