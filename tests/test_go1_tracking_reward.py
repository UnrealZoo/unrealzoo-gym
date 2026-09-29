"""Tracking precision is an explicit experiment; old tasks remain reproducible."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'example/rl/locomotion'))
from ue_go1_task import Go1Task
from train_go1_ue import build_parser, validate_linear_tracking_reward_resume
from test_ue_go1_task import fresh_keyboard_state, advance_keyboard


class TrackingRewardTests(unittest.TestCase):
    def test_precision_changes_only_linear_reward_and_retains_far_signal(self):
        errors = [0., .04, .1, 1.]
        results = []
        for profile in ('source', 'precision_v1'):
            task = Go1Task(4, profile='ue_keyboard_flat_v2', observation_noise=False,
                           command_sampling='axis_balanced_v1', linear_tracking_reward=profile)
            initial = [fresh_keyboard_state() for _ in errors]
            task.reset(range(4), initial)
            task.set_commands([[.3, .2, 0]] * 4)
            states = [advance_keyboard(s) for s in initial]
            for state, error in zip(states, errors):
                state['obs'][:2] = [.3, .2 - error]
            results.append(task.step(states, np.zeros((4, 12))))
        original, precision = results
        for key in original[0]:
            np.testing.assert_array_equal(original[0][key], precision[0][key])
        for index in (2, 3):
            np.testing.assert_array_equal(original[index], precision[index])
        for name, values in original[4]['reward_terms'].items():
            if name != 'track_linear_velocity':
                np.testing.assert_array_equal(values, precision[4]['reward_terms'][name])
        broad = original[4]['reward_terms']['track_linear_velocity']
        fine = precision[4]['reward_terms']['track_linear_velocity']
        self.assertAlmostEqual(broad[0], fine[0])
        self.assertAlmostEqual(fine[0], .04)
        self.assertGreater(fine[0] - fine[1], 2 * (broad[0] - broad[1]))
        self.assertLess(fine[0] - fine[1], 4 * (broad[0] - broad[1]))
        self.assertGreaterEqual(fine[-1], .5 * broad[-1])
        self.assertTrue(np.all(np.diff(fine) < 0))


    def test_reward_resume_requires_explicit_opt_in_and_manifest(self):
        base=['--connect','127.0.0.1:23924','--log-dir','unused']
        args=build_parser().parse_args(base)
        self.assertEqual(args.linear_tracking_reward,'source')
        self.assertFalse(args.allow_linear_tracking_reward_change)
        validate_linear_tracking_reward_resume('source','source')
        with self.assertRaisesRegex(ValueError,'reward changed'):
            validate_linear_tracking_reward_resume('source','precision_v1')
        validate_linear_tracking_reward_resume('source','precision_v1',True)
        with self.assertRaises(ValueError):
            validate_linear_tracking_reward_resume('teacher','precision_v1',True)
        with self.assertRaises(ValueError):
            Go1Task(1,linear_tracking_reward='precision_v1')
        task=Go1Task(1,profile='ue_keyboard_flat_v2',linear_tracking_reward='precision_v1')
        self.assertEqual(task.task_manifest['linear_tracking_reward'],'precision_v1')
        self.assertEqual(task.task_manifest['linear_tracking_reward_experiment']['mixture_weights'],[.5,.5])


if __name__ == '__main__':
    unittest.main()
