from pathlib import Path

from gym_unrealcv.envs.tracking.reward import Reward


def test_tracking_reward_uses_modern_class_declaration():
    text = Path('gym_unrealcv/envs/tracking/reward.py').read_text()
    assert 'class Reward:' in text
    assert 'class Reward()' not in text


def test_reward_distance_handles_direction_symmetry_and_clamping():
    reward = Reward(
        {
            'exp_distance': 100.0,
            'max_distance': 1000.0,
            'min_distance': 10.0,
            'max_direction': 180.0,
        }
    )

    same_pos = reward.reward_distance(100.0, 0.0)
    assert same_pos == 1

    positive_angle = reward.reward_distance(100.0, 15.0)
    negative_angle = reward.reward_distance(100.0, -15.0)
    assert positive_angle == negative_angle

    heavily_misaligned = reward.reward_distance(5000.0, 360.0)
    assert heavily_misaligned == -1
