from pathlib import Path


def test_tracking_reward_uses_modern_class_declaration():
    text = Path('gym_unrealcv/envs/tracking/reward.py').read_text()
    assert 'class Reward:' in text
    assert 'class Reward()' not in text
