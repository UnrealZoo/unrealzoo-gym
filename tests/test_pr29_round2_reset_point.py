from pathlib import Path


def test_reset_point_uses_modern_class_declaration():
    text = Path('gym_unrealcv/envs/utils/reset_point.py').read_text()
    assert 'class ResetPoint:' in text
    assert 'class ResetPoint()' not in text
