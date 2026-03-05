from pathlib import Path


def test_navigationmulti_uses_modern_super_call():
    text = Path('gym_unrealcv/envs/navigationmulti.py').read_text()
    assert 'super().__init__(' in text
    assert 'super(NavigationMulti, self).__init__(' not in text
