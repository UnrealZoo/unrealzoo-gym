from pathlib import Path


def test_configue_wrapper_has_no_noop_step_or_reset_overrides():
    text = Path('gym_unrealcv/envs/wrappers/configUE.py').read_text()
    assert 'def step(self, action):' not in text
    assert 'def reset(self, **kwargs):' not in text
