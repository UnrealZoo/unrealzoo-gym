from pathlib import Path


def test_time_dilation_wrapper_has_no_unused_bare_gym_import():
    text = Path('gym_unrealcv/envs/wrappers/time_dilation.py').read_text()
    assert '\nimport gym\n' not in text
