from pathlib import Path


def test_rendezvous_init_supports_reset_type():
    text = Path('gym_unrealcv/envs/rendezvous.py').read_text()
    assert 'reset_type=0' in text
    assert 'reset_type=reset_type' in text
