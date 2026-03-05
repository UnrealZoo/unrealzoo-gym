from pathlib import Path


def test_navigation_has_no_redundant_observation_type_assignment():
    text = Path('gym_unrealcv/envs/navigation.py').read_text()
    assert 'self.observation_type = observation_type' not in text
