from pathlib import Path


def test_random_population_wrapper_has_no_noop_step_override():
    text = Path('gym_unrealcv/envs/wrappers/augmentation.py').read_text()
    assert 'class RandomPopulationWrapper' in text
    assert 'def step(self, action):' not in text
