from pathlib import Path


def test_agents_wrapper_has_no_unused_bare_gym_import():
    text = Path('gym_unrealcv/envs/wrappers/agents.py').read_text()
    assert '\nimport gym\n' not in text
