from pathlib import Path
import ast


def test_configue_wrapper_has_no_noop_step_or_reset_overrides():
    tree = ast.parse(Path('gym_unrealcv/envs/wrappers/configUE.py').read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ConfigUEWrapper')
    methods = {node.name for node in wrapper.body if isinstance(node, ast.FunctionDef)}

    assert '__init__' in methods
    assert 'step' not in methods
    assert 'reset' not in methods


def test_configue_wrapper_init_still_supports_comm_mode_and_resolution_update():
    text = Path('gym_unrealcv/envs/wrappers/configUE.py').read_text()
    assert 'comm_mode' in text
    assert 'define_observation_space' in text
