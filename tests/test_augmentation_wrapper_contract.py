from pathlib import Path
import ast


def test_random_population_wrapper_has_no_noop_step_override():
    text = Path('gym_unrealcv/envs/wrappers/augmentation.py').read_text()
    tree = ast.parse(text)
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RandomPopulationWrapper')
    methods = {node.name for node in wrapper.body if isinstance(node, ast.FunctionDef)}
    assert 'step' not in methods
    assert 'reset' in methods


def test_extract_reset_type_has_safe_fallback_branch():
    text = Path('gym_unrealcv/envs/wrappers/augmentation.py').read_text()
    assert 'getattr(env.unwrapped, "reset_type", 0)' in text
