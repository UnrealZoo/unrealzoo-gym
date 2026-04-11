from pathlib import Path
import ast


def test_time_dilation_wrapper_has_no_unused_bare_gym_import():
    tree = ast.parse(Path('gym_unrealcv/envs/wrappers/time_dilation.py').read_text())
    bare_imports = [
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert 'gym' not in bare_imports


def test_time_dilation_wrapper_still_implements_runtime_methods():
    tree = ast.parse(Path('gym_unrealcv/envs/wrappers/time_dilation.py').read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TimeDilationWrapper')
    methods = {node.name for node in wrapper.body if isinstance(node, ast.FunctionDef)}
    assert {'step', 'reset'}.issubset(methods)
