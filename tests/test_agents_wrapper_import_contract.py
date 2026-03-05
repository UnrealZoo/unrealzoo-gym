from pathlib import Path
import ast


def test_agents_wrapper_has_no_unused_bare_gym_import():
    tree = ast.parse(Path('gym_unrealcv/envs/wrappers/agents.py').read_text())
    bare_imports = [
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert 'gym' not in bare_imports

    wrapper_imports = [
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == 'gym'
        for alias in node.names
    ]
    assert 'Wrapper' in wrapper_imports


def test_navagents_class_still_exposes_step_and_reset():
    tree = ast.parse(Path('gym_unrealcv/envs/wrappers/agents.py').read_text())
    nav_agents = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NavAgents')
    methods = {node.name for node in nav_agents.body if isinstance(node, ast.FunctionDef)}
    assert 'step' in methods
    assert 'reset' in methods
