from pathlib import Path
import ast


def test_keyboard_navigation_multi_agent_uses_modern_class_declaration():
    text = Path('example/Keyboard_NavigationMultiAgent.py').read_text()
    tree = ast.parse(text)
    random_agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RandomAgent')
    assert random_agent.bases == []
    assert 'class RandomAgent(object)' not in text


def test_keyboard_navigation_multi_agent_keeps_core_agent_methods():
    tree = ast.parse(Path('example/Keyboard_NavigationMultiAgent.py').read_text())
    random_agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RandomAgent')
    methods = {node.name for node in random_agent.body if isinstance(node, ast.FunctionDef)}
    assert {'__init__', 'act'}.issubset(methods)
