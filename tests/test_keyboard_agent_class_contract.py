from pathlib import Path
import ast


def test_keyboard_agent_uses_modern_class_declaration():
    text = Path('example/keyboard_agent.py').read_text()
    tree = ast.parse(text)
    random_agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RandomAgent')
    assert random_agent.bases == []
    assert 'class RandomAgent(object)' not in text


def test_keyboard_agent_act_returns_sampled_action_expression():
    tree = ast.parse(Path('example/keyboard_agent.py').read_text())
    random_agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RandomAgent')
    act = next(node for node in random_agent.body if isinstance(node, ast.FunctionDef) and node.name == 'act')
    returns = [node for node in ast.walk(act) if isinstance(node, ast.Return)]
    assert returns
    call = returns[-1].value
    assert isinstance(call, ast.Call)
