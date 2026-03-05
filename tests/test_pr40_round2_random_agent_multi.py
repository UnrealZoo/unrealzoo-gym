from pathlib import Path
import ast


def test_random_agent_multi_uses_setdefault_for_unrealenv():
    text = Path('example/random_agent_multi.py').read_text()
    tree = ast.parse(text)

    setdefault_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'setdefault'
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == 'os'
        and node.func.value.attr == 'environ'
    ]
    assert setdefault_calls
    assert any(isinstance(call.args[0], ast.Constant) and call.args[0].value == 'UnrealEnv' for call in setdefault_calls)

    assert "os.environ['UnrealEnv']=" not in text
