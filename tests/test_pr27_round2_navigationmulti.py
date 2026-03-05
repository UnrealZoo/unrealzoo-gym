from pathlib import Path
import ast


def test_navigationmulti_uses_modern_super_call():
    text = Path('gym_unrealcv/envs/navigationmulti.py').read_text()
    tree = ast.parse(text)
    navigation_multi = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NavigationMulti')
    init_func = next(node for node in navigation_multi.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')

    assert 'super(NavigationMulti, self).__init__(' not in text
    super_calls = [
        node for node in ast.walk(init_func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == 'super'
    ]
    assert super_calls


def test_navigationmulti_init_exposes_reset_type_parameter():
    tree = ast.parse(Path('gym_unrealcv/envs/navigationmulti.py').read_text())
    navigation_multi = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NavigationMulti')
    init_func = next(node for node in navigation_multi.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    arg_names = [arg.arg for arg in init_func.args.args]
    assert 'reset_type' in arg_names
