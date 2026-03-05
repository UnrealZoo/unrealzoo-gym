from pathlib import Path
import ast


def test_rendezvous_init_supports_reset_type():
    tree = ast.parse(Path('gym_unrealcv/envs/rendezvous.py').read_text())
    rendezvous = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Rendezvous')
    init_func = next(node for node in rendezvous.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')

    arg_map = {arg.arg: idx for idx, arg in enumerate(init_func.args.args)}
    assert 'reset_type' in arg_map

    default_idx = arg_map['reset_type'] - (len(init_func.args.args) - len(init_func.args.defaults))
    default_val = init_func.args.defaults[default_idx]
    assert isinstance(default_val, ast.Constant)
    assert default_val.value == 0

    super_calls = [
        node for node in ast.walk(init_func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == '__init__'
    ]
    assert super_calls
    forwarded = any(
        any(keyword.arg == 'reset_type' and isinstance(keyword.value, ast.Name) and keyword.value.id == 'reset_type'
            for keyword in call.keywords)
        for call in super_calls
    )
    assert forwarded
