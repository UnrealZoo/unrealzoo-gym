from pathlib import Path
import ast


def test_rescue_init_supports_reset_type():
    tree = ast.parse(Path('gym_unrealcv/envs/rescue.py').read_text())
    rescue = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Rescue')
    init_func = next(node for node in rescue.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')

    arg_map = {arg.arg: idx for idx, arg in enumerate(init_func.args.args)}
    assert 'reset_type' in arg_map

    default_idx = arg_map['reset_type'] - (len(init_func.args.args) - len(init_func.args.defaults))
    default_val = init_func.args.defaults[default_idx]
    assert isinstance(default_val, ast.Constant)
    assert default_val.value == 0

    forwarded = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and any(keyword.arg == 'reset_type' and isinstance(keyword.value, ast.Name) and keyword.value.id == 'reset_type'
                for keyword in node.keywords)
        for node in ast.walk(init_func)
    )
    assert forwarded
