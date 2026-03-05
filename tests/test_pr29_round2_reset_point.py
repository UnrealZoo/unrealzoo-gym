from pathlib import Path
import ast


def test_reset_point_uses_modern_class_declaration():
    text = Path('gym_unrealcv/envs/utils/reset_point.py').read_text()
    tree = ast.parse(text)
    reset_point = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ResetPoint')

    assert reset_point.bases == []
    assert 'class ResetPoint()' not in text


def test_reset_point_init_signature_keeps_required_parameters():
    tree = ast.parse(Path('gym_unrealcv/envs/utils/reset_point.py').read_text())
    reset_point = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ResetPoint')
    init_func = next(node for node in reset_point.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    arg_names = [arg.arg for arg in init_func.args.args]
    assert arg_names[:4] == ['self', 'setting', 'type', 'init_pose']
