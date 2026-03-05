from pathlib import Path
import ast


def test_tracking_demo_import_cleanup_is_preserved():
    text = Path('example/tracking_demo.py').read_text()
    tree = ast.parse(text)

    import_from_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert 'gym' not in import_from_modules

    plain_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert 'gym_unrealcv' in plain_imports


def test_tracking_demo_keeps_side_effect_registration_comment():
    text = Path('example/tracking_demo.py').read_text()
    assert '# noqa: F401' in text
