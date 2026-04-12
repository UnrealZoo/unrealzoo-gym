from pathlib import Path
import ast


def test_navigation_has_no_redundant_observation_type_assignment():
    text = Path('gym_unrealcv/envs/navigation.py').read_text()
    tree = ast.parse(text)

    navigation = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Navigation')
    init_func = next(node for node in navigation.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')

    redundant_assignments = [
        node
        for node in ast.walk(init_func)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == 'self'
            and target.attr == 'observation_type'
            for target in node.targets
        )
    ]
    assert redundant_assignments == []


def test_navigation_init_keeps_observation_type_parameter():
    tree = ast.parse(Path('gym_unrealcv/envs/navigation.py').read_text())
    navigation = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Navigation')
    init_func = next(node for node in navigation.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')

    arg_names = [arg.arg for arg in init_func.args.args]
    assert 'observation_type' in arg_names
