import tomllib
from pathlib import Path


def test_pyproject_declares_readme_metadata():
    data = tomllib.loads(Path('pyproject.toml').read_text())
    assert data['project']['readme'] == 'README.md'


def test_pyproject_project_dependencies_remain_non_empty():
    data = tomllib.loads(Path('pyproject.toml').read_text())
    deps = data['project']['dependencies']
    assert isinstance(deps, list)
    assert len(deps) > 0
    assert any(dep.startswith('gym') for dep in deps)
