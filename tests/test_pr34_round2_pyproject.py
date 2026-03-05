import tomllib
from pathlib import Path


def test_pyproject_declares_readme_metadata():
    data = tomllib.loads(Path('pyproject.toml').read_text())
    assert data['project']['readme'] == 'README.md'
