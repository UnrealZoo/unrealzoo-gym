from pathlib import Path


def test_addenv_doc_uses_current_project_name():
    text = Path('doc/addEnv.md').read_text()
    assert 'unrealzoo-gym' in text
    assert 'gym-unrealzoo' not in text
