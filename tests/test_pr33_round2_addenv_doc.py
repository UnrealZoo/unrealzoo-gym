from pathlib import Path


def test_addenv_doc_uses_current_project_name():
    text = Path('doc/addEnv.md').read_text()
    assert 'unrealzoo-gym' in text
    assert 'gym-unrealzoo' not in text


def test_addenv_doc_first_paragraph_mentions_current_name_once():
    text = Path('doc/addEnv.md').read_text()
    first_section = text.splitlines()[:6]
    first_block = '\n'.join(first_section)
    assert first_block.count('unrealzoo-gym') == 1
