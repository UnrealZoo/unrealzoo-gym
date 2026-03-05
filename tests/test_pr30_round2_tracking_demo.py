from pathlib import Path


def test_tracking_demo_import_cleanup_is_preserved():
    text = Path('example/tracking_demo.py').read_text()
    assert 'from gym import wrappers' not in text
