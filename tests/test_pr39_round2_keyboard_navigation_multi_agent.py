from pathlib import Path


def test_keyboard_navigation_multi_agent_uses_modern_class_declaration():
    text = Path('example/Keyboard_NavigationMultiAgent.py').read_text()
    assert 'class RandomAgent:' in text
    assert 'class RandomAgent(object)' not in text
