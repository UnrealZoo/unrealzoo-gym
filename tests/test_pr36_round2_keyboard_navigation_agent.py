from pathlib import Path


def test_keyboard_navigation_agent_uses_modern_class_declaration():
    text = Path('example/Keyboard_NavigationAgent.py').read_text()
    assert 'class RandomAgent:' in text
    assert 'class RandomAgent(object)' not in text
