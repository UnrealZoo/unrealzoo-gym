from pathlib import Path


def test_keyboard_agent_uses_modern_class_declaration():
    text = Path('example/keyboard_agent.py').read_text()
    assert 'class RandomAgent:' in text
    assert 'class RandomAgent(object)' not in text
