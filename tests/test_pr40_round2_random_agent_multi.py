from pathlib import Path


def test_random_agent_multi_uses_setdefault_for_unrealenv():
    text = Path('example/random_agent_multi.py').read_text()
    assert "os.environ.setdefault('UnrealEnv'" in text
    assert "os.environ['UnrealEnv']=" not in text
