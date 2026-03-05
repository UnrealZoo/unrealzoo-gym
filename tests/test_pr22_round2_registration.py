from pathlib import Path
import re


def test_unrealagent_registration_format_has_no_target_kwarg():
    text = Path('gym_unrealcv/__init__.py').read_text()
    match = re.search(r"name\s*=\s*'UnrealAgent-\{env\}-\{action\}\{obs\}-v\{reset\}'.*", text)
    assert match is not None
    assert "target=target" not in match.group(0)
