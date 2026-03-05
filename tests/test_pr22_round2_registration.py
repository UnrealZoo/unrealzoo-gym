from pathlib import Path
import re


def test_unrealagent_registration_format_has_no_target_kwarg():
    text = Path('gym_unrealcv/__init__.py').read_text()
    lines = re.findall(r"name\s*=\s*'UnrealAgent-\{env\}-\{action\}\{obs\}-v\{reset\}'.*", text)
    assert lines, 'Expected at least one UnrealAgent registration line'

    for line in lines:
        assert "format(" in line
        assert "env=env" in line
        assert "action=action" in line
        assert "obs=obs" in line
        assert "reset=i" in line
        assert "target=target" not in line
