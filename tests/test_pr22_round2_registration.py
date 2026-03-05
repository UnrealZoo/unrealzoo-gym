from pathlib import Path


def test_unrealagent_registration_format_has_no_target_kwarg():
    text = Path('gym_unrealcv/__init__.py').read_text()
    assert "UnrealAgent-{env}-{action}{obs}-v{reset}" in text
    assert "target=target" not in text
