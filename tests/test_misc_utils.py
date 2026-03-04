import json
from pathlib import Path

import pytest

from gym_unrealcv.envs.utils import misc


def test_load_env_setting_rejects_non_json() -> None:
    with pytest.raises(ValueError):
        misc.load_env_setting('Demo_Roof.yaml')


def test_load_env_setting_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing_path = tmp_path / 'missing.json'
    monkeypatch.setattr(misc, 'get_settingpath', lambda filename: str(missing_path))

    with pytest.raises(FileNotFoundError):
        misc.load_env_setting('missing.json')


def test_load_env_setting_reads_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    setting_path = tmp_path / 'demo.json'
    payload = {'env_name': 'Demo_Roof', 'height': 100}
    setting_path.write_text(json.dumps(payload), encoding='utf-8')
    monkeypatch.setattr(misc, 'get_settingpath', lambda filename: str(setting_path))

    setting = misc.load_env_setting('demo.json')

    assert setting == payload


def test_get_textures_missing_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr('unrealcv.util.get_path2UnrealEnv', lambda: str(tmp_path))

    with pytest.raises(FileNotFoundError):
        misc.get_textures('textures')


def test_get_textures_returns_absolute_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    texture_root = tmp_path / 'custom_textures'
    texture_root.mkdir(parents=True)
    (texture_root / 'a.png').write_text('a', encoding='utf-8')
    (texture_root / 'b.png').write_text('b', encoding='utf-8')
    monkeypatch.setattr('unrealcv.util.get_path2UnrealEnv', lambda: str(tmp_path))

    textures = misc.get_textures('custom_textures')

    assert sorted(Path(path).name for path in textures) == ['a.png', 'b.png']
    assert all(Path(path).is_absolute() for path in textures)
