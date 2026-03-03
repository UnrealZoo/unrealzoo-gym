import pytest

from gym_unrealcv.envs.utils.misc import validate_env_setting


def valid_setting():
    return {
        'env_name': 'Demo_Roof',
        'agents': {
            'player': {
                'name': ['agent_0'],
                'cam_id': [0],
                'class_name': ['bp_character_C'],
                'scale': [1.0, 1.0, 1.0],
            }
        },
        'interval': 0.1,
        'height': 100,
        'third_cam': {'cam_id': 0, 'cam_view_id': 0, 'location': [0, 0, 300], 'rotation': [0, 0, 0]},
        'safe_start': [[0, 0, 100, 0, 0, 0]],
        'reset_area': [-100, 100, -100, 100, 0, 100],
        'env_bin': 'Fake/Binary',
    }


def test_validate_env_setting_passes_with_minimal_valid_config():
    setting = valid_setting()
    validated = validate_env_setting(setting, filename='test.json')
    assert validated['env_name'] == 'Demo_Roof'


def test_validate_env_setting_missing_required_key_raises():
    setting = valid_setting()
    del setting['interval']
    with pytest.raises(KeyError):
        validate_env_setting(setting, filename='test.json')


def test_validate_env_setting_invalid_agents_type_raises():
    setting = valid_setting()
    setting['agents'] = []
    with pytest.raises(TypeError):
        validate_env_setting(setting, filename='test.json')


def test_validate_env_setting_inconsistent_agent_list_lengths_raises():
    setting = valid_setting()
    setting['agents']['player']['cam_id'] = [0, 1]
    with pytest.raises(ValueError):
        validate_env_setting(setting, filename='test.json')


def test_validate_env_setting_missing_env_bin_keys_raises():
    setting = valid_setting()
    del setting['env_bin']
    with pytest.raises(KeyError):
        validate_env_setting(setting, filename='test.json')


def test_validate_env_setting_env_bin_mac_without_env_bin_passes():
    setting = valid_setting()
    del setting['env_bin']
    setting['env_bin_mac'] = 'Fake/MacBinary'

    validated = validate_env_setting(setting, filename='test.json')
    assert validated['env_bin_mac'] == 'Fake/MacBinary'
