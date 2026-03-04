"""Tests for typed data structures in gym_unrealcv.envs.utils.types."""
import pytest

from gym_unrealcv.envs.utils.types import AgentConfig, EnvSettings


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig(agent_type='player')
        assert cfg.agent_type == 'player'
        assert cfg.cam_id == -1
        assert cfg.internal_nav is False
        assert cfg.move_action is None
        assert cfg.extra == {}

    def test_from_dict_known_keys(self):
        raw = {
            'agent_type': 'animal',
            'name': 'deer_0',
            'cam_id': 3,
            'class_name': 'animal_C',
            'scale': [1, 1, 1],
            'internal_nav': True,
            'relative_location': [0, 50, 80],
            'relative_rotation': [10, 0, 0],
            'move_action': [[0, 100], [30, 100]],
        }
        cfg = AgentConfig.from_dict(raw)
        assert cfg.agent_type == 'animal'
        assert cfg.name == 'deer_0'
        assert cfg.cam_id == 3
        assert cfg.move_action == [[0, 100], [30, 100]]
        assert cfg.extra == {}

    def test_from_dict_extra_keys(self):
        raw = {
            'agent_type': 'drone',
            'custom_field': 42,
            'another_unknown': 'hello',
        }
        cfg = AgentConfig.from_dict(raw)
        assert cfg.agent_type == 'drone'
        assert cfg.extra == {'custom_field': 42, 'another_unknown': 'hello'}

    def test_to_dict_roundtrip(self):
        raw = {
            'agent_type': 'player',
            'name': 'player_0',
            'cam_id': 1,
            'internal_nav': False,
            'bonus': 99,
        }
        cfg = AgentConfig.from_dict(raw)
        d = cfg.to_dict()
        assert d['agent_type'] == 'player'
        assert d['name'] == 'player_0'
        assert d['bonus'] == 99  # extras survive round-trip


class TestEnvSettings:
    def _sample_dict(self):
        return {
            'env_name': 'Demo_Roof',
            'height': 200,
            'third_cam': {'cam_id': 0, 'height_top_view': 1500},
            'agents': {'player': {'name': ['p0']}},
            'env': {'targets': {}},
            'reset_area': [0, 1000, 0, 1000, 0, 500],
            'safe_start': [[100, 200, 0]],
            'interval': 0.5,
            'random_init': True,
            'env_bin': '/path/to/bin',
        }

    def test_from_dict(self):
        s = EnvSettings.from_dict(self._sample_dict())
        assert s.env_name == 'Demo_Roof'
        assert s.third_cam_id == 0
        assert s.height_top_view == 1500
        assert s.random_init is True

    def test_frozen(self):
        s = EnvSettings.from_dict(self._sample_dict())
        with pytest.raises(AttributeError):
            setattr(s, 'env_name', 'Other')

    def test_optional_env_map(self):
        d = self._sample_dict()
        d['env_map'] = 'CustomMap'
        s = EnvSettings.from_dict(d)
        assert s.env_map == 'CustomMap'

    def test_missing_env_map_defaults_none(self):
        s = EnvSettings.from_dict(self._sample_dict())
        assert s.env_map is None
