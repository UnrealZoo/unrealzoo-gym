import json
from pathlib import Path


def test_track_train_config_contains_random_init_key():
    data = json.loads(Path('gym_unrealcv/envs/setting/Track/track_train.json').read_text())
    assert 'random_init' in data


def test_track_train_random_init_has_boolean_type_and_stable_defaults():
    data = json.loads(Path('gym_unrealcv/envs/setting/Track/track_train.json').read_text())
    assert isinstance(data['random_init'], bool)
    assert data['random_init'] is False
    assert isinstance(data.get('interval'), int)
    assert data['interval'] > 0
