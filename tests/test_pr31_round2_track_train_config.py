import json
from pathlib import Path


def test_track_train_config_contains_random_init_key():
    data = json.loads(Path('gym_unrealcv/envs/setting/Track/track_train.json').read_text())
    assert 'random_init' in data
