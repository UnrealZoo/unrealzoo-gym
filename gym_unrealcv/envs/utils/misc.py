import os
import numpy as np
import json
import unrealcv


def validate_env_setting(setting, filename=''):
    required_keys = ['env_name', 'agents', 'interval', 'height', 'third_cam', 'safe_start', 'reset_area']
    missing_keys = [key for key in required_keys if key not in setting]
    if missing_keys:
        raise KeyError(f'Missing required keys in {filename}: {missing_keys}')

    if not isinstance(setting['agents'], dict) or len(setting['agents']) == 0:
        raise TypeError(f'Invalid agents in {filename}: expected non-empty dict')

    for agent_type, info in setting['agents'].items():
        for key in ['name', 'cam_id', 'class_name']:
            if key not in info:
                raise KeyError(f'Missing agents.{agent_type}.{key} in {filename}')
            if not isinstance(info[key], list):
                raise TypeError(f'Invalid agents.{agent_type}.{key} in {filename}: expected list')
        n = len(info['name'])
        if len(info['cam_id']) != n or len(info['class_name']) != n:
            raise ValueError(f'Inconsistent list lengths for agents.{agent_type} in {filename}')

    if not isinstance(setting['safe_start'], list) or len(setting['safe_start']) == 0:
        raise TypeError(f'Invalid safe_start in {filename}: expected non-empty list')
    if not isinstance(setting['reset_area'], list) or len(setting['reset_area']) < 4:
        raise TypeError(f'Invalid reset_area in {filename}: expected list with at least 4 values')

    if not any(key in setting for key in ['env_bin', 'env_bin_mac', 'env_bin_win']):
        raise KeyError(f'Missing env binary path in {filename}: one of env_bin/env_bin_mac/env_bin_win is required')


def load_env_setting(filename):
    ext = os.path.splitext(filename)[1]
    if ext not in ['.json']:
        raise ValueError(ext + ' is not supported')
    with open(get_settingpath(filename)) as f:
        setting = json.load(f)
    return validate_env_setting(setting, filename=filename)


def get_settingpath(filename):
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists(filename):
        return filename
    import gym_unrealcv
    gympath = os.path.dirname(gym_unrealcv.__file__)
    return os.path.join(gympath, 'envs', 'setting', filename)

def get_action_size(action):
    return len(action)

def get_direction(current_pose, target_pose):  # get relative angle between current pose and target pose in x-y plane
    y_delt = target_pose[1] - current_pose[1]
    x_delt = target_pose[0] - current_pose[0]
    if x_delt == 0 and y_delt == 0:  # if the same position
        return 0
    angle_abs = np.arctan2(y_delt, x_delt)/np.pi*180
    angle_relative = angle_abs - current_pose[4]
    if angle_relative > 180:
        angle_relative -= 360
    if angle_relative < -180:
        angle_relative += 360
    return angle_relative


def get_textures(texture_name="textures", docker=False):
    try:
        texture_dir = os.path.join(unrealcv.util.get_path2UnrealEnv(), "textures")
    except AttributeError:
        raise ImportError(
            "Function get_path2UnrealEnv() not found. "
            "Please upgrade unrealcv to version 1.1.5 or higher using: \n"
            "pip install --upgrade unrealcv"
            )
    textures_list = os.listdir(texture_dir)
    # relative to abs
    for i in range(len(textures_list)):
        if docker:
            textures_list[i] = os.path.join('/unreal', texture_dir, textures_list[i])
        else:
            textures_list[i] = os.path.join(texture_dir, textures_list[i])
    return textures_list

def convert_dict(old_dict):
    new_dict = {}
    for agent, info in old_dict.items():
        names = info["name"]
        for i, name in enumerate(names):
            new_dict[name] = {
                "agent_type": agent,
            }
            for key in info.keys():
                if key == "name" or key == "cam_id" or key == "class_name":
                    new_dict[name][key] = info[key][i]
                else:
                    new_dict[name][key] = info[key]
    return new_dict
