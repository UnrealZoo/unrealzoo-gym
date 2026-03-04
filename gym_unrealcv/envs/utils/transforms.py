import numpy as np
import numbers
from gym_unrealcv._gym_compat import spaces


def prepare_observation(observation_type, img_list, mask_list, depth_list, pose_list):
    if observation_type == 'Depth':
        return np.array(depth_list)
    if observation_type == 'Mask':
        return np.array(mask_list)
    if observation_type in ['Color', 'Gray', 'CG']:
        return np.array(img_list)
    if observation_type == 'Rgbd':
        return np.append(np.array(img_list), np.array(depth_list), axis=-1)
    if observation_type == 'Pose':
        return np.array(pose_list)
    if observation_type == 'MaskDepth':
        return np.append(np.array(mask_list), np.array(depth_list), axis=-1)
    if observation_type == 'ColorMask':
        return np.append(np.array(img_list), np.array(mask_list), axis=-1)
    raise ValueError('Unknown observation type: {}'.format(observation_type))


def action_mapping(actions, player_list, action_spaces, agents):
    actions2move = []
    actions2animate = []
    actions2head = []

    for i, obj in enumerate(player_list):
        action_space = action_spaces[i]
        act = actions[i]
        if act is None:
            actions2move.append(None)
            actions2animate.append(None)
            actions2head.append(None)
            continue

        if isinstance(action_space, spaces.Discrete):
            actions2move.append(agents[obj]['move_action'][act])
            actions2animate.append(None)
            actions2head.append(None)
        elif isinstance(action_space, spaces.Box):
            actions2move.append(act)
            actions2animate.append(None)
            actions2head.append(None)
        elif isinstance(action_space, spaces.Tuple):
            for j, action in enumerate(actions[i]):
                if j == 0:
                    if isinstance(action, numbers.Integral):
                        actions2move.append(agents[obj]['move_action'][action])
                    else:
                        actions2move.append(action)
                elif j == 1:
                    if isinstance(action, numbers.Integral):
                        actions2head.append(agents[obj]['head_action'][action])
                    else:
                        actions2head.append(action)
                elif j == 2:
                    actions2animate.append(agents[obj]['animation_action'][action])

    return actions2move, actions2head, actions2animate


def get_cam_flag(observation_type, use_color=False, use_mask=False, use_depth=False, use_cam_pose=False):
    flag = [False, False, False, False]
    flag[0] = use_cam_pose
    flag[1] = observation_type in ['Color', 'Rgbd', 'ColorMask', 'Gray', 'CG'] or use_color
    flag[2] = observation_type in ['Mask', 'MaskDepth', 'ColorMask'] or use_mask
    flag[3] = observation_type in ['Depth', 'Rgbd', 'MaskDepth'] or use_depth
    return flag
