import numbers
from typing import Any, Sequence, Tuple

import numpy as np

from gym_unrealcv._gym_compat import spaces


def prepare_observation(
    observation_type: str,
    img_list: Sequence[Any],
    mask_list: Sequence[Any],
    depth_list: Sequence[Any],
    pose_list: Sequence[Any],
) -> np.ndarray:
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


def action_mapping(
    actions: Sequence[Any],
    player_list: Sequence[str],
    action_spaces: Sequence[Any],
    agents: dict[str, dict[str, Any]],
) -> Tuple[list[Any], list[Any], list[Any]]:
    actions2move: list[Any] = []
    actions2animate: list[Any] = []
    actions2head: list[Any] = []

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


def get_cam_flag(
    observation_type: str,
    use_color: bool = False,
    use_mask: bool = False,
    use_depth: bool = False,
    use_cam_pose: bool = False,
) -> list[bool]:
    flag = [False, False, False, False]
    flag[0] = use_cam_pose
    flag[1] = observation_type in ['Color', 'Rgbd', 'ColorMask', 'Gray', 'CG'] or use_color
    flag[2] = observation_type in ['Mask', 'MaskDepth', 'ColorMask'] or use_mask
    flag[3] = observation_type in ['Depth', 'Rgbd', 'MaskDepth'] or use_depth
    return flag
