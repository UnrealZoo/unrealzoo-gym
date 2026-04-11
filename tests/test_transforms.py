import numpy as np

from gym_unrealcv._gym_compat import spaces
from gym_unrealcv.envs.utils import transforms


def test_prepare_observation_color_shape():
    img_list = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]
    out = transforms.prepare_observation('Color', img_list, [], [], [])
    assert out.shape == (2, 4, 4, 3)


def test_prepare_observation_rgbd_shape():
    img_list = [np.zeros((4, 4, 3), dtype=np.uint8)]
    depth_list = [np.zeros((4, 4, 1), dtype=np.float32)]
    out = transforms.prepare_observation('Rgbd', img_list, [], depth_list, [])
    assert out.shape == (1, 4, 4, 4)


def test_prepare_observation_maskdepth_shape():
    mask_list = [np.zeros((4, 4, 3), dtype=np.uint8)]
    depth_list = [np.zeros((4, 4, 1), dtype=np.float32)]
    out = transforms.prepare_observation('MaskDepth', [], mask_list, depth_list, [])
    assert out.shape == (1, 4, 4, 4)


def test_get_cam_flag_for_colormask():
    flag = transforms.get_cam_flag('ColorMask')
    assert flag == [False, True, True, False]


def test_get_cam_flag_for_gray_and_cg():
    assert transforms.get_cam_flag('Gray')[1] is True
    assert transforms.get_cam_flag('CG')[1] is True


def test_get_cam_flag_with_overrides():
    flag = transforms.get_cam_flag('Pose', use_color=True, use_depth=True, use_cam_pose=True)
    assert flag == [True, True, False, True]


def test_action_mapping_discrete():
    player_list = ['p0']
    action_spaces = [spaces.Discrete(2)]
    agents = {'p0': {'move_action': [[1, 0], [0, 1]]}}
    actions = [1]

    move, head, animate = transforms.action_mapping(actions, player_list, action_spaces, agents)
    assert move == [[0, 1]]
    assert head == [None]
    assert animate == [None]


def test_action_mapping_continuous():
    player_list = ['p0']
    action_spaces = [spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)]
    agents = {'p0': {'move_action': [[1, 0], [0, 1]]}}
    actions = [np.array([0.5, -0.2], dtype=np.float32)]

    move, head, animate = transforms.action_mapping(actions, player_list, action_spaces, agents)
    assert np.allclose(move[0], np.array([0.5, -0.2], dtype=np.float32))
    assert head == [None]
    assert animate == [None]


def test_action_mapping_mixed():
    player_list = ['p0']
    action_spaces = [spaces.Tuple((
        spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32),
        spaces.Discrete(2),
        spaces.Discrete(2),
    ))]
    agents = {
        'p0': {
            'move_action': [[1, 0], [0, 1]],
            'head_action': [[0, 0], [1, 0]],
            'animation_action': ['stand', 'jump'],
        }
    }
    actions = [(np.array([0.2, -0.4], dtype=np.float32), np.int64(0), np.int64(1))]

    move, head, animate = transforms.action_mapping(actions, player_list, action_spaces, agents)
    assert np.allclose(move[0], np.array([0.2, -0.4], dtype=np.float32))
    assert head == [[0, 0]]
    assert animate == ['jump']
