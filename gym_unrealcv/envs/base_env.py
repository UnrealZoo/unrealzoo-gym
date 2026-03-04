from __future__ import annotations
# pyright: reportGeneralTypeIssues=false, reportMissingTypeStubs=false

import os.path
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import cv2
import numpy as np
from gym_unrealcv._gym_compat import Env as GymEnv
from gym_unrealcv._gym_compat import spaces
from gym_unrealcv.envs.utils import misc
from gym_unrealcv.envs.utils.transforms import action_mapping as map_actions
from gym_unrealcv.envs.utils.transforms import prepare_observation as build_observation
from gym_unrealcv.envs.utils.transforms import get_cam_flag as build_cam_flag
from unrealcv.launcher import RunUnreal
from gym_unrealcv.envs.agent.character import Character_API
import random
import sys
''' 
It is a base env for general purpose agent-env interaction, including single/multi-agent navigation, tracking, etc.
Observation : raw color image and depth
Action:  Discrete/Continuous
Done : define by the task wrapper
'''

# TODO: agent apis for blueprints commands
# TODO: config env by parapmeters
# TODO: maintain a general agent list
class UnrealCv_base(GymEnv):
    """
    A base environment for general purpose agent-environment interaction, including single/multi-agent navigation, tracking, etc.
    Observation: color image, depth image, rgbd image, mask image, pose
    Action: Discrete, Continuous, Mixed
    Done: defined by the task wrapper
    """
    def __init__(
        self,
        setting_file: str,
        action_type: str = 'Discrete',
        observation_type: str = 'Color',
        resolution: Tuple[int, int] = (160, 160),
        reset_type: int = 0,
    ) -> None:
        """
        Initialize the UnrealCv_base environment.

        Args:
            setting_file (str): The setting file to define the task and environments (path2binary, action space, reset area).
            action_type (str): Type of action space ('Discrete', 'Continuous').
            observation_type (str): Type of observation space ('Color', 'Depth', 'Rgbd', 'Gray').
            resolution (tuple): Resolution of the observation space.
            reset_type (int): Type of reset.
        """
        setting = misc.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        # self.max_steps = setting['max_steps']
        self.height = setting['height']
        self.cam_id = [setting['third_cam']['cam_id']]
        self.agent_configs = setting['agents']
        self.env_configs = setting["env"]
        self.agents = misc.convert_dict(self.agent_configs)
        self.reset_type = reset_type
        # TODO: it is useless.
        self.character: dict[str, list[Any]] = {
            'player': [],  # the list of player to control
            'npc': [],  # the list of Non-player character
            'freeze': [],  # the list of player that exists in the scene, but it is frozen
        }

        self.height_top_view = setting['third_cam']['height_top_view']

        # self.env_obj_list = self.env_configs[""]
        self.objects_list: list[str] = []
        self.reset_area = setting['reset_area']

        self.safe_start = setting['safe_start']
        self.interval = setting['interval']
        self.random_init = setting['random_init']
        self.start_area = self.get_start_area(self.safe_start[0], 500) # the start area of the agent, where we don't put obstacles

        self.count_eps = 0
        self.count_steps = 0

        # env configs
        self.docker = False
        self.resolution = resolution
        self.display = None
        self.use_opengl = False
        self.offscreen_rendering = False
        self.nullrhi = False
        self.gpu_id = None  # None means using the default gpu
        self.sleep_time = 5
        self.launched = False
        self.comm_mode = 'tcp'

        self.agents_category = ['player'] # the agent category we use in the env
        self.protagonist_id = 0

        # init agents
        self.player_list = list(self.agents.keys())
        self.cam_list = [self.agents[player]['cam_id'] for player in self.player_list]

        # define action space
        self.action_type = action_type
        assert self.action_type in ['Discrete', 'Continuous', 'Mixed']
        self.action_space = [self.define_action_space(self.action_type, self.agents[obj]) for obj in self.player_list]

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray', 'CG', 'Mask', 'Pose','MaskDepth','ColorMask']
        self.observation_space = [self.define_observation_space(self.cam_list[i], self.observation_type, resolution)
                                  for i in range(len(self.player_list))]

        # config unreal env
        if 'linux' in sys.platform:
            env_bin = setting['env_bin']
        elif 'darwin' in sys.platform:
            env_bin = setting['env_bin_mac']
        elif 'win' in sys.platform:
            env_bin = setting['env_bin_win']
        if 'env_map' in setting.keys():
            env_map = setting['env_map']
        else:
            env_map = None

        self.ue_binary = RunUnreal(ENV_BIN=env_bin, ENV_MAP=env_map)

    def step(self, actions: List[Any]) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            actions (list): List of actions to be performed by the agents.

        Returns:
            tuple: Observations, rewards, done flag, and additional info.
        """
        info = dict(
            Collision=0,
            Done=False,
            Reward=0.0,
            Action=actions,
            Pose=[],
            Steps=self.count_steps,
            Direction=None,
            Distance=None,
            Color=None,
            Depth=None,
            Relative_Pose=[],
            Success=False
        )
        actions2move, actions2turn, actions2animate = self.action_mapping(actions, self.player_list)
        move_cmds = [self.unrealcv.set_move_bp(obj, actions2move[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2move[i] is not None]
        head_cmds = [self.unrealcv.set_cam(obj, self.agents[obj]['relative_location'], actions2turn[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2turn[i] is not None]
        anim_cmds = [self.unrealcv.set_animation(obj, actions2animate[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2animate[i] is not None]
        self.unrealcv.batch_cmd(move_cmds+head_cmds+anim_cmds, None)
        self.count_steps += 1

        # get states
        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_list, self.cam_flag)
        self.obj_poses = obj_poses
        observations = self.prepare_observation(self.observation_type, imgs, masks, depths, obj_poses)
        self.img_show = self.prepare_img2show(self.protagonist_id, observations)

        pose_obs, relative_pose = self.get_pose_states(obj_poses)

        # prepare the info
        info['Pose'] = obj_poses
        info['Relative_Pose'] = relative_pose
        info['Pose_Obs'] = pose_obs
        info['Reward'] = np.zeros(len(self.player_list))

        return cast(np.ndarray, observations), cast(np.ndarray, info['Reward']), bool(info['Done']), info

    def reset(self) -> np.ndarray:
        if not self.launched:  # first time to launch
            self.launched = self.launch_ue_env()
            self.init_agents()
            self.init_objects()

        self.count_close = 0
        self.count_steps = 0
        self.count_eps += 1

        # stop move and disable physics
        for i, obj in enumerate(self.player_list):
            if self.agents[obj]['agent_type'] in self.agents_category:
                if not self.agents[obj]['internal_nav']:
                    # self.unrealcv.set_move_bp(obj, [0, 100])
                    # self.unrealcv.set_max_speed(obj, 100)
                    continue
                    # self.unrealcv.set_phy(obj, 1)
            elif self.agents[obj]['agent_type'] == 'drone':
                self.unrealcv.set_move_bp(obj, [0, 0, 0, 0])
                self.unrealcv.set_phy(obj, 1)

        # reset target location
        init_poses = self.sample_init_pose(self.random_init, len(self.player_list))
        for i, obj in enumerate(self.player_list):
            self.unrealcv.set_obj_location(obj, init_poses[i])
            # set view point
            self.unrealcv.set_cam(obj, self.agents[obj]['relative_location'], self.agents[obj]['relative_rotation'])

        # Match real camera assignments
        self.unrealcv.cam = self.unrealcv.get_camera_config()
        self.update_camera_assignments()
        # set global view to the top location
        self.set_topview(init_poses[self.protagonist_id], self.cam_id[0])
        # get state
        observations, self.obj_poses, self.img_show = self.update_observation(self.player_list, self.cam_list, self.cam_flag, self.observation_type)

        return cast(np.ndarray, observations)

    def close(self) -> None:
        if self.launched:
            self.unrealcv.client.disconnect()
            self.ue_binary.close()

    def render(self, mode: str = 'rgb_array', close: bool = False) -> Optional[np.ndarray]:
        """
        Show the rendered image.

        Args:
            mode (str): Mode of rendering.
            close (bool): Flag to close the rendering.

        Returns:
            np.array: Image to be rendered.
        """
        if close==True:
            self.ue_binary.close()
        return self.img_show

    def seed(self, seed: Optional[int] = None) -> None:
        np.random.seed(seed)

    def select_target_by_distance(self, current_pos, targets_pos):
        """Find the nearest target from *targets_pos* to *current_pos*.

        Args:
            current_pos (list): The current position [x, y, z, ...].
            targets_pos (dict): Mapping of target_id -> position list.

        Returns:
            tuple: ``(min_distance, target_id)``
        """
        target_id = next(iter(targets_pos))
        distance_min = self.unrealcv.get_distance(targets_pos[target_id], current_pos, 3)

        for key, target_pos in targets_pos.items():
            distance = self.unrealcv.get_distance(target_pos, current_pos, 3)
            if distance < distance_min:
                target_id = key
                distance_min = distance
        return distance_min, target_id

    def update_observation(self, player_list: List[str], cam_list: List[int], cam_flag: List[bool], observation_type: str) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        Update the observations for the agents.

        Args:
            player_list (list): List of player agents.
            cam_list (list): List of camera IDs.
            cam_flag (list): List of camera flags.
            observation_type (str): Type of observation.

        Returns:
            tuple: Updated observations, object poses, and image to show.
        """
        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(player_list, cam_list, cam_flag)
        observations = self.prepare_observation(observation_type, imgs, masks, depths, obj_poses)
        img_show = self.prepare_img2show(self.protagonist_id, observations)
        return cast(np.ndarray, observations), obj_poses, cast(np.ndarray, img_show)

    def get_start_area(self, safe_start: List[float], safe_range: int) -> List[float]:
        """
        Get the start area for the agents.

        Args:
            safe_start (list): Safe start coordinates.
            safe_range (int): Safe range value.

        Returns:
            list: Start area coordinates.
        """
        start_area = [safe_start[0]-safe_range, safe_start[0]+safe_range,
                     safe_start[1]-safe_range, safe_start[1]+safe_range]
        return start_area

    def set_topview(self, current_pose: List[float], cam_id: int) -> None:
        """
        Set the virtual camera on top of a point(current pose) to capture images from the bird's eye view.

        Args:
            current_pose (list): Current pose of the camera.
            cam_id (int): Camera ID.
        """
        cam_loc = current_pose[:3]
        cam_loc[-1] = self.height_top_view
        cam_rot = [-90, 0, 0]
        self.unrealcv.set_cam_location(cam_id, cam_loc)
        self.unrealcv.set_cam_rotation(cam_id, cam_rot)

    def get_relative(self, pose0: List[float], pose1: List[float]) -> Tuple[List[float], float, float]:
        """
        Get the relative pose between two objects, pose0 is the reference object.

        Args:
            pose0 (list): Pose of the reference object (the center of the coordinate system).
            pose1 (list): Pose of the target object.

        Returns:
            tuple: Relative observation vector, distance, and angle.
        """
        delt_yaw = pose1[4] - pose0[4]
        angle = misc.get_direction(pose0, pose1)
        distance = self.unrealcv.get_distance(pose1, pose0, 3)
        obs_vector = [np.sin(delt_yaw/180*np.pi), np.cos(delt_yaw/180*np.pi),
                      np.sin(angle/180*np.pi), np.cos(angle/180*np.pi),
                      distance]
        return obs_vector, distance, angle

    def prepare_observation(self, observation_type: str, img_list: List, mask_list: List, depth_list: List, pose_list: List) -> Optional[np.ndarray]:
        """
        Prepare the observation based on the observation type.

        Args:
            observation_type (str): Type of observation.
            img_list (list): List of images.
            mask_list (list): List of masks.
            depth_list (list): List of depth images.
            pose_list (list): List of poses.

        Returns:
            np.array: Prepared observation.
        """
        try:
            return build_observation(observation_type, img_list, mask_list, depth_list, pose_list)
        except ValueError:
            return None



    def rotate2exp(self, yaw_exp, obj, th=1):
        """
        Rotate the object to the expected yaw.

        Args:
            yaw_exp (float): Expected yaw.
            obj (str): Object name.
            th (int): Threshold value.

        Returns:
            float: Delta yaw.
        """
        yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
        delta_yaw = yaw_exp - yaw_pre
        while abs(delta_yaw) > th:
            if 'Drone' in obj:
                self.unrealcv.set_move_bp(obj, [0, 0, 0, np.clip(delta_yaw, -60, 60)/60*np.pi])
            else:
                self.unrealcv.set_move_bp(obj, [np.clip(delta_yaw, -60, 60), 0])
            yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
            delta_yaw = (yaw_exp - yaw_pre) % 360
            if delta_yaw > 180:
                delta_yaw = 360 - delta_yaw
        return delta_yaw

    def relative_metrics(self, relative_pose):
        """
        Compute the relative metrics among agents for rewards and evaluation.

        Args:
            relative_pose (np.array): Relative pose array.

        Returns:
            dict: Dictionary containing collision and average distance metrics.
        """
        info = dict()
        relative_dis = relative_pose[:, :, 0]
        relative_ori = relative_pose[:, :, 1]
        collision_mat = np.zeros_like(relative_dis)
        collision_mat[np.where(relative_dis < 100)] = 1
        collision_mat[np.where(np.fabs(relative_ori) > 45)] = 0  # collision should be at the front view
        info['collision'] = collision_mat
        info['dis_ave'] = relative_dis.mean() # average distance among players, regard as a kind of density metric

        return info

    def add_agent(self, name, loc, refer_agent):
        """
        Add a new agent to the environment.

        Args:
            name (str): Name of the new agent.
            loc (list): Location of the new agent.
            refer_agent (dict): Reference agent configuration.

        Returns:
            dict: New agent configuration.
        """
        new_dict = refer_agent.copy()
        cam_num = self.unrealcv.get_camera_num()
        self.unrealcv.new_obj(refer_agent['class_name'], name, random.sample(self.safe_start, 1)[0])
        self.player_list.append(name)
        if self.unrealcv.get_camera_num() > cam_num:
            new_dict['cam_id'] = cam_num
        else:
            new_dict['cam_id'] = -1
        self.cam_list.append(new_dict['cam_id'])
        self.unrealcv.set_obj_scale(name, refer_agent['scale'])
        self.unrealcv.set_obj_color(name, np.random.randint(0, 255, 3))
        self.unrealcv.set_random(name, 0)
        self.unrealcv.set_interval(self.interval, name)
        self.unrealcv.set_obj_location(name, loc)
        self.action_space.append(self.define_action_space(self.action_type, agent_info=new_dict))
        self.observation_space.append(self.define_observation_space(new_dict['cam_id'], self.observation_type, self.resolution))
        self.unrealcv.set_phy(name, 0)
        return new_dict

    def remove_agent(self, name):
        """
        Remove an agent from the environment.

        Args:
            name (str): Name of the agent to be removed.
        """
        # print(f'remove {name}')
        agent_index = self.player_list.index(name)
        self.player_list.remove(name)
        last_cam_list=self.cam_list
        self.cam_list = self.remove_cam(name)
        self.action_space.pop(agent_index)
        self.observation_space.pop(agent_index)
        self.unrealcv.destroy_obj(name)  # the agent is removed from the scene
        self.agents.pop(name)
        st_time=time.time()
        time.sleep(1)
        while self.unrealcv.get_camera_num() > len(last_cam_list) + 1:  # UE4 needs +1, UE5 does not
            pass

    def remove_cam(self, name):
        """
        Remove the camera associated with an agent.

        Args:
            name (str): Name of the agent.

        Returns:
            list: Updated list of camera IDs.
        """
        cam_id = self.agents[name]['cam_id']
        cam_list = []
        for obj in self.player_list:
            if self.agents[obj]['cam_id'] > cam_id and cam_id > 0:
                self.agents[obj]['cam_id'] -= 1
            cam_list.append(self.agents[obj]['cam_id'])
        return cam_list

    def define_action_space(self, action_type, agent_info):
        """
        Define the action space for an agent.

        Args:
            action_type (str): Type of action space ('Discrete', 'Continuous', 'Mixed').
            agent_info (dict): Agent configuration.

        Returns:
            gym.Space: Defined action space.
        """
        if action_type == 'Discrete':
            return spaces.Discrete(len(agent_info["move_action"]))
        elif action_type == 'Continuous':
            return spaces.Box(low=np.array(agent_info["move_action_continuous"]['low']),
                              high=np.array(agent_info["move_action_continuous"]['high']), dtype=np.float32)
        else:  # Hybrid
            move_space = spaces.Box(low=np.array(agent_info["move_action_continuous"]['low']),
                                    high=np.array(agent_info["move_action_continuous"]['high']), dtype=np.float32)
            turn_space = spaces.Discrete(2)
            animation_space = spaces.Discrete(2)
            if "head_action" in agent_info.keys():
                turn_space = spaces.Discrete(len(agent_info["head_action"]))
            if "animation_action" in agent_info.keys():
                animation_space = spaces.Discrete(len(agent_info["animation_action"]))
            return spaces.Tuple((move_space, turn_space, animation_space))

    def define_observation_space(self, cam_id, observation_type, resolution=(160, 120)):
        """
        Define the observation space for an agent.

        Args:
            cam_id (int): Camera ID.
            observation_type (str): Type of observation space.
            resolution (tuple): Resolution of the observation space.

        Returns:
            gym.Space: Defined observation space.
        """
        if observation_type == 'Pose' or cam_id < 0:
            observation_space = spaces.Box(low=-100, high=100, shape=(6,),
                                               dtype=np.float16)  # TODO check the range and shape
        else:
            if observation_type == 'Color' or observation_type == 'CG' or observation_type == 'Mask':
                img_shape = (resolution[1], resolution[0], 3)
                observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
            elif observation_type == 'Depth':
                img_shape = (resolution[1], resolution[0], 1)
                observation_space = spaces.Box(low=0, high=100, shape=img_shape, dtype=np.float16)
            elif observation_type == 'Rgbd':
                s_low = np.zeros((resolution[1], resolution[0], 4))
                s_high = np.ones((resolution[1], resolution[0], 4))
                s_high[:, :, -1] = 100.0  # max_depth
                s_high[:, :, :-1] = 255  # max_rgb
                observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float16)
            elif observation_type == 'MaskDepth':
                s_low = np.zeros((resolution[1], resolution[0], 4))
                s_high = np.ones((resolution[1], resolution[0], 4))
                s_high[:, :, -1] = 100.0  # max_depth
                s_high[:, :, :-1] = 255  # max_rgb
                observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float16)
            elif observation_type=='ColorMask':
                img_shape = (resolution[1], resolution[0], 6)
                observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        return observation_space

    def sample_init_pose(self, use_reset_area=False, num_agents=1):
        """
        Sample initial poses to reset the agents.

        Args:
            use_reset_area (bool): Flag to indicate whether to use the reset area for sampling.
            num_agents (int): Number of agents to sample poses for.

        Returns:
            list: List of sampled locations for the agents.
        """
        if num_agents > len(self.safe_start):
            use_reset_area = True
            warnings.warn('The number of agents is less than the number of pre-defined start points, random sample points from the pre-defined area instead.')
        if use_reset_area:
            locations = self.sample_from_area(self.reset_area, num_agents)  # sample from a pre-defined area
        else:
            locations = random.sample(self.safe_start, num_agents) # sample one pre-defined start point
        return locations

    def random_app(self):
        """
        Randomly assign an appearance to each agent in the player list based on their category.

        The appearance is selected from a predefined range of IDs for each category.

        Categories:
            - player: IDs from 1 to 18
            - animal: IDs from 0 to 26
        """
        app_map = {
            'player': range(1, 19),
            'animal': range(0, 27),
            'drone':range(0,1)
        }
        for obj in self.player_list:
            category = self.agents[obj]['agent_type']
            if category not in app_map.keys():
                continue
            app_id = np.random.choice(app_map[category])
            self.unrealcv.set_appearance(obj, app_id)

    def environment_augmentation(self, player_mesh=False, player_texture=False,
                                 light=False, background_texture=False,
                                 layout=False, layout_texture=False):
        """
        Randomly assign an appearance to each agent in the player list based on their category.

        The appearance is selected from a predefined range of IDs for each category.

        Categories:
            - player: IDs from 1 to 18
            - animal: IDs from 0 to 26
        """
        app_map = {
            'player': range(1, 19),
            'animal': range(0, 27),
            'drone': range(0,1)
        }
        if player_mesh:  # random human mesh
            for obj in self.player_list:
                app_id = np.random.choice(app_map[self.agents[obj]['agent_type']])
                self.unrealcv.set_appearance(obj, app_id)
        # random light and texture of the agents
        if player_texture:
            if self.env_name == 'MPRoom':  # random target texture
                for obj in self.player_list:
                    if self.agents[obj]['agent_type'] == 'player':
                        self.unrealcv.random_player_texture(obj, self.textures_list, 3)
        if light:
            self.unrealcv.random_lit(self.env_configs["lights"])

        # random the texture of the background
        if background_texture:
            self.unrealcv.random_texture(self.env_configs["backgrounds"], self.textures_list, 5)

        # random place the obstacle`
        if layout:
            self.unrealcv.clean_obstacles()
            self.unrealcv.random_obstacles(self.objects_list, self.textures_list,
                                           random.randint(5,len(self.objects_list)), self.reset_area, self.start_area, layout_texture)

    def get_pose_states(self, obj_pos):
        # get the relative pose of each agent and the absolute location and orientation of the agent
        pose_obs = []
        player_num = len(obj_pos)
        relative_pose = np.zeros((player_num, player_num, 2))
        for j in range(player_num):
            vectors = []
            for i in range(player_num):
                obs, distance, direction = self.get_relative(obj_pos[j], obj_pos[i])
                yaw = obj_pos[j][4]/180*np.pi
                # rescale the absolute location and orientation
                abs_loc = [obj_pos[i][0], obj_pos[i][1],
                           obj_pos[i][2], np.cos(yaw), np.sin(yaw)]
                obs = obs + abs_loc
                vectors.append(obs)
                relative_pose[j, i] = np.array([distance, direction])
            pose_obs.append(vectors)

        return np.array(pose_obs), relative_pose

    def launch_ue_env(self):
        # launch the UE4 binary
        env_ip, env_port = self.ue_binary.start(docker=self.docker, resolution=self.resolution, display=self.display,
                                               opengl=self.use_opengl, offscreen=self.offscreen_rendering,
                                               nullrhi=self.nullrhi,sleep_time=10)


        # connect to UnrealCV Server
        self.unrealcv = Character_API(port=env_port, ip=env_ip, resolution=self.resolution, comm_mode=self.comm_mode)
        # self.unrealcv.client.request('r.Vulkan.EnableDefrag=0')
        self.unrealcv.set_map(self.env_name)
        return True

    def init_agents(self):
        for obj in self.player_list.copy(): # the agent will be fully removed in self.agents
            if self.agents[obj]['agent_type'] not in self.agents_category:
                self.remove_agent(obj)
        for obj in self.player_list:
            self.unrealcv.set_obj_scale(obj, self.agents[obj]['scale'])
            self.unrealcv.set_random(obj, 0)
            self.unrealcv.set_interval(self.interval, obj)

        self.unrealcv.build_color_dict(self.player_list)
        self.cam_flag = self.get_cam_flag(self.observation_type)

    def init_objects(self):
        self.unrealcv.init_objects(self.objects_list)

    def prepare_img2show(self, index, states):
        if self.observation_type == 'Rgbd':
            return states[index][:, :, :3]
        elif self.observation_type in ['Color', 'Gray', 'CG', 'Mask']:
            return states[index]
        elif self.observation_type == 'Depth':
            return states[index]/states[index].max()  # normalize the depth image
        else:
            return None

    def set_population(self, num_agents):
        while len(self.player_list) < num_agents:
            refer_agent = self.agents[random.choice(list(self.agents.keys()))]
            name = f'{refer_agent["agent_type"]}_EP{self.count_eps}_{len(self.player_list)}'
            self.agents[name] = self.add_agent(name, random.choice(self.safe_start), refer_agent)
        while len(self.player_list) > num_agents:
            self.remove_agent(self.player_list[-1])  # remove the last one

    def set_npc(self):
        # TODO: set the NPC agent
        return self.player_list.index(random.choice([x for x in self.player_list if x > 0]))

    def set_agent(self):
        # the agent is controlled by the external controller
        return self.cam_list.index(random.choice([x for x in self.cam_list if x > 0]))

    def action_mapping(self, actions, player_list):
        return map_actions(actions, player_list, self.action_space, self.agents)


    def get_cam_flag(self, observation_type, use_color=False, use_mask=False, use_depth=False, use_cam_pose=False):
        return build_cam_flag(
            observation_type,
            use_color=use_color,
            use_mask=use_mask,
            use_depth=use_depth,
            use_cam_pose=use_cam_pose,
        )

    def sample_from_area(self, area, num):
        x = np.random.randint(area[0], area[1], num)
        y = np.random.randint(area[2], area[3], num)
        z = np.random.randint(area[4], area[5], num)
        return np.vstack((x, y, z)).T

    def get_startpoint(self, target_pos=[], distance=None, reset_area=[], exp_height=200, direction=None):
        for i in range(5):  # searching a safe point
            if direction == None:
                direction = 2 * np.pi * np.random.sample(1)
            else:
                direction = direction % (2 * np.pi)
            if distance == None:
                x = np.random.randint(reset_area[0], reset_area[1])
                y = np.random.randint(reset_area[2], reset_area[3])
            else:
                dx = float(distance * np.cos(direction))
                dy = float(distance * np.sin(direction))
                x = dx + target_pos[0]
                y = dy + target_pos[1]
            cam_pos_exp = [x, y, exp_height]
            if reset_area[0] < x < reset_area[1] and reset_area[2] < y < reset_area[3]:
                cam_pos_exp[0] = x
                cam_pos_exp[1] = y
                return cam_pos_exp
        return []


    def update_camera_assignments(self):
        """Update camera assignments for all agents, assigning each agent the nearest camera."""
        cam_locs = []
        for cam_id in range(0, self.unrealcv.get_camera_num()):
            cam_loc = self.unrealcv.get_cam_location(cam_id)
            cam_locs.append(cam_loc)

        for obj in self.player_list:
            obj_loc = self.unrealcv.get_obj_location(obj)
            dis_list = [self.unrealcv.get_distance(loc, obj_loc, 3) for loc in cam_locs]
            nearest_cam_id = dis_list.index(min(dis_list))
            self.agents[obj]['cam_id'] = nearest_cam_id
            agent_idx = self.player_list.index(obj)
            self.cam_list[agent_idx] = nearest_cam_id