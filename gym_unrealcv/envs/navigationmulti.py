import warnings
import numpy as np
from gym_unrealcv.envs.base_env import UnrealCv_base
from gym_unrealcv.envs.utils import misc, reward
from gym_unrealcv.envs.navigation import NAV_TARGETS_EMPTY_MSG

'''
It is a general env for multi agent to collabrate to find a target object.

State : raw color image and depth (640x480) 
Action:  (linear velocity ,angle velocity , trigger) 
Done : Collision or get target place or False trigger three times.
Task: Learn to avoid obstacle and search for a target object, 
      you can select the target name according to the Recommend object list in setting files

'''


class NavigationMulti(UnrealCv_base):
    def __init__(self,
                 env_file,  # the setting file to define the task
                 task_file=None,  # the file to define the task
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 resolution=(160, 160),
                 reset_type=0
                 ):
        super(NavigationMulti, self).__init__(setting_file=env_file,  # the setting file to define the task
                                    action_type=action_type,  # 'discrete', 'continuous'
                                    observation_type=observation_type,  # 'color', 'depth', 'rgbd', 'Gray'
                                    resolution=resolution,
                                    reset_type=reset_type)


        # self.cam_id = self.setting['cam_id']
        self.target_list = self.env_configs['targets']['Point']
        self.target_id = self.target_list[0] if self.target_list else None
        if not self.target_list:
            warnings.warn(NAV_TARGETS_EMPTY_MSG, UserWarning, stacklevel=2)

        self.observation_type = observation_type
        # assert self.observation_type == 'Color' or self.observation_type == 'Depth' or self.observation_type == 'Rgbd' or self.observation_type == 'Mask'
        # self.observation_space = self.unrealcv.define_observation(self.cam_id, self.observation_type, 'direct')

        # define reward type
        # distance, bbox, bbox_distance,
        # self.reward_type = reward_type
        self.reward_type = 'distance'
        self.reward_function = reward.Reward()
        self.trigger_count = 0

        #define shared message
        #可以自定义存储agent之间的通信信息
        self.message_list=[]


        self.count_steps = 0

    def set_navigation_targets(self, point_names):
        """Set Point target actor names (UE object names). Use before reset or after connect; empty list clears targets."""
        if isinstance(point_names, str):
            point_names = [point_names]
        self.target_list = list(point_names)
        self.env_configs['targets']['Point'] = list(point_names)
        self.target_id = self.target_list[0] if self.target_list else None
        if not self.target_list:
            self.targets_pos = {}
            return
        if getattr(self, 'unrealcv', None) is not None and getattr(self.unrealcv, 'client', None):
            self.targets_pos = self.unrealcv.build_pose_dic(self.target_list)
            self.unrealcv.set_obj_color(self.target_list[0], (255, 255, 255))

    def step(self, action):
        obs, rewards, done, info = super(NavigationMulti, self).step(action)

        #detect if any agent collision with environment
        # if sum([self.unrealcv.get_hit(self.player_list[i]) for i in range(len(self.player_list))]) == 0:
        if self.unrealcv.get_hit(self.player_list[0])== 0: #目前无人机还没有get_hit函数，后期添加，当前仅判断human character碰撞
            info['Collision'] = 0
        else:
            info['Collision'] += 1


        #get all agent's pose
        info['Pose'] = np.array([self.unrealcv.get_obj_pose(self.player_list[i]) for i in range(len(self.player_list))])


        # the robot think that it found the target object,the episode is done
        # and get a reward by bounding box size
        # only three times false trigger allowed in every episode
        # if info['Trigger'] > self.trigger_th:
        #     self.trigger_count += 1
        #     # get reward
        #     if 'bbox' in self.reward_type:
        #         object_mask = self.unrealcv.read_image(self.cam_id, 'object_mask')
        #         boxes = self.unrealcv.get_bboxes(object_mask, self.target_list)
        #         info['Reward'], info['Bbox'] = self.reward_function.reward_bbox(boxes)
        #     else:
        #         info['Reward'] = 0
        #
        #     if info['Reward'] > 0 or self.trigger_count > 3:
        #         info['Done'] = True
        #         if info['Reward'] > 0 and self.reset_type == 'waypoint':
        #             self.reset_module.success_waypoint(self.count_steps)
        # else:

        if self.target_list and self.targets_pos and self.target_id is not None:
            distance = np.array([self.unrealcv.get_distance(info['Pose'][i][:3], self.targets_pos[self.target_id], n=3) for i in range(len(self.player_list))])
            distance_min = np.min(distance)
            distance_min_id = np.argmin(distance)
            info['Target'] = self.targets_pos[self.target_id]
            info['Direction'] = [misc.get_direction(info['Pose'][i], np.array(self.targets_pos[self.target_id])) for i in range(len(self.player_list))]

            if 'distance' in self.reward_type:
                relative_oir_norm = np.fabs(info['Direction']) / 90.0
                reward_norm = np.array([np.tanh(self.reward_function.reward_distance(distance[i]) - relative_oir_norm) for i in range(len(self.player_list))])
                info['Reward'] = reward_norm
            else:
                info['Reward'] = 0

            if info['Collision'] > 100:
                info['Reward'] = -1
                info['Done'] = True
            if distance_min < 300 and np.fabs(info['Direction'][distance_min_id]) < 10:
                info['Success'] = True
                info['Done'] = True
                info['Reward'] = 100
        else:
            info['Target'] = None
            info['Direction'] = [0.0 for _ in self.player_list]
            info['Reward'] = np.zeros(len(self.player_list))
            if info['Collision'] > 100:
                info['Reward'] = -1 * np.ones(len(self.player_list))
                info['Done'] = True


        # save the trajectory
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory


        return obs, info['Reward'], info['Done'], info

    def reset(self, ):
        # double check the resetpoint, it is necessary for random reset type
        observations = super(NavigationMulti, self).reset()

        current_pose = self.unrealcv.get_pose(self.cam_id[self.protagonist_id])
        if self.target_list:
            self.targets_pos = self.unrealcv.build_pose_dic(self.target_list)
            self.unrealcv.set_obj_color(self.target_list[0], (255, 255, 255))
            self.target_id = self.target_list[0]
        else:
            self.targets_pos = {}
            self.target_id = None
        # state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
        observations, self.obj_poses, self.img_show = self.update_observation(self.player_list, self.cam_list, self.cam_flag, self.observation_type)

        self.trajectory = []
        self.trajectory.append(current_pose)
        self.trigger_count = 0
        self.count_steps = 0
        if self.targets_pos:
            self.reward_function.dis2target_initial, self.targetID_last = \
                self.select_target_by_distance(current_pose, self.targets_pos)

        return observations

    def seed(self, seed=None):
        return seed

    def render(self, mode='rgb_array', close=False):
        if close:
            self.unreal.close()
        return self.unrealcv.img_color

    def close(self):
        self.unrealcv.client.disconnect()
        self.ue_binary.close()

    def get_action_size(self):
        return len(self.action)

    def select_target_by_distance(self, current_pos, targets_pos):
        # find the nearest target, return distance and targetid
        target_id = list(self.targets_pos.keys())[0]
        # distance_min = self.unrealcv.get_distance(targets_pos[target_id], current_pos, 2)
        distance_min = self.unrealcv.get_distance(targets_pos[target_id], current_pos, 3)

        for key, target_pos in targets_pos.items():
            # distance = self.unrealcv.get_distance(target_pos, current_pos, 2)
            distance = self.unrealcv.get_distance(target_pos, current_pos, 3)
            if distance < distance_min:
                target_id = key
                distance_min = distance
        return distance_min, target_id


