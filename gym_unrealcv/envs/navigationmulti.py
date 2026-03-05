import numpy as np
from gym_unrealcv.envs.base_env import UnrealCv_base
from gym_unrealcv.envs.utils import misc, reward

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

        self.task_config = self.load_task_setting(task_file)

        # self.cam_id = self.setting['cam_id']
        self.target_list = self.task_config.get('target_list', self.env_configs['targets']['Point'])
        if isinstance(self.target_list, str):
            self.target_list = [self.target_list]
        if len(self.target_list) == 0:
            raise ValueError('target_list in task_file should not be empty')
        self.target_id=self.target_list[0]

        self.observation_type = observation_type
        # assert self.observation_type == 'Color' or self.observation_type == 'Depth' or self.observation_type == 'Rgbd' or self.observation_type == 'Mask'
        # self.observation_space = self.unrealcv.define_observation(self.cam_id, self.observation_type, 'direct')

        # define reward type
        # distance, bbox, bbox_distance,
        # self.reward_type = reward_type
        self.reward_type = self.task_config.get('reward_type', 'distance')
        self.reward_function = reward.Reward()
        self.trigger_count = 0
        self.max_collision_steps = self.task_config.get('max_collision_steps', 100)
        self.success_distance = self.task_config.get('success_distance', 300)
        self.success_direction = self.task_config.get('success_direction', 10)

        #define shared message
        #可以自定义存储agent之间的通信信息
        self.message_list=[]


        self.count_steps = 0

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

        # get reward
        # select_target_by_distance可以根据任务setting 需要更改
        # distance, self.target_id = [self.select_target_by_distance(info['Pose'][i][:3], self.targets_pos) for i in range(len(self.player_list))]
        distance = np.array([self.unrealcv.get_distance(info['Pose'][i][:3], self.targets_pos[self.target_id],n=3) for i in range(len(self.player_list))])
        distance_min = np.min(distance)
        distance_min_id=np.argmin(distance)
        info['Target'] = self.targets_pos[self.target_id]

        info['Direction'] = [misc.get_direction(info['Pose'][i], np.array(self.targets_pos[self.target_id])) for i in range(len(self.player_list))]

        # calculate reward according to the distance to target object
        if 'distance' in self.reward_type:
            # info['Reward'] = self.reward_function.reward_distance(distance)
            relative_oir_norm = np.fabs(info['Direction']) / 90.0
            reward_norm = np.array([np.tanh(self.reward_function.reward_distance(distance[i]) - relative_oir_norm) for i in range(len(self.player_list))])
            info['Reward'] = reward_norm
        else:
            info['Reward'] = 0

        # if frequent collision detected, the episode is done and reward is -1
        if info['Collision'] > self.max_collision_steps:
            info['Reward'] = -1
            info['Done'] = True
        print(distance_min,np.fabs(info['Direction'][distance_min_id]))
        if distance_min < self.success_distance and np.fabs(info['Direction'][distance_min_id]) < self.success_direction:
            info['Success'] = True
            info['Done'] = True
            info['Reward'] = 100


        # save the trajectory
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory


        return obs, info['Reward'], info['Done'], info

    def reset(self, ):
        # double check the resetpoint, it is necessary for random reset type
        observations = super(NavigationMulti, self).reset()

        current_pose = self.unrealcv.get_pose(self.cam_id[self.protagonist_id])
        self.targets_pos = self.unrealcv.build_pose_dic(self.target_list)

        self.unrealcv.set_obj_color(self.target_list[0], (255, 255, 255))
        # state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
        observations, self.obj_poses, self.img_show = self.update_observation(self.player_list, self.cam_list, self.cam_flag, self.observation_type)

        self.trajectory = []
        self.trajectory.append(current_pose)
        self.trigger_count = 0
        self.count_steps = 0
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


